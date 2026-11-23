"""Opt-in workbench around existing secondary-motion CLIs; body path stays separate."""
from pathlib import Path
import copy,json,os,re,subprocess,sys,threading,time,uuid,queue,zipfile
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse,StreamingResponse,PlainTextResponse
from pydantic import BaseModel,ConfigDict,StrictBool,Field
from .secondary_cache import atomic_json,read_json,file_sha,key_for,process_identity
from .secondary_tasks import RunSession
from .blender_plugins import mmd_settings,mmd_command

ACTIVE={'queued','processing','cancelling'}
def valid_preview(path,renderer_sha,source_sha):
 try:
  proof=read_json(Path(path)/'preview.json')
  return proof.get('passed') is True and proof.get('renderer_sha256')==renderer_sha and proof.get('source_sha256')==source_sha
 except (OSError,ValueError):return False
class StartRequest(BaseModel):
 model_config=ConfigDict(extra='forbid')
 source_id:str
 profile:str='rin'
 enabled:StrictBool=False
 mode:str='hair'
 preview:StrictBool=True
class ExportRequest(BaseModel):
 model_config=ConfigDict(extra='forbid')
 formats:list[str]=Field(default_factory=lambda:['blender','skeleton'])
 experimental:StrictBool=False

class Workbench:
 def __init__(self,root,data,pool):
  self.root=Path(root).resolve();self.store=Path(data).resolve()/'secondary-jobs';self.pool=pool
  self.store.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock();self.events={};self.runs={}
  for p in self.store.glob('*/job.json'):
   try:
    job=read_json(p)
    if not re.fullmatch('[a-f0-9]{32}',job['id']) or p.parent.name!=job['id']:continue
    if job['status'] in ACTIVE:job.update(status='interrupted',error='上次工作台运行中断，可恢复；已有产物保留。');atomic_json(p,job)
    if job.get('kind')=='surface' and job.get('result'):
     audit_path=Path(job['result']['pair']).resolve()/'surface-audit.json'
     if audit_path.is_relative_to(Path(job['output']).resolve()) and audit_path.is_file():
      audit=read_json(audit_path);job['physics_quality']='accepted' if audit.get('collision_quality_passed') is True else 'rejected'
      job['selection_gate_passed']=bool(read_json(Path(job['output'])/'latest.json')['quality']['selected']);atomic_json(p,job)
    self.runs[job['id']]=job
   except (ValueError,OSError,KeyError):continue

 def profiles(self):
  result=[]
  for path in sorted((self.root/'profiles/secondary_motion').glob('*.json')):
   try:
    p=read_json(path);model=(self.root/p['model']['path']).resolve()
    ready=model.is_relative_to(self.root) and model.is_file() and file_sha(model)==p['model']['sha256']
    surface=ready and bool(p.get('surface_dynamics'));surface_reason='';segmentation=None;regression=None
    if surface and (p.get('surface_segmentation',{}).get('required_for_surface_dynamics') or 'secondary_contact_rules' in p):
     try:
      from .secondary_plan import compile_profile
      from .secondary_segmentation import require_review
      plan=compile_profile(p,self.root);segmentation=plan.get('surface_segmentation',{}).get('review_gate')
      from .secondary_regression import regression_readiness
      regression=regression_readiness(plan);require_review(plan)
     except (ValueError,OSError) as exc:surface=False;surface_reason=str(exc)
    result.append({'id':p['id'],'label':p.get('display_name',p['id']),'ready':ready,
     'mmd_tools':ready and p['id']=='rin', 'hair':ready and bool(p.get('hair_dynamics')),'surface':surface,
     'surface_reason':surface_reason,'surface_segmentation':segmentation,'regression_preflight':regression,
     'reason':'' if ready else '模型文件缺失或指纹改变，须重新配置。',
     'note':'实验衣发；主动作独立保护，完整碰撞尚未验收。'})
   except (KeyError,ValueError,OSError):continue
  return result

 def sources(self):
  result=[];registry=self.root/'profiles/secondary_sources.json'
  entries=read_json(registry).get('sources',[]) if registry.is_file() else []
  # Newly generated Rin character jobs become inputs without editing a list.
  for p in self.store.parent.glob('*/outputs/Rin_MotionBridge_Dance.blend'):
   try:
    job=read_json(p.parent.parent/'job.json')
    if job.get('status')=='completed':entries.append({'profile':'rin','scene':str(p),'label':'Rin · '+job['name']})
   except (ValueError,OSError,KeyError):continue
  seen=set()
  for entry in entries:
   path=(self.root/entry['scene']).resolve()
   if path in seen or not path.is_relative_to(self.root) or path.suffix!='.blend' or not path.is_file():continue
   seen.add(path);sid=key_for(path.relative_to(self.root).as_posix())[:24]
   result.append({'id':sid,'profile':entry['profile'],'label':entry['label'],'scene':str(path),
    'case':entry.get('case'),'export_only':entry.get('export_only',False)})
  return result

 def _save(self,job):atomic_json(self.store/job['id']/'job.json',job)
 def _update(self,jid,**values):
  with self.lock:self.runs[jid].update(values,updated=time.time());self._save(self.runs[jid])
 def get(self,jid):
  with self.lock:
   if jid not in self.runs:raise ValueError('衣发任务不存在。')
   job=copy.deepcopy(self.runs[jid])
  output=Path(job['output']);runfile=output/'run.json'
  if runfile.is_file():
   try:
    state=read_json(runfile);job['stages']=[]
    for r in sorted(state.get('stages',{}).values(),key=lambda row:row.get('started',0)):
     row={k:v for k,v in r.items() if k in ('name','status','last_access')}
     if r.get('output') and r['status']=='running':
      progress=Path(r['output'])/'solver-progress.json'
      if progress.is_file():
       try:row['frame_progress']={k:v for k,v in read_json(progress).items() if k in ('frames','total')}
       except (ValueError,OSError):pass
     job['stages'].append(row)
   except (ValueError,OSError):job['stages']=[]
  job['files']=self.files(job)
  if job.get('kind')=='surface' and job.get('result') and (output/'latest.json').is_file():
   latest=read_json(output/'latest.json');chosen=job['result']['pair']
   job['selection_mode']=next((d['mode'] for d in latest['quality'].get('decisions',[]) if d['variant']==chosen),None)
   case=Path(job['result']['case']).resolve()
   if case.is_relative_to(output):
    guards=sorted(case.glob('cloth-guarded-*/component-guard.json'))
    if guards:
     guard=read_json(guards[-1]);job['cloth_components']={k:guard[k] for k in ('accepted_components','rejected_components')}
  if job.get('preview_dir'):
   preview=Path(job['preview_dir']).resolve()
   if preview.is_relative_to(output) and (preview/'preview.json').is_file():
    job['preview_relative']=preview.relative_to(output).as_posix()
    job['preview_frames']=read_json(preview/'preview.json')['frames']
  return job
 def files(self,job):
  output=Path(job['output']).resolve();result=[]
  if not output.is_relative_to(self.root/'output/secondary_motion'):return []
  allowed={'.blend','.fbx','.abc','.mdd','.png','.mp4','.json','.md'}
  for p in output.rglob('*'):
   if not p.is_file() or p.suffix not in allowed or 'reference' in p.parts:continue
   bucket=p.relative_to(output).parts[0]
   if bucket.startswith('mmd-') and str(output/bucket)!=(job.get('result') or {}).get('pair'):continue
   if p.suffix=='.json' and p.name not in {'physics-validation.json','surface-audit.json','surface-validation.json','manifest.json','preview.json','blend-package-validation.json','fbx-validation.json','alembic-validation.json','rigid-fbx-validation.json','fbx-character-validation.json'}:continue
   if not p.resolve().is_relative_to(output):continue
   rel=p.relative_to(output).as_posix();fid=key_for(rel)[:24]
   result.append({'id':fid,'name':p.name,'relative':rel,'bytes':p.stat().st_size,'selected_result':str(p.parent.resolve())==(job.get('result') or {}).get('pair'),'url':f'/api/secondary/runs/{job["id"]}/files/{fid}'})
  return result
 def _new(self,values):
  with self.lock:
   if any(j['status'] in ACTIVE for j in self.runs.values()):raise ValueError('已有衣发任务排队或运行，请等待或取消。')
   if any(self._live_child(j) for j in self.runs.values()):raise ValueError('上次子进程仍在运行，不能重复启动。')
   jid=uuid.uuid4().hex;output=self.root/'output/secondary_motion/dev6/workbench'/jid
   job={'id':jid,'output':str(output),'status':'queued','created':time.time(),'error':None,'physics_quality':'not_validated','format_validated':False,**values}
   self.runs[jid]=job;self.events[jid]=threading.Event();self._save(job)
   self.pool.submit(self.execute,jid,False);return self.get(jid)
 def start(self,request):
  if not request.enabled:raise ValueError('衣发开关关闭；现有动作保持，不创建物理任务。')
  profile=next((p for p in self.profiles() if p['id']==request.profile),None)
  if request.mode not in ('hair','surface','mmd_tools'):raise ValueError('不支持的处理模式。')
  if not profile or not profile.get(request.mode):raise ValueError((profile or {}).get('surface_reason') if request.mode=='surface' and (profile or {}).get('surface_reason') else '此角色没有可用的对应衣发配置。')
  source=next((p for p in self.sources() if p['id']==request.source_id),None)
  if not source or source['profile']!=request.profile:raise ValueError('工程与角色配置不匹配。')
  if source['export_only']:raise ValueError('该工程已含物理结果，仅可用于导出，不能重复叠加求解。')
  return self._new({'kind':request.mode,'profile':request.profile,'source':source,'source_sha256':file_sha(Path(source['scene'])),'preview':request.preview})
 def export(self,jid,request):
  parent=self.get(jid)
  if parent['status'] not in ('completed','quality_rejected') or not parent.get('result'):raise ValueError('尚无完整结果可以导出。')
  if parent['physics_quality']!='accepted' and not request.experimental:raise ValueError('完整碰撞未验收，需勾选仅验证导出格式。')
  allowed={'blender','skeleton','rigid','alembic','fbx-character'}
  if not request.formats or len(request.formats)!=len(set(request.formats)) or set(request.formats)-allowed:raise ValueError('导出格式无效。')
  if 'fbx-character' in request.formats and 'skeleton' not in request.formats:raise ValueError('角色FBX回读须同时包含骨骼FBX。')
  ordered=[n for n in ['blender','skeleton','rigid','alembic','fbx-character'] if n in request.formats]
  return self._new({'kind':'export','profile':parent['profile'],'source':parent['source'],'source_sha256':parent['source_sha256'],
   'parent':jid,'result':parent['result'],'formats':ordered,'experimental':request.experimental,'preview':False,'physics_quality':parent['physics_quality']})
 def cancel(self,jid):
  job=self.get(jid)
  if job['status'] not in ACTIVE:raise ValueError('任务当前没有运行。')
  self.events.setdefault(jid,threading.Event()).set()
  self._update(jid,status='cancelled' if job['status']=='queued' else 'cancelling');return self.get(jid)
 def resume(self,jid):
  job=self.get(jid)
  if job['status'] not in ('failed','cancelled','interrupted'):raise ValueError('仅可恢复失败、中断或取消的任务。')
  if self._live_child(job):raise ValueError('上次子进程仍在运行，不能重复恢复。')
  if file_sha(Path(job['source']['scene']))!=job['source_sha256']:raise ValueError('源工程已改变，请新建任务，旧结果保留。')
  with self.lock:
   if any(j['status'] in ACTIVE for j in self.runs.values()):raise ValueError('已有衣发任务运行。')
   self.events[jid]=threading.Event();self._update(jid,status='queued',error=None)
   self.pool.submit(self.execute,jid,True)
  return self.get(jid)
 def _live_child(self,job):
  return bool(job.get('child_identity') and not job.get('child_finished',True) and process_identity(job.get('child_pid'))==job['child_identity'])
 def refresh_preview(self,jid):
  job=self.get(jid)
  if job['kind']=='export' or job['status'] not in ('completed','quality_rejected') or not job.get('result'):raise ValueError('请选择已有物理结果的任务。')
  if file_sha(Path(job['source']['scene']))!=job['source_sha256']:raise ValueError('源工程已改变，不能复用。')
  with self.lock:
   if any(j['status'] in ACTIVE for j in self.runs.values()) or any(self._live_child(j) for j in self.runs.values()):raise ValueError('已有任务运行。')
   self.events[jid]=threading.Event();self._update(jid,status='queued',preview=True,error=None,
    operation='preview_only',preview_terminal_status=job['status'])
   self.pool.submit(self.execute,jid,True)
  return self.get(jid)
 def command(self,job,resume):
  scripts=self.root/'scripts';out=Path(job['output'])
  if job['kind']=='export':
   args=['export_secondary_motion.py','--scene',Path(job['result']['pair'])/'after.blend','--input-dir',job['result']['case'],
    '--output-dir',out,'--formats',','.join(job['formats'])]
   if job.get('experimental'):args+=['--experimental']
  elif job['kind']=='hair':args=['simulate_secondary_workbench_hair.py','--scene',job['source']['scene'],'--output-dir',out,'--profile',job['profile']]
  else:args=['simulate_secondary_surface.py','--scene',job['source']['scene'],'--output-dir',out,'--profile',job['profile'],'--enable-surface']
  if resume and (out/'run.json').exists():args+=['--resume']
  return [sys.executable,'-B','-X','utf8',str(scripts/args[0]),*map(str,args[1:])]
 def _call(self,jid,command,managed):
  folder=self.store/jid;logfile=folder/('process-'+uuid.uuid4().hex[:8]+'.log')
  self._update(jid,process_log=str(logfile))
  with logfile.open('x',encoding='utf-8') as log:
   child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=self.root,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   self._update(jid,child_pid=child.pid,child_identity=process_identity(child.pid),child_finished=False)
   while child.poll() is None:
    if self.events[jid].is_set():
     if managed and (managed/'control.json').exists():RunSession.cancel(managed)
     elif managed is None:child.terminate()
    time.sleep(.2)
   self._update(jid,child_finished=True)
   return child.returncode
 def execute(self,jid,resume):
  job=self.get(jid)
  if self.events[jid].is_set():return
  self._update(jid,status='processing',error=None)
  try:
   if job.get('operation')=='preview_only':
    self._render_preview(jid,job,job['result'])
    if self.events[jid].is_set():self._update(jid,status='cancelled');return
    self._update(jid,status=job['preview_terminal_status'],operation='physics');return
   plugin_result=None
   if job['kind']=='mmd_tools':
    pair=Path(job['output'])/('mmd-'+uuid.uuid4().hex)
    try:
     for phase in ('bake','apply'):
      self._update(jid,plugin_phase=phase)
      code=self._call(jid,mmd_command(self.root,job,pair,phase),None)
      if self.events[jid].is_set():self._update(jid,status='cancelled');return
      if code!=0:raise ValueError(f'MMD Tools {phase} 退出码 {code}，详见插件日志')
     proof=read_json(pair/'physics-validation.json')
     if (proof.get('passed') is not True or proof.get('source_sha256')!=job['source_sha256']
         or proof.get('provider')!='mmd_tools' or not all((pair/n).is_file() for n in ('before.blend','after.blend','surface-final.mdd'))):
      raise ValueError('插件结果未完成保护验证')
     plugin_result={'pair':str(pair),'case':str(pair)}
     self._update(jid,actual_provider='mmd_tools')
    except (OSError,ValueError,KeyError) as exc:
     if self.events[jid].is_set():self._update(jid,status='cancelled');return
     if not mmd_settings(self.root)['fallback']:raise
     if file_sha(Path(job['source']['scene']))!=job['source_sha256']:raise ValueError('源工程变化，无法回退')
     self._update(jid,kind=mmd_settings(self.root)['fallback_mode'],output=str(Path(job['output'])/'internal-fallback'),requested_provider='mmd_tools',actual_provider='internal',
                  fallback_reason=str(exc),plugin_log=self.get(jid).get('process_log'))
     job=self.get(jid)
   code=0 if plugin_result else self._call(jid,self.command(job,resume),Path(job['output']))
   if self.events[jid].is_set():self._update(jid,status='cancelled');return
   out=Path(job['output'])
   if code not in (0,3):raise RuntimeError(f'处理退出码{code}，请查看任务日志。')
   latest={'result':plugin_result} if plugin_result else read_json(out/'latest.json')
   if job['kind']=='export':self._update(jid,status='completed',format_validated=latest['format_validation_passed'],delivery=latest['delivery']);return
   if job['kind'] in ('hair','mmd_tools'):result=latest['result'];quality='not_fully_validated';status='completed'
   else:
    q=latest['quality'];selected=q['selected'];pair=selected or q['last_experimental_candidate']
    if pair is None and q.get('stopped_stage')=='cloth':
     self._update(jid,status='quality_rejected',result=None,physics_quality='rejected',protection_passed=None,
      selection_gate_passed=False,stopped_stage='cloth',error=q.get('stop_reason','衣物预热未通过，已停止。'),operation='physics')
     return
    result={'pair':pair,'case':latest['case']}
    audit=read_json(Path(result['pair'])/'surface-audit.json')
    if audit.get('protection_passed') is not True:raise ValueError('主动作保护审核未通过，不能发布')
    quality='accepted' if audit.get('collision_quality_passed') is True else 'rejected'
    status='completed' if selected else 'quality_rejected'
    self._update(jid,selection_gate_passed=bool(selected))
   self._update(jid,result=result,physics_quality=quality,protection_passed=True)
   if job['preview']:
    # Once physics is complete, retry/cancel/resume must never recalculate it
    # merely because a newer solver invalidates the managed stage cache.
    self._update(jid,operation='preview_only',preview_terminal_status=status)
    self._render_preview(jid,job,result)
   if self.events[jid].is_set():self._update(jid,status='cancelled');return
   self._update(jid,status=status,operation='physics')
  except Exception as error:self._update(jid,status='failed',error=str(error))

 def _render_preview(self,jid,job,result):
  out=Path(job['output']);renderer_sha=file_sha(self.root/'scripts/render_secondary_workbench_preview.py')
  pair_sha={v:file_sha(Path(result['pair'])/(v+'.blend')) for v in ['before','after']}
  valid_previews=[p for p in out.glob('preview*') if valid_preview(p,renderer_sha,pair_sha)]
  preview=valid_previews[-1] if valid_previews else out/('preview-'+uuid.uuid4().hex[:8])
  command=[str(read_json(self.root/'config.json')['blender_path']),'--background','--factory-startup','--disable-autoexec','--threads','2','--python-exit-code','23','--python',str(self.root/'scripts/render_secondary_workbench_preview.py'),'--','--pair-dir',result['pair'],'--case',result['case'],'--output-dir',str(preview)]
  if not valid_previews and self._call(jid,command,None)!=0:
   if self.events[jid].is_set():return
   raise RuntimeError('物理计算已保留，但预览生成失败；可恢复重试。')
  if self.events[jid].is_set():return
  if not valid_preview(preview,renderer_sha,pair_sha):raise RuntimeError('预览未完成或校验失败；半成品保留，可恢复重试。')
  if pair_sha!={v:file_sha(Path(result['pair'])/(v+'.blend')) for v in ['before','after']}:raise RuntimeError('渲染期间物理结果发生变化；预览不发布。')
  self._update(jid,preview_dir=str(preview))

def package_entries(job):
 delivery=Path(job.get('delivery','')).resolve();output=Path(job['output']).resolve()
 if not job.get('format_validated') or not delivery.is_relative_to(output):raise ValueError('尚无完整导出包。')
 manifest=read_json(delivery/'manifest.json');entries=[]
 for name,record in manifest['files'].items():
  path=(delivery/name).resolve()
  if not path.is_relative_to(delivery) or not path.is_file() or path.stat().st_size!=record['bytes'] or file_sha(path)!=record['sha256']:raise ValueError('导出包文件缺失或已改变，请重新导出。')
  entries.append((name,path))
 entries.append(('manifest.json',delivery/'manifest.json'));return entries

def stream_zip(entries):
 """Bounded RAM streaming; no second multi-GB copy on disk."""
 chunks=queue.Queue(maxsize=4);stop=threading.Event()
 def send(value):
  while not stop.is_set():
   try:chunks.put(value,timeout=.1);return
   except queue.Full:pass
  raise InterruptedError('Download closed')
 class Writer:
  def write(self,data):
   if data:send(bytes(data))
   return len(data)
  def flush(self):pass
 def produce():
  try:
   with zipfile.ZipFile(Writer(),'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
    for name,path in entries:
     with archive.open(name,'w',force_zip64=True) as destination,path.open('rb') as source:
      while data:=source.read(64*1024):destination.write(data)
   send(None)
  except BaseException as error:
   if not stop.is_set():send(error)
 worker=threading.Thread(target=produce,daemon=True);worker.start()
 try:
  while True:
   item=chunks.get()
   if item is None:break
   if isinstance(item,BaseException):raise item
   yield item
 finally:stop.set();worker.join(timeout=2)

def router(provider):
 api=APIRouter(prefix='/api/secondary')
 def invoke(method,*args):
  try:return method(*args)
  except (ValueError,OSError) as e:raise HTTPException(409,str(e))
 @api.get('/catalog')
 def catalog():
  w=provider();sources=[{k:v for k,v in p.items() if k not in ('scene','case')} for p in w.sources()]
  settings=mmd_settings(w.root)
  return {'plugins':{'mmd_tools':{k:settings[k] for k in ('ready','fallback','fallback_mode','reason')}},'profiles':w.profiles(),'sources':sources,'default_enabled':False,'full_collision_accepted':False}
 @api.get('/runs')
 def runs():
  w=provider();return {'runs':[w.get(jid) for jid in sorted(w.runs,key=lambda k:w.runs[k]['created'],reverse=True)]}
 @api.post('/runs',status_code=202)
 def start(request:StartRequest):return invoke(provider().start,request)
 @api.get('/runs/{jid}')
 def get(jid:str):return invoke(provider().get,jid)
 @api.post('/runs/{jid}/cancel')
 def cancel(jid:str):return invoke(provider().cancel,jid)
 @api.post('/runs/{jid}/resume',status_code=202)
 def resume(jid:str):return invoke(provider().resume,jid)
 @api.post('/runs/{jid}/export',status_code=202)
 def export(jid:str,request:ExportRequest):return invoke(provider().export,jid,request)
 @api.post('/runs/{jid}/refresh-preview',status_code=202)
 def refresh_preview(jid:str):return invoke(provider().refresh_preview,jid)
 @api.get('/runs/{jid}/files/{fid}')
 def file(jid:str,fid:str):
  job=invoke(provider().get,jid);record=next((p for p in job['files'] if p['id']==fid),None)
  if record is None:raise HTTPException(404,'文件不存在。')
  path=(Path(job['output'])/record['relative']).resolve()
  if not path.is_relative_to(Path(job['output']).resolve()):raise HTTPException(403,'非法路径。')
  return FileResponse(path,filename=path.name)
 @api.get('/runs/{jid}/download')
 def download(jid:str):
  entries=invoke(package_entries,invoke(provider().get,jid))
  return StreamingResponse(stream_zip(entries),media_type='application/zip',headers={'Content-Disposition':f'attachment; filename="MotionBridge-secondary-{jid}.zip"'})
 @api.get('/runs/{jid}/log',response_class=PlainTextResponse)
 def log(jid:str):
  w=provider();job=invoke(w.get,jid);parts=[]
  for field in ('plugin_log','process_log'):
   if not job.get(field):continue
   path=Path(job[field]).resolve()
   if not path.is_relative_to(w.store/jid):raise HTTPException(403,'非法日志路径。')
   if path.is_file():parts.append(path.read_text(encoding='utf-8',errors='replace')[-20000:])
  out=Path(job['output']).resolve();manifest=out/'run.json'
  if manifest.is_file():
   records=sorted(read_json(manifest).get('stages',{}).values(),key=lambda r:r.get('started',0))
   for row in records[-2:]:
    if row.get('log'):
     path=Path(row['log']).resolve()
     if path.is_relative_to(out) and path.is_file():parts.append(row.get('name','stage')+'\n'+path.read_text(encoding='utf-8',errors='replace')[-20000:])
  return '\n\n'.join(parts) or '尚无处理日志。'
 return api

