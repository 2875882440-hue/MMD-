"""Read verified human mocap cache and crop a new-to-physics three-second window."""
from pathlib import Path
import copy,datetime,hashlib,json,subprocess,uuid
import imageio_ffmpeg
R=Path(r'E:\MotionBridge_Physics');V=R/'validation/secondary-motion-dev6';data=R/'data/physics-jobs'
source=R/'research/v071_compare/gvhmr/motion.json';video=Path(r'D:\AI素材\动捕源素材\真人舞蹈_许佳琪.mp4')
digest=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
record=V/'regression-input.json'
if record.exists():raise ValueError('Keep the existing regression fixture')
original=json.loads(source.read_text(encoding='utf-8'));count=len(original['frames']);start,end=450,540
def crop(value):
 if isinstance(value,dict):return {k:crop(v) for k,v in value.items()}
 if isinstance(value,list) and len(value)==count:return copy.deepcopy(value[start:end])
 return copy.deepcopy(value)
motion=crop(original);motion['num_frames']=end-start
for i,frame in enumerate(motion['frames']):frame['time']=i/motion['fps']
motion['source'].update(frames=end-start,duration=(end-start)/motion['fps'])
motion['regression_source']={'original_cache':str(source),'original_cache_sha256':digest(source),'source_video':str(video),
 'video_sha256':digest(video),'source_video_window_seconds':[60,63],'original_30s_cache_window':[450,540],
 'new_inference':False,'note':'Previously verified real GVHMR source; crop only. Old aggregate diagnostics retained, not recomputed accuracy metrics.'}
jid=uuid.uuid4().hex;folder=data/jid;output=folder/'outputs';output.mkdir(parents=True)
(output/'motion.json').write_text(json.dumps(motion,ensure_ascii=False),encoding='utf-8')
clip=output/'source.mp4';ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-ss','60','-i',str(video),'-t','3','-r','30','-an','-c:v','libx264','-crf','22','-pix_fmt','yuv420p',str(clip)],check=True)
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
job={'id':jid,'name':'第6步回归 · 许佳琪60–63秒（已有GVHMR缓存）','status':'completed','stage':'已导入真实解算缓存的独立回归片段','progress':100,
 'created_at':now,'updated_at':now,'error':None,'warnings':['本片段复用已验证的真人GVHMR解算，没有新运行识别。衣发测试不代表动作准确率。'],
 'settings':motion['settings'],'meta':{'fps':30,'frames':90,'duration':3},'is_demo':False,'software_version':'0.7.0-dev.6-validation',
 'files':[{'name':'motion.json','label':'已验证的源动作片段','size':(output/'motion.json').stat().st_size,'url':f'/api/jobs/{jid}/files/motion.json'},
          {'name':'source.mp4','label':'真人源60–63秒','size':clip.stat().st_size,'url':f'/api/jobs/{jid}/files/source.mp4'}]}
(folder/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
proof={'source_job_id':jid,'source_motion':str(output/'motion.json'),'frames':90,'original_motion_sha256':digest(source),'source_video_sha256':digest(video),
       'new_inference':False,'source_video_window_seconds':[60,63],'old_source_unchanged':digest(source)==motion['regression_source']['original_cache_sha256']}
record.write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8');print('DEV6_REAL_SOURCE_READY',jid)
