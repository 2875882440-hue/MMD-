"""Cached model-owned secondary stages with independent quality gates."""
from pathlib import Path
import json,subprocess,uuid
from .secondary_cache import ArtifactCache,atomic_json,file_sha,key_for,stage_keys
from .secondary_tasks import RunSession
from .secondary_guard import attachment_repair_gate,prewarm_stop_gate
from .secondary_surface_config import surface_settings
from .secondary_contact_constraints import simulation_quality_gate

CAPTURE=['before.blend','source-positions.npy','protected-before.npy','input.json','model-plan.json','partitions.npz']
CLOTH=['cloth-positions.npy','cloth-index-map.npz','cloth-validation.json']
HAIR=['hair-after.blend','hair-coupled.npz','hair-validation.json','warmup.json','cross-joints.json','body-faces.npy']
BUNDLE=['before.blend','after.blend','surface-final.mdd','rigid-transforms.npy','surface-validation.json','surface-audit.json']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def commands(root,blender,script,args):
 return [str(blender),'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','23','--python',str(Path(root)/'scripts'/script),'--',*map(str,args)]

def code_hashes(root):
 root=Path(root)
 groups={
 'capture':['scripts/capture_secondary_surface_inputs.py','scripts/normalize_secondary_cache_blends.py','scripts/bind_secondary_model_blender.py','motionbridge/secondary_motion.py','motionbridge/secondary_plan.py','motionbridge/secondary_topology.py','motionbridge/secondary_segmentation.py','motionbridge/secondary_contact_policy.py','motionbridge/secondary_attachment_driver.py','motionbridge/secondary_joint_contacts.py','motionbridge/secondary_prewarm.py','motionbridge/secondary_units.py','motionbridge/secondary_surface.py','motionbridge/secondary_surface_config.py','motionbridge/secondary_contact_constraints.py'],
 'cloth':['scripts/solve_secondary_cloth_native.py','motionbridge/secondary_surface.py','motionbridge/secondary_surface_config.py','motionbridge/secondary_contact_constraints.py'],
 'hair':['scripts/solve_secondary_hair_coupled.py','scripts/solve_secondary_hair_blender.py','motionbridge/secondary_hair.py','motionbridge/secondary_coupling.py','motionbridge/secondary_surface.py'],
 'bundle':['motionbridge/secondary_attachment_driver.py','motionbridge/secondary_joint_contacts.py','motionbridge/secondary_prewarm.py','scripts/assemble_secondary_surface_cache.py','scripts/audit_secondary_surface.py','motionbridge/blender_secondary_rigid.py','motionbridge/blender_triangle_contacts.py','motionbridge/triangle_contacts.py','motionbridge/secondary_guard.py','motionbridge/secondary_contact_constraints.py'],
 'guard':['scripts/guard_secondary_cloth_components.py','motionbridge/blender_triangle_contacts.py','motionbridge/triangle_contacts.py']}
 return {k:{p:file_sha(root/p) for p in names} for k,names in groups.items()}

def pipeline_keys(root,stamp,profile):
 code=code_hashes(root);keys=stage_keys({k:stamp[k] for k in ('source','source_sha256','dependencies','blender_version','blender_build')},profile,code)
 keys['bundle-coupled']=key_for({'input':keys['capture'],'cloth':keys['cloth'],'hair':keys['hair'],'code':code['bundle']})
 keys['bundle-preserved']=key_for({'input':keys['capture'],'cloth':keys['cloth'],'hair':'source','code':code['bundle']})
 keys['cloth-guarded']=key_for({'input':keys['capture'],'cloth':keys['cloth'],'sample':keys['bundle-preserved'],'code':code['guard']})
 keys['bundle-guarded']=key_for({'input':keys['capture'],'cloth':keys['cloth-guarded'],'hair':'source','code':code['bundle']})
 return keys

def canonical_prepare(root,blender,plan,run=None):
 root=Path(root)
 def prepare(source,payload):
  if not any(payload.glob('*.blend')):return
  command=commands(root,blender,'normalize_secondary_cache_blends.py',['--source-dir',source,'--payload',payload,'--plan',plan(source)])
  if run:run.execute(command)
  else:
   log=root/'logs'/f'cache-normalize-{uuid.uuid4().hex[:12]}.log';log.parent.mkdir(exist_ok=True)
   with log.open('x',encoding='utf-8') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
 return prepare

def bundle_files(path):return BUNDLE+(['audit-body-before.npy'] if (path/'audit-body-before.npy').exists() else [])

def run_pipeline(root,scene,out,profile_id,*,resume=False,use_cache=True,compare=False):
 root=Path(root).resolve();scene=Path(scene).resolve();out=Path(out).resolve()
 if not scene.is_relative_to(root) or not out.is_relative_to(root/'output/secondary_motion'):raise ValueError('Use the isolated development copy')
 profile=read(root/f'profiles/secondary_motion/{profile_id}.json')
 if profile.get('surface_segmentation',{}).get('required_for_surface_dynamics') or 'secondary_contact_rules' in profile:
  from .secondary_plan import compile_profile
  from .secondary_segmentation import require_review
  require_review(compile_profile(profile,root))
 settings=surface_settings(profile);blender=Path(read(root/'config.json')['blender_path'])
 cache=ArtifactCache(root/'data/secondary-cache') if use_cache else None
 spec={'source':str(scene),'source_sha256':file_sha(scene),'profile':profile,'cache_graph_version':1,'acceptance_policy':'before_strict_single_candidate_v1','code':code_hashes(root),'blender_executable':file_sha(blender)}
 with RunSession(out,spec,cache,resume=resume) as run:
  stamp_path=run.revision_root/f'preflight-{uuid.uuid4().hex[:8]}.json'
  run.utility(commands(root,blender,'preflight_secondary_source.py',['--scene',scene,'--output',stamp_path,'--profile',profile_id]))
  stamp=read(stamp_path);keys=pipeline_keys(root,stamp,profile)
  def execute(script,args,call):return call(commands(root,blender,script,args))
  def capture(dest,call):execute('capture_secondary_surface_inputs.py',['--scene',scene,'--output-dir',dest,'--profile',profile_id],call)
  case=run.stage('capture',keys['capture'],run.revision_root,capture,CAPTURE,
                 prepare=canonical_prepare(root,blender,lambda source:source/'model-plan.json',run),provenance={'stamp':stamp,'profile_structure_key':keys['capture']})
  atomic_json(case/'profile.json',profile)
  prepare=canonical_prepare(root,blender,lambda _:case/'model-plan.json',run)
  def solve(script):
   return lambda dest,call:execute(script,['--input-dir',case,'--tag',dest.name],call)
  cloth=run.stage('cloth',keys['cloth'],case,solve('solve_secondary_cloth_native.py'),CLOTH,provenance={'settings':profile.get('surface_dynamics')})
  warm_gate=prewarm_stop_gate(read(cloth/'cloth-validation.json'),settings)
  if not warm_gate['accepted']:
   selection={'decisions':[{'variant':str(cloth),'mode':'prewarm_stop','gate':warm_gate}],
    'selected':None,'last_experimental_candidate':None,'complete_collision_acceptance':False,'stopped_stage':'cloth',
    'stop_reason':'衣物预热未稳定或未验证，已停止；未执行头发耦合、表面装配或自动回退试算。',
    'automatic_retry':False,'reference':'captured_input_before'}
   atomic_json(case/'quality-selection.json',selection);atomic_json(out/'latest.json',{'revision':run.revision,'case':str(case),'quality':selection})
   run.state['physics_status']='quality_rejected';run.save()
   return {'computation_completed':False,'physics_result_published':False,'status':'quality_rejected','case':str(case),'selection':selection}
  hair=run.stage('hair',keys['hair'],case,solve('solve_secondary_hair_coupled.py'),HAIR,prepare=prepare,provenance={'settings':profile['hair_dynamics'],'physics_quality':'not_approved_by_cache'})
  def assemble(cloth_dir,hair_dir=None,baseline=None):
   def produce(dest,call):
    args=['--input-dir',case,'--cloth-dir',cloth_dir,'--tag',dest.name]
    if hair_dir:args+=['--hair-scene',hair_dir/'hair-after.blend']
    execute('assemble_secondary_surface_cache.py',args,call)
    args=['--input-dir',case,'--pair-dir',dest]
    if baseline:args+=['--reuse-before',baseline/'surface-audit.json']
    execute('audit_secondary_surface.py',args,call)
   return produce
  first=run.stage('bundle-coupled',keys['bundle-coupled'],case,assemble(cloth,hair),bundle_files,prepare=prepare)
  pair=first;decisions=[]
  def check(path,mode):
   gate=simulation_quality_gate(attachment_repair_gate(read(path/'surface-audit.json')),read(cloth/'cloth-validation.json'),read(path/'surface-validation.json'))
   decisions.append({'variant':str(path),'mode':mode,'gate':gate});return gate
  gate=check(first,'coupled')
  selection={'decisions':decisions,'selected':str(pair) if gate['accepted'] else None,'last_experimental_candidate':str(pair),'complete_collision_acceptance':False,
   'automatic_retry':False,'reference':'captured_input_before','stopped_stage':None if gate['accepted'] else 'bundle-coupled'}
  atomic_json(case/'quality-selection.json',selection);atomic_json(out/'latest.json',{'revision':run.revision,'case':str(case),'quality':selection})
  if not gate['accepted']:
   run.state['physics_status']='quality_rejected';run.save()
   return {'computation_completed':True,'physics_result_published':False,'status':'quality_rejected','case':str(case),'selection':selection}
  if compare:
   comparison=pair/'comparison'
   if comparison.exists() and not (comparison/'camera.json').is_file():
    target=pair/('comparison.partial-'+uuid.uuid4().hex[:8]);assert comparison.resolve().is_relative_to(out) and target.resolve().is_relative_to(out);comparison.rename(target)
   if not comparison.exists():execute('frame_secondary_surface_compare.py',['--input-dir',case,'--pair-dir',pair],run.utility)
   if not (comparison/'protection-verification.json').exists():execute('verify_secondary_surface_pair.py',['--input-dir',case,'--pair-dir',comparison],run.utility)
   if not (comparison/'cache-verification.json').exists():execute('verify_secondary_surface_cache.py',['--pair-dir',comparison],run.utility)
   if not (comparison/'frames/render-complete.json').exists():execute('render_secondary_hair_compare.py',['--pair-dir',comparison,'--resume'],run.utility)
   if not (comparison/'video-verification.json').exists():run.utility([__import__('sys').executable,'-B',root/'scripts/compose_secondary_surface_compare.py','--pair-dir',comparison,'--input-dir',case])
  return {'computation_completed':True,'physics_result_published':True,'status':'regression_gate_accepted_not_zero_collision','case':str(case),'selection':selection}
