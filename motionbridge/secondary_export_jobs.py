"""Format-validated packages with independent physics-quality labels."""
from pathlib import Path
import json,shutil,sys,uuid
from .secondary_cache import ArtifactCache,atomic_json,file_sha,key_for,checked_file
from .secondary_tasks import RunSession,matches
from .secondary_pipeline import commands,read

FORMAT_WORKERS={
 'blender':('package_secondary_blend.py','blend-package-validation.json'),
 'skeleton':('export_secondary_skeleton_blender.py','fbx-validation.json'),
 'rigid':('export_secondary_rigid_blender.py','rigid-fbx-validation.json'),
 'alembic':('export_secondary_alembic_blender.py','alembic-validation.json'),
 'fbx-character':('apply_secondary_fbx_blender.py','fbx-character-validation.json')}

def export_files(path):
 return [p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file() and 'reference' not in p.relative_to(path).parts and not p.name.endswith('-reference.npz') and p.suffix not in ('.blend1','.blend2')]

def format_key(stamp,name,settings,code,dependencies=None):
 value={'schema':1,'kind':'format_export','source':{k:stamp[k] for k in ('source_sha256','dependencies','blender_version','blender_build')},'format':name,'settings':settings,'code':code}
 if dependencies:value['format_dependencies']=dependencies
 return key_for(value)

def validate_export_receipt(report,stamp,take=None,kind=None):
 if report.get('passed') is not True:raise ValueError('Format readback failed')
 if report.get('source_sha256')!=stamp['source_sha256']:raise ValueError('Do not relabel a different source scene as this export')
 if report.get('physics_quality_approval') is True:raise ValueError('An export receipt cannot approve physics quality')
 if report.get('skipped'):
  if kind!='rigid' or stamp.get('rigid_object_count')!=0:raise ValueError('Cannot skip an export that has actual source objects')
  return
 expected_frames=min(take or (stamp['frame_end']-stamp['frame_start']+1),stamp['frame_end']-stamp['frame_start']+1)
 if report.get('sample_frames',report.get('matrix_samples'))!=2*expected_frames-1:raise ValueError('Do not relabel a short smoke test as a full export')
 if report.get('source_file_unchanged') is not True:raise ValueError('Source changed during export')

def export_package(root,scene,case,out,*,formats=('blender','skeleton','rigid','alembic'),experimental=False,resume=False,use_cache=True,take=None,render_check=False,adopt=None):
 root=Path(root).resolve();scene=Path(scene).resolve();case=Path(case).resolve();out=Path(out).resolve();adopt=adopt or {}
 if not scene.is_relative_to(root) or not case.is_relative_to(root/'output/secondary_motion') or not out.is_relative_to(root/'output/secondary_motion'):raise ValueError('Use this isolated project')
 if take is not None and (type(take) is not int or take<=0):raise ValueError('take must be a positive frame count')
 if set(formats)-set(FORMAT_WORKERS):raise ValueError('Unsupported export format')
 if 'fbx-character' in formats and 'skeleton' not in formats:raise ValueError('Character readback requires skeleton export')
 profile=read(case/'profile.json');blender=Path(read(root/'config.json')['blender_path'])
 settings={'take':take,'sample_step':.5,'render_check':render_check,'model_sha256':profile['model']['sha256']}
 code={n:{'worker':file_sha(root/'scripts'/FORMAT_WORKERS[n][0]),'geometry_helper':file_sha(root/'motionbridge/blender_secondary_rigid.py'),'protection_helper':file_sha(root/'scripts/solve_secondary_hair_blender.py')} for n in formats}
 spec={'kind':'format_export','source':str(scene),'source_sha256':file_sha(scene),'formats':sorted(formats),'settings':settings,'code':code,'experimental_permission':experimental}
 cache=ArtifactCache(root/'data/secondary-cache') if use_cache else None
 with RunSession(out,spec,cache,resume=resume) as run:
  stamp_path=run.revision_root/f'export-preflight-{uuid.uuid4().hex[:8]}.json'
  run.utility(commands(root,blender,'preflight_secondary_source.py',['--scene',scene,'--output',stamp_path,'--profile',profile['id'],'--purpose','export']))
  stamp=read(stamp_path)
  if stamp['physics_quality']['collision_quality_passed'] is False and not experimental:raise ValueError('该物理结果未通过；仅格式验证请显式--experimental，不能升级物理状态')
  results={};keys={}
  for name in formats:
   worker,report_name=FORMAT_WORKERS[name];deps={'skeleton_key':keys['skeleton'],'skeleton_file_sha256':file_sha(results['skeleton']/'skeleton.fbx')} if name=='fbx-character' else None
   key=format_key(stamp,name,settings,code[name],deps);keys[name]=key
   def validate(path,report_name=report_name,name=name):
    report=read(path/report_name)
    validate_export_receipt(report,stamp,take,name)
    if file_sha(scene)!=stamp['source_sha256'] or any(file_sha(Path(p))!=digest for p,digest in stamp['dependencies'].items()):raise ValueError('Source/dependency changed during export')
    return report
   if cache and name in adopt:
    origin=Path(adopt[name]).resolve()
    if not origin.is_relative_to(root/'output/secondary_motion'):raise ValueError('Adopt only local validated exports')
    validate(origin)
    with cache.reserve('export-'+name,key):cache.publish('export-'+name,key,origin,export_files(origin),provenance={'adopted_validated_export':str(origin),'physics_quality':stamp['physics_quality'],'format_validation_only':True})
   def produce(dest,call,name=name,worker=worker,validate=validate):
    args=['--scene',scene,'--input-dir',case,'--output-dir',dest]
    if take:args+=['--take',take]
    if render_check and name in ('alembic','fbx-character'):args+=['--render-check']
    if name=='fbx-character':args+=['--fbx',results['skeleton']/'skeleton.fbx']
    call(commands(root,blender,worker,args));validate(dest)
   results[name]=run.stage('export-'+name,key,run.revision_root,produce,export_files,provenance={'physics_quality':stamp['physics_quality'],'format_validation_only':True})
   validate(results[name])
  # A rollback is another NEW package of the original captured input, not a
  # destructive change to an existing model/action/job.
  if 'blender' in formats:
   original=case/'before.blend';rollback_key=key_for({'source':file_sha(original),'kind':'input_rollback','code':code['blender'],'settings':settings})
   def rollback(dest,call):
    args=['--scene',original,'--input-dir',case,'--output-dir',dest]
    if take:args+=['--take',take]
    call(commands(root,blender,'package_secondary_blend.py',args))
    assert read(dest/'blend-package-validation.json')['passed']
   results['rollback']=run.stage('export-rollback',rollback_key,run.revision_root,rollback,export_files)
   after=read(results['blender']/'blend-package-validation.json');before=read(results['rollback']/'blend-package-validation.json')
   if after['protected_matrix_sha256']!=before['protected_matrix_sha256']:raise ValueError('Rollback would change primary body motion')
  delivery=run.revision_root/'delivery';manifest_path=delivery/'manifest.json'
  existing=read(manifest_path) if manifest_path.is_file() else None
  if existing is None or not matches(delivery,existing.get('files',{})):
   if delivery.exists():delivery=run.revision_root/('delivery-'+uuid.uuid4().hex[:8])
   needed=sum(p.stat().st_size for folder in results.values() for n in export_files(folder) for p in [folder/n])
   if shutil.disk_usage(run.revision_root).free<needed+512*1024**2:raise RuntimeError('E盘空间不足以复制独立交付包；已验证阶段产物保留')
   delivery.mkdir();files={}
   for name,folder in results.items():
    for entry in export_files(folder):
     src=checked_file(folder,entry);relative=name+'/'+entry;dest=checked_file(delivery,relative);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
     files[relative]={'sha256':file_sha(dest),'bytes':dest.stat().st_size}
   text='''# MotionBridge 第5步导出包

此包仅验证格式/数据搬运，不代表碰撞、穿模或识别准确率通过。
- blender/scene.blend：保留原曲线、蒙皮和形态键；同目录cache与textures一起保留。
- skeleton/skeleton.fbx：骨骼动画（无衣物顶点/无形态键）；绑定基底可能不同，不能盲目复制局部Action。
- fbx-character/retargeted-character.blend（若有）：同源FBX按世界姿态接回原模型副本的验证；不是任意新动作自动适配。
- rigid/rigid-attachments.fbx：独立硬饰动画，非自由6DOF求解。
- alembic/geometry.abc：完整可见人物的烘焙形变；readback.blend已恢复原材质。ABC没有骨骼和形态键。
- rollback/scene.blend：原捕获输入的独立回退副本，主动作与当前结果一致，未覆盖任何旧工程。
完整Alembic已经包含硬饰，勿再叠加rigid.fbx重复显示。C4D运行时/着色器未实际验证；贴图映射见blender/materials.json。
'''
   (delivery/'README.md').write_text(text,encoding='utf-8');files['README.md']={'sha256':file_sha(delivery/'README.md'),'bytes':(delivery/'README.md').stat().st_size}
   atomic_json(delivery/'manifest.json',{'schema':1,'format_validation_passed':True,'physics_quality_approved':False,'source_stamp':stamp,'experimental_permission':experimental,'formats':list(results),'keys':keys,'files':files})
  if existing is not None and (delivery/'manifest.json').is_file():
   current=read(delivery/'manifest.json');current['keys']=keys;atomic_json(delivery/'manifest.json',current)
  summary={'format_validation_passed':True,'physics_quality_approved':False,'physics_quality':stamp['physics_quality'],'delivery':str(delivery),'outputs':{k:str(v) for k,v in results.items()}}
  atomic_json(out/'latest.json',summary);run.state['format_status']='validated';run.save();return summary
