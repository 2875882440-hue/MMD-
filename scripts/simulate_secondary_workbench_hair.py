"""Managed cached wrapper; the accepted hair solver is unchanged."""
from pathlib import Path
import argparse,json,sys,uuid
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from motionbridge.secondary_cache import ArtifactCache,atomic_json,file_sha,key_for,read_json
from motionbridge.secondary_tasks import RunSession
from motionbridge.secondary_pipeline import commands,canonical_prepare
p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--profile',default='rin',choices=['rin']);p.add_argument('--resume',action='store_true');args=p.parse_args()
scene=args.scene.resolve();out=args.output_dir.resolve()
if not scene.is_relative_to(ROOT) or not out.is_relative_to(ROOT/'output/secondary_motion'):raise ValueError('Use isolated project')
profile=read_json(ROOT/f'profiles/secondary_motion/{args.profile}.json');blender=read_json(ROOT/'config.json')['blender_path']
paths=['scripts/solve_secondary_hair_blender.py','scripts/bind_secondary_model_blender.py','scripts/normalize_secondary_cache_blends.py','motionbridge/secondary_hair.py','motionbridge/secondary_motion.py','motionbridge/secondary_plan.py','motionbridge/secondary_topology.py','motionbridge/secondary_units.py']
code={name:file_sha(ROOT/name) for name in paths};spec={'kind':'accepted_hair_workbench','source_sha256':file_sha(scene),'source':str(scene),'profile':profile,'code':code}
with RunSession(out,spec,ArtifactCache(ROOT/'data/secondary-cache'),resume=args.resume) as run:
 stamp=run.revision_root/('preflight-'+uuid.uuid4().hex[:8]+'.json')
 run.utility(commands(ROOT,blender,'preflight_secondary_source.py',['--scene',scene,'--output',stamp,'--profile',args.profile]))
 key=key_for({'spec':spec,'stamp':read_json(stamp)})
 def producer(dest,call):
  call(commands(ROOT,blender,'solve_secondary_hair_blender.py',['--scene',scene,'--output-dir',dest,'--profile',args.profile,'--enable-hair']))
  proof=read_json(dest/'physics-validation.json')
  if not proof.get('passed') or proof['protected_max_matrix_error']!=0:raise ValueError('原动作保护或发束验证未通过')
 def files(path):return [p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file() and p.suffix not in ('.blend1','.blend2')]
 pair=run.stage('hair-basic',key,run.revision_root,producer,files,
  prepare=canonical_prepare(ROOT,blender,lambda source:source/'model-plan.json',run),
  provenance={'solver_unchanged':True,'full_collision_quality':'not_validated'})
 proof=read_json(pair/'physics-validation.json')
 if not proof['passed'] or proof['world_before_sha256']!=proof['world_after_sha256']:raise ValueError('Cached protection failed')
 atomic_json(out/'latest.json',{'result':{'pair':str(pair),'case':str(pair)},'protection_passed':True,'full_collision_quality':'not_validated'})
print('WORKBENCH_HAIR_COMPLETED',flush=True)
