"""Adopt an existing audited, frozen dev.4 run; never rerun physics."""
from pathlib import Path
import argparse,json,sys,subprocess,uuid
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from motionbridge.secondary_cache import ArtifactCache,file_sha,atomic_json
from motionbridge.secondary_pipeline import CAPTURE,CLOTH,HAIR,read,pipeline_keys,canonical_prepare,bundle_files,commands
from motionbridge.secondary_surface_config import surface_settings
parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--profile',default='rin')
parser.add_argument('--adopt-audited-dev4',action='store_true',required=True)
args=parser.parse_args();case=args.input_dir.resolve()
if not case.is_relative_to(ROOT/'output/secondary_motion'):raise ValueError('只能读取本副本已验证产物')
profile=read(ROOT/f'profiles/secondary_motion/{args.profile}.json');meta=read(case/'input.json')
if read(case/'profile.json')!=profile:raise ValueError('模型/物理参数改变，不能给旧结果换当前参数标签')
source=Path(meta['source']);assert file_sha(source)==meta['source_sha256']
blender=Path(read(ROOT/'config.json')['blender_path']);out=ROOT/'validation/secondary-motion-dev5'/f'seed-preflight-{uuid.uuid4().hex[:8]}.json'
subprocess.run(commands(ROOT,blender,'preflight_secondary_source.py',['--scene',source,'--output',out,'--profile',args.profile]),check=True)
stamp=read(out);keys=pipeline_keys(ROOT,stamp,profile);cache=ArtifactCache(ROOT/'data/secondary-cache')
prepare=canonical_prepare(ROOT,blender,lambda _:case/'model-plan.json')
mapping=[('capture',case,CAPTURE),('cloth',case/'cloth-native-full',CLOTH),('hair',case/'hair-coupled-full',HAIR),
 ('bundle-coupled',case/'surface-result',None),('bundle-preserved',case/'surface-preserve-hair',None),
 ('cloth-guarded',case/'cloth-guarded',CLOTH+['component-guard.json']),('bundle-guarded',case/'surface-guarded',None)]
assert read(case/'cloth-native-full/cloth-validation.json')['settings']==surface_settings(profile)
assert read(case/'hair-coupled-full/hair-validation.json')['matrix_max_error']==0
records=[]
for stage,origin,names in mapping:
 if stage.startswith('bundle'):
  assert read(origin/'surface-audit.json')['protection_passed'] and read(origin/'surface-validation.json')['passed']
  names=bundle_files(origin)
 with cache.reserve(stage,keys[stage]):
  entry=cache.publish(stage,keys[stage],origin,names,prepare=prepare,
      provenance={'adopted_from_audited_dev4':str(origin),'legacy_key_inference':True,'source_sha256':meta['source_sha256'],
                  'physics_quality':'experimental_rejected' if stage.startswith('bundle') else 'not_approved_by_cache',
                  'note':'Frozen original computation, current equivalent code fingerprint; not a fresh execution of physics'})
 records.append({'stage':stage,'key':keys[stage],'files':len(entry['files'])});print('SEEDED',stage,flush=True)
atomic_json(ROOT/'validation/secondary-motion-dev5/seed-verification.json',{'passed':True,'records':records,'source_unchanged':file_sha(source)==meta['source_sha256'],'source_case':str(case),'physics_was_rerun':False})
