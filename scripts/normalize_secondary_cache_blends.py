"""Path-only canonical snapshots for the immutable cache, with real readback."""
from pathlib import Path
import argparse,json,sys,hashlib
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from solve_secondary_hair_blender import digest_actions,mesh_digest,protected_matrices
from motionbridge.blender_secondary_rigid import evaluated_mesh_records
parser=argparse.ArgumentParser();parser.add_argument('--source-dir',type=Path,required=True);parser.add_argument('--payload',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=args.source_dir.resolve();payload=args.payload.resolve();plan=json.loads(args.plan.read_text(encoding='utf-8'))
if not source.is_relative_to(ROOT/'output/secondary_motion') or not payload.is_relative_to(ROOT/'data/secondary-cache'):raise ValueError('Cache normalization is confined to the development copy')
names=[plan['bones'][i]['blender_name'] for i in plan['protected_bone_ids']];records=[]
def proof():
 scene=bpy.context.scene;rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
 frames=sorted(set([float(scene.frame_start),min(float(scene.frame_end),(scene.frame_start+scene.frame_end)//2+.5),float(scene.frame_end)]))
 matrices=protected_matrices(rig,names,frames);geometry=hashlib.sha256()
 for frame in frames:
  scene.frame_set(int(frame),subframe=frame-int(frame))
  for obj in sorted([o for o in scene.objects if o.type=='MESH' and (o.name==mesh.name or o.name.startswith('MB4_RigidAttachment_'))],key=lambda o:o.name):
   p,ids,f,fids=evaluated_mesh_records(obj);geometry.update(obj.name.encode());geometry.update(p.tobytes());geometry.update(f.tobytes())
 scene.frame_set(scene.frame_start)
 return {'actions':digest_actions(),'skin_morph':mesh_digest(mesh),'matrix':hashlib.sha256(matrices.tobytes()).hexdigest(),'geometry':geometry.hexdigest(),'sample_frames':frames}
for target in sorted(payload.glob('*.blend')):
 original=source/target.name
 bpy.ops.wm.open_mainfile(filepath=str(original),load_ui=False,use_scripts=False);before=proof()
 bpy.ops.file.make_paths_absolute()
 for obj in bpy.context.scene.objects:
  for modifier in obj.modifiers:
   if modifier.type=='MESH_CACHE':
    path=Path(bpy.path.abspath(modifier.filepath)).resolve()
    if path.is_relative_to(source) and (payload/path.relative_to(source)).is_file():modifier.filepath='//'+path.relative_to(source).as_posix()
 for cache in bpy.data.cache_files:
  path=Path(bpy.path.abspath(cache.filepath)).resolve()
  if path.is_relative_to(source) and (payload/path.relative_to(source)).is_file():cache.filepath='//'+path.relative_to(source).as_posix()
 bpy.ops.file.pack_all();bpy.context.preferences.filepaths.save_version=0
 bpy.ops.wm.save_as_mainfile(filepath=str(target),relative_remap=False)
 bpy.ops.wm.open_mainfile(filepath=str(target),load_ui=False,use_scripts=False);after=proof()
 assert before==after,'Canonical cache changed animation/data/evaluated geometry'
 records.append({'file':target.name,'passed':True,'sample_frames':before['sample_frames'],'actions_unchanged':len(before['actions']),'skin_morph_sha256':before['skin_morph'],'matrix_sha256':before['matrix'],'geometry_sha256':before['geometry']})
(payload/'cache-path-verification.json').write_text(json.dumps({'passed':True,'scope':'path/packing only; all actions/skin/morph and 3 actual geometry/matrix samples; physics quality unchanged','files':records},ensure_ascii=False,indent=2),encoding='utf-8')
print('CANONICAL_CACHE_PASSED',len(records),flush=True)
