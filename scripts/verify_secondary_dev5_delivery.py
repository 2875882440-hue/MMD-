"""Reopen delivered files at their final paths; no exports or source writes."""
from pathlib import Path
import hashlib,json,sys
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from solve_secondary_hair_blender import digest_actions,mesh_digest,protected_matrices
from motionbridge.blender_secondary_rigid import evaluated_mesh_records

case=ROOT/'output/secondary_motion/dev4/rin-dance'
latest=json.loads((ROOT/'output/secondary_motion/dev5/managed-export/latest.json').read_text(encoding='utf-8'))
delivery=Path(latest['delivery']);source=case/'surface-guarded/comparison/after.blend'
plan=json.loads((case/'model-plan.json').read_text(encoding='utf-8'))
names=[plan['bones'][i]['blender_name'] for i in plan['protected_bone_ids']]
source_sha=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
frames=[1.,450.5,849.,900.];allframes=np.arange(1,900.01,.5).tolist()
def actors():
 return sorted([o for o in bpy.context.scene.objects if o.type=='MESH' and not o.hide_render and (o.name==plan['bindings']['mesh'] or o.name.startswith('MB4_RigidAttachment_'))],key=lambda o:o.name)
def geometry():
 result={}
 for frame in frames:
  bpy.context.scene.frame_set(int(frame),subframe=frame-int(frame))
  result[str(frame)]={o.name:evaluated_mesh_records(o)[0].copy() for o in actors()}
 return result
def open_scene(path):
 bpy.ops.wm.open_mainfile(filepath=str(path),load_ui=False,use_scripts=False)
open_scene(source);scene=bpy.context.scene
rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
reference=geometry();matrices=protected_matrices(rig,names,allframes);scene.frame_set(1)
actions=digest_actions();skin=mesh_digest(mesh);shape_count=len(mesh.data.shape_keys.key_blocks)
results={}
for name,path in [('blender',delivery/'blender/scene.blend'),('alembic',delivery/'alembic/readback.blend')]:
 open_scene(path);scene=bpy.context.scene;paths=[]
 for obj in scene.objects:
  for modifier in obj.modifiers:
   if modifier.type=='MESH_CACHE':paths.append(Path(bpy.path.abspath(modifier.filepath)).resolve())
 for cache in bpy.data.cache_files:paths.append(Path(bpy.path.abspath(cache.filepath)).resolve())
 assert all(p.is_relative_to(delivery) and p.is_file() for p in paths),'Delivered cache escaped or missing'
 observed=geometry();error=0.
 for f in reference:
  assert observed[f].keys()==reference[f].keys()
  for obj in reference[f]:
   assert reference[f][obj].shape==observed[f][obj].shape
   error=max(error,float(np.linalg.norm(reference[f][obj]-observed[f][obj],axis=1).max(initial=0)))
 assert error<1e-5
 result={'reopened':True,'resolved_caches':[str(p.relative_to(delivery)) for p in paths],
         'actual_geometry_samples':frames,'actual_geometry_max_error_m':error,'objects':len(actors())}
 if name=='blender':
  rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
  after=protected_matrices(rig,names,allframes);assert np.array_equal(matrices,after)
  scene.frame_set(1);assert digest_actions(set(actions))==actions and mesh_digest(mesh)==skin
  assert len(mesh.data.shape_keys.key_blocks)==shape_count
  result.update(protected_bones=len(names),matrix_samples=len(allframes),matrix_error=0,
                original_action_count=len(actions),shape_keys=shape_count,skin_and_morph_unchanged=True)
 results[name]=result
 assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==source_sha
report={'passed':True,'source_file_unchanged':True,'delivery':str(delivery),'results':results,
        'visual_review':'Alembic frame849 and FBX-character frame450 reviewed; sampled views only',
        'C4D_runtime_verified':False,'physics_quality_approved':False}
(ROOT/'validation/secondary-motion-dev5/delivery-readback.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('FINAL_DELIVERY_READBACK_PASSED',flush=True)
