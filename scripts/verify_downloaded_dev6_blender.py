"""Native reopening of the downloaded/extracted package on a different drive."""
from pathlib import Path
import hashlib,json,sys
import bpy,numpy as np
R=Path(r'E:\MotionBridge_Physics');sys.dont_write_bytecode=True;sys.path[:0]=[str(R),str(R/'scripts')]
from solve_secondary_hair_blender import digest_actions,mesh_digest,protected_matrices
from motionbridge.blender_secondary_rigid import evaluated_mesh_records
job=json.loads((R/'data/physics-jobs/secondary-jobs/45b463b174714a6ca642fe413ae0467b/job.json').read_text(encoding='utf-8'))
source=Path(job['result']['pair'])/'after.blend';dest=Path(r'D:\codex\danap\output\playwright\dev6-unpacked-cloth')
plan=json.loads((Path(job['result']['case'])/'model-plan.json').read_text(encoding='utf-8'));names=[plan['bones'][i]['blender_name'] for i in plan['protected_bone_ids']]
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest();source_sha=sha(source);frames=np.arange(1,90.01,.5).tolist();samples=[1.,45.5,90.]
def geometry():
 result={}
 for f in samples:
  bpy.context.scene.frame_set(int(f),subframe=f-int(f));result[f]={}
  for o in bpy.context.scene.objects:
   if o.type=='MESH' and not o.hide_render and (o.name==plan['bindings']['mesh'] or o.name.startswith('MB4_RigidAttachment_')):result[f][o.name]=evaluated_mesh_records(o)[0].copy()
 return result
bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
before=protected_matrices(rig,names,frames);reference=geometry();bpy.context.scene.frame_set(1);actions=digest_actions();skin=mesh_digest(mesh)
bpy.ops.wm.open_mainfile(filepath=str(dest/'blender/scene.blend'),load_ui=False,use_scripts=False)
rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
after=protected_matrices(rig,names,frames);assert np.array_equal(before,after)
observed=geometry();error=0.
for f in samples:
 assert reference[f].keys()==observed[f].keys()
 for name in reference[f]:error=max(error,float(np.linalg.norm(reference[f][name]-observed[f][name],axis=1).max(initial=0)))
assert error<1e-6
bpy.context.scene.frame_set(1);assert digest_actions(set(actions))==actions and mesh_digest(mesh)==skin
caches=[]
for o in bpy.context.scene.objects:
 for m in o.modifiers:
  if m.type=='MESH_CACHE':
   path=Path(bpy.path.abspath(m.filepath)).resolve();assert path.is_relative_to(dest) and path.is_file();caches.append(str(path.relative_to(dest)))
assert caches and sha(source)==source_sha
proof={'passed':True,'actual_browser_zip_reopened_on_D':True,'source_unchanged':True,'protected_bones':len(names),'samples':len(frames),'protected_matrix_error':0,
 'original_actions_unchanged':len(actions),'original_skin_and_morphs_unchanged':True,'shape_keys':len(mesh.data.shape_keys.key_blocks),
 'actual_geometry_samples':samples,'actual_geometry_max_error_m':error,'actual_geometry_objects':len(reference[1.]),'relative_cache_inside_extracted_package':caches,
 'physics_quality_approved':False,'C4D_runtime_verified':False}
(R/'validation/secondary-motion-dev6/downloaded-package-readback.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
print('DOWNLOADED_CLOTH_PACKAGE_REOPENED',len(names),len(frames),error)
