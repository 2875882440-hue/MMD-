"""Export model-owned rigid attachments as animated FBX objects, read back."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import bpy,numpy as np
from mathutils.kdtree import KDTree
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from solve_secondary_hair_blender import digest_actions,mesh_digest
parser=argparse.ArgumentParser();parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--input-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--take',type=int)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=args.scene.resolve();out=args.output_dir.resolve();case=args.input_dir.resolve()
if not source.is_relative_to(ROOT) or not out.is_relative_to(ROOT/'output/secondary_motion') or out.exists():raise ValueError('New isolated export directory required')
sha=hashlib.file_digest(source.open('rb'),'sha256').hexdigest();started=time.perf_counter();bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
scene=bpy.context.scene;start=scene.frame_start;end=min(scene.frame_end,start+args.take-1) if args.take else scene.frame_end;frames=np.arange(start,end+.01,.5).tolist();fps=scene.render.fps/scene.render.fps_base
objects=sorted([o for o in scene.objects if o.type=='MESH' and o.name.startswith('MB4_RigidAttachment_')],key=lambda o:o.name);out.mkdir()
if not objects:
 (out/'rigid-fbx-validation.json').write_text(json.dumps({'passed':True,'skipped':True,'reason':'source has no separate rigid attachment objects','source_sha256':sha},indent=2),encoding='utf-8');raise SystemExit(0)
names=[o.name for o in objects];scene.frame_set(start);actions=digest_actions()
plan=json.loads((case/'model-plan.json').read_text(encoding='utf-8'));mesh=bpy.data.objects[plan['bindings']['mesh']];skin=mesh_digest(mesh)
local={o.name:np.asarray([v.co[:] for v in o.data.vertices],dtype=np.float32) for o in objects}
triangles={o.name:len(o.data.polygons) for o in objects}
assert all(not o.modifiers and not o.data.shape_keys for o in objects),'Only immutable local rigid meshes are supported'
matrices=np.empty((len(frames),len(objects),4,4),dtype=np.float32)
for f,frame in enumerate(frames):
 scene.frame_set(int(frame),subframe=frame-int(frame))
 for i,obj in enumerate(objects):matrices[f,i]=np.asarray(obj.matrix_world)
bpy.ops.object.select_all(action='DESELECT')
for obj in objects:obj.select_set(True)
scene.frame_start=start;scene.frame_end=end;scene.frame_set(start)
bpy.ops.export_scene.fbx(filepath=str(out/'rigid-attachments.fbx'),use_selection=True,object_types={'MESH'},add_leaf_bones=False,
 bake_anim=True,bake_anim_use_nla_strips=False,bake_anim_use_all_actions=False,bake_anim_simplify_factor=0.,bake_anim_step=.5,
 axis_forward='-Z',axis_up='Y',apply_unit_scale=True,path_mode='STRIP',embed_textures=False)
scene.frame_set(start);assert digest_actions(set(actions))==actions and mesh_digest(mesh)==skin
np.savez(out/'rigid-reference.npz',frames=frames,matrices=matrices,names=np.asarray(names))
bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene;scene.render.fps=round(fps);scene.render.fps_base=round(fps)/fps
bpy.ops.import_scene.fbx(filepath=str(out/'rigid-attachments.fbx'),use_anim=True,anim_offset=0.,automatic_bone_orientation=False)
imported={o.name:o for o in scene.objects if o.type=='MESH'};assert set(names)==set(imported)
scene.frame_set(start);correction={n:np.linalg.inv(np.asarray(imported[n].matrix_world))@matrices[0,i] for i,n in enumerate(names)}
trajectory_error=0.;geometry_error=0.;samples=set(np.linspace(0,len(frames)-1,min(9,len(frames))).round().astype(int).tolist())
def nearest_error(a,b):
 tree=KDTree(len(b))
 for i,p in enumerate(b):tree.insert(p,i)
 tree.balance();return max((tree.find(p)[2] for p in a),default=0.)
for f,frame in enumerate(frames):
 scene.frame_set(int(frame),subframe=frame-int(frame))
 for i,name in enumerate(names):
  obj=imported[name];matrix=np.asarray(obj.matrix_world);trajectory_error=max(trajectory_error,float(np.max(np.abs(matrix@correction[name]-matrices[f,i]))))
  if f in samples:
   assert len(obj.data.polygons)==triangles[name]
   p=np.asarray([v.co[:] for v in obj.data.vertices]);actual=p@matrix[:3,:3].T+matrix[:3,3];expected=local[name]@matrices[f,i,:3,:3].T+matrices[f,i,:3,3]
   geometry_error=max(geometry_error,nearest_error(actual,expected),nearest_error(expected,actual))
 if f%300==0:print('RIGID_FBX_READBACK',f+1,len(frames),flush=True)
passed=trajectory_error<.002 and geometry_error<.0001
report={'passed':passed,'skipped':False,'source':str(source),'source_sha256':sha,'source_file_unchanged':hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==sha,
 'source_actions_skin_morphs_unchanged':True,'objects':len(names),'integer_frames':end-start+1,'sample_frames':len(frames),'sampling_step':.5,
 'maximum_basis_aligned_trajectory_error':trajectory_error,'geometry_world_point_error_m':geometry_error,'geometry_sample_indices':sorted(map(int,samples)),
 'geometry_method':'all rigid world trajectories plus bidirectional vertex nearest distances at listed samples; no mesh deformation',
 'physics_quality_approval':False,'free_6DOF_physics_added':False,'C4D_runtime_verified':False,'materials':'UV/material metadata carried; use Blender package/materials.json assets for manual shader relinking','seconds':time.perf_counter()-started}
(out/'rigid-fbx-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');assert passed,report
print('RIGID_FBX_PASSED',trajectory_error,geometry_error,flush=True)
