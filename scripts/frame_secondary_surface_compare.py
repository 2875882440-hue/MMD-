"""Shared complete-actor camera, preserving every incoming source action."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
import bpy,numpy as np
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from solve_secondary_hair_blender import digest_actions,mesh_digest
parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True)
parser.add_argument('--pair-dir',type=Path,required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=args.input_dir.resolve();pair=args.pair_dir.resolve();out=pair/'comparison'
if not pair.is_relative_to(ROOT/'output/secondary_motion') or out.exists():raise ValueError('使用新的comparison目录')
meta=json.loads((root/'input.json').read_text(encoding='utf-8'));plan=json.loads((root/'model-plan.json').read_text(encoding='utf-8'))
tracks=[];minimum=np.full(2,np.inf);maximum=-minimum;source_sha={};checks=[];range_info=None;rotation=None
def actor_meshes():
 return [o for o in bpy.context.scene.objects if o.type=='MESH' and not o.hide_render and (o.name==plan['bindings']['mesh'] or o.name.startswith('MB4_RigidAttachment_'))]
for variant in ('before','after'):
 path=pair/f'{variant}.blend';source_sha[variant]=hashlib.sha256(path.read_bytes()).hexdigest()
 bpy.ops.wm.open_mainfile(filepath=str(path),load_ui=False,use_scripts=False)
 scene=bpy.context.scene;rig=bpy.data.objects[plan['bindings']['armature']]
 if range_info is None:range_info=(scene.frame_start,scene.frame_end,scene.render.fps/scene.render.fps_base)
 assert range_info==(scene.frame_start,scene.frame_end,scene.render.fps/scene.render.fps_base)
 if rotation is None:rotation=scene.camera.rotation_euler.to_matrix();orientation=scene.camera.rotation_euler.copy()
 right=np.asarray(rotation.col[0]);up=np.asarray(rotation.col[1])
 for index,frame in enumerate(range(scene.frame_start,scene.frame_end+1)):
  scene.frame_set(frame);position=rig.matrix_world@rig.pose.bones['腰'].head;track=np.array([position.x,position.y,0.])
  if variant=='before':tracks.append(track)
  else:assert np.array_equal(track,tracks[index])
  for obj in actor_meshes():
   ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
   bounds=np.asarray([obj.matrix_world@Vector(corner) for corner in ev.bound_box])
   projected=np.column_stack(((bounds-track)@right,(bounds-track)@up))
   minimum=np.minimum(minimum,projected.min(0));maximum=np.maximum(maximum,projected.max(0))
  if index%150==0:print('SURFACE_CAMERA_BOUNDS',variant,frame,flush=True)
center=(minimum+maximum)/2;span=maximum-minimum;aspect=scene.render.resolution_x/scene.render.resolution_y
scale=float(max(span[1],span[0]/aspect)*1.12);positions=np.asarray(tracks)+right*center[0]+up*center[1]+np.asarray(rotation.col[2])*6
out.mkdir()
if (pair/'surface-final.mdd').exists():shutil.copy2(pair/'surface-final.mdd',out/'surface-final.mdd')
for variant in ('before','after'):
 bpy.ops.wm.open_mainfile(filepath=str(pair/f'{variant}.blend'),load_ui=False,use_scripts=False)
 scene=bpy.context.scene;camera=scene.camera
 # Preserve the existing camera action as well as all body/hair actions.
 if camera.animation_data and camera.animation_data.action:camera.animation_data.action.use_fake_user=True
 camera.animation_data_clear();camera.rotation_euler=orientation;camera.data.ortho_scale=scale
 for index,frame in enumerate(range(scene.frame_start,scene.frame_end+1)):
  camera.location=positions[index];camera.keyframe_insert(data_path='location',frame=frame)
 camera.animation_data.action.name='MB4_SharedSurfaceComparisonCamera'
 for layer in camera.animation_data.action.layers:
  for strip in layer.strips:
   for bag in getattr(strip,'channelbags',[]):
    for curve in bag.fcurves:
     for key in curve.keyframe_points:key.interpolation='LINEAR'
 for obj in scene.objects:
  for modifier in obj.modifiers:
   if modifier.type=='MESH_CACHE':modifier.filepath=str(Path(bpy.path.abspath(modifier.filepath)).resolve())
 mesh=bpy.data.objects[plan['bindings']['mesh']]
 cache=mesh.modifiers.get('MB4_DynamicSurfaceCache')
 if cache:cache.filepath=str(out/'surface-final.mdd')
 scene.frame_set(scene.frame_start);bpy.context.preferences.filepaths.save_version=0
 bpy.ops.wm.save_as_mainfile(filepath=str(out/f'{variant}.blend'))
 if cache:cache.filepath='//surface-final.mdd';bpy.ops.wm.save_as_mainfile(filepath=str(out/f'{variant}.blend'))
 bpy.ops.wm.open_mainfile(filepath=str(out/f'{variant}.blend'),load_ui=False,use_scripts=False)
 scene=bpy.context.scene;camera=scene.camera;mesh=bpy.data.objects[plan['bindings']['mesh']]
 assert digest_actions(set(meta['original_actions']))==meta['original_actions']
 assert mesh_digest(mesh)==meta['skin_morph_sha256']
 overflow=0.;camera_hash=hashlib.sha256()
 for frame in range(scene.frame_start,scene.frame_end+1):
  scene.frame_set(frame);camera_hash.update(np.asarray(camera.matrix_world,dtype=np.float32).tobytes())
  for obj in actor_meshes():
   ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());bounds=np.asarray([obj.matrix_world@Vector(c) for c in ev.bound_box])
   delta=bounds-np.asarray(camera.location);xy=np.column_stack((delta@right,delta@up))
   overflow=max(overflow,float(np.max(np.abs(xy)/[scale*aspect/2,scale/2]-1)))
 assert overflow==0,'角色边界出画'
 checks.append({'variant':variant,'frames':len(tracks),'out_of_frame_frames':0,'camera_sha256':camera_hash.hexdigest(),'all_original_actions_preserved':True})
 assert source_sha[variant]==hashlib.sha256((pair/f'{variant}.blend').read_bytes()).hexdigest()
assert checks[0]['camera_sha256']==checks[1]['camera_sha256']
report={'passed':True,'same_camera':True,'includes_separate_rigid_objects':True,'checks':checks,'ortho_scale':scale,
 'projected_min':minimum.tolist(),'projected_max':maximum.tolist(),'padding':.12,'source_pair_sha256':source_sha,
 'framing_scope':'all integer frames, complete rendered actor bounds; no continuous-between-frame framing claim'}
(out/'camera.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('SURFACE_CAMERA_PASSED',json.dumps(report),flush=True)
