"""Fit a shared XY-follow camera to BOTH saved scenes, without changing motion.

The source pair is retained. Derived comparison/before.blend and after.blend
contain the same camera action, driven only by a protected source body bone.
Every-frame evaluated bounds determine a conservative common orthographic fit.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
import bpy,numpy as np
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
parser=argparse.ArgumentParser();parser.add_argument('--pair-dir',type=Path,required=True)
parser.add_argument('--follow-bone',default='腰')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=args.pair_dir.resolve();out=root/'comparison'
if not root.is_relative_to(ROOT/'output/secondary_motion') or out.exists():raise ValueError('必须使用本副本内尚无comparison子目录的产物')
plan=json.loads((root/'model-plan.json').read_text(encoding='utf-8'))
protected={plan['bones'][i]['blender_name'] for i in plan['protected_bone_ids']}
if args.follow_bone not in protected:raise ValueError('跟随相机只能由受保护身体骨骼驱动')
tracks=[];minimum=np.array([np.inf,np.inf]);maximum=-minimum;source_sha={};range_info=None;rotation=None;bounds_checks=[]
for variant in ('before','after'):
 path=root/f'{variant}.blend';source_sha[variant]=hashlib.sha256(path.read_bytes()).hexdigest()
 bpy.ops.wm.open_mainfile(filepath=str(path),load_ui=False,use_scripts=False)
 scene=bpy.context.scene;rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
 if range_info is None:range_info=(scene.frame_start,scene.frame_end,scene.render.fps/scene.render.fps_base)
 assert range_info==(scene.frame_start,scene.frame_end,scene.render.fps/scene.render.fps_base)
 if rotation is None:rotation=scene.camera.rotation_euler.to_matrix();orientation=scene.camera.rotation_euler.copy()
 right=np.asarray(rotation.col[0]);up=np.asarray(rotation.col[1])
 for index,frame in enumerate(range(scene.frame_start,scene.frame_end+1)):
  scene.frame_set(frame);position=rig.matrix_world@rig.pose.bones[args.follow_bone].head
  track=np.array([position.x,position.y,0.])
  if variant=='before':tracks.append(track)
  else:assert np.max(np.abs(track-tracks[index]))==0,'身体跟随轨迹不同'
  evaluated=mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
  bounds=np.array([mesh.matrix_world@Vector(corner) for corner in evaluated.bound_box])
  projected=np.column_stack(((bounds-track)@right,(bounds-track)@up))
  minimum=np.minimum(minimum,projected.min(axis=0));maximum=np.maximum(maximum,projected.max(axis=0))
  if frame in (scene.frame_start,571,889,scene.frame_end):
   data=evaluated.to_mesh();vertices=np.empty(len(data.vertices)*3,dtype=np.float32);data.vertices.foreach_get('co',vertices)
   world=np.asarray(mesh.matrix_world);vertices=vertices.reshape(-1,3)@world[:3,:3].T+world[:3,3]
   evaluated.to_mesh_clear()
   assert np.all(vertices.min(0)>=bounds.min(0)-2e-6) and np.all(vertices.max(0)<=bounds.max(0)+2e-6),'变形边界未覆盖实际网格'
   bounds_checks.append({'variant':variant,'frame':frame,'actual_vertices_within_evaluated_bounds':True})
center=(minimum+maximum)/2;span=maximum-minimum;aspect=scene.render.resolution_x/scene.render.resolution_y
scale=float(max(span[1],span[0]/aspect)*1.1)
center_offset=right*center[0]+up*center[1]
back=np.asarray(rotation.col[2])*5
camera_positions=np.asarray(tracks)+center_offset+back
out.mkdir()
metadata={'method':'shared_protected_body_XY_follow, fixed orientation and orthographic scale',
          'follow_bone':args.follow_bone,'orientation':list(orientation),'ortho_scale':scale,
          'positions_sha256':hashlib.sha256(camera_positions.tobytes()).hexdigest(),
          'source_scene_sha256':source_sha,'evaluated_bounds_frames_per_scene':len(tracks),
          'actual_mesh_bounds_checks':bounds_checks,'projected_union_min':minimum.tolist(),
          'projected_union_max':maximum.tolist(),'padding_fraction':.1,'source_pair_unchanged':True}
for variant in ('before','after'):
 bpy.ops.wm.open_mainfile(filepath=str(root/f'{variant}.blend'),load_ui=False,use_scripts=False)
 camera=bpy.context.scene.camera;camera.rotation_euler=orientation;camera.data.ortho_scale=scale
 for index,frame in enumerate(range(bpy.context.scene.frame_start,bpy.context.scene.frame_end+1)):
  camera.location=camera_positions[index].tolist();camera.keyframe_insert(data_path='location',frame=frame)
 action=camera.animation_data.action;action.name='MBH_SharedComparisonCamera'
 for layer in action.layers:
  for strip in layer.strips:
   for bag in getattr(strip,'channelbags',[]):
    for curve in bag.fcurves:
     for key in curve.keyframe_points:key.interpolation='LINEAR'
 bpy.context.scene.frame_set(bpy.context.scene.frame_start)
 bpy.ops.wm.save_as_mainfile(filepath=str(out/f'{variant}.blend'))
 assert source_sha[variant]==hashlib.sha256((root/f'{variant}.blend').read_bytes()).hexdigest()
for name in ('profile.json','model-plan.json','hair-nodes.json','hair-simulation.npz'):
 shutil.copy2(root/name,out/name)
report=json.loads((root/'physics-validation.json').read_text(encoding='utf-8'))
report.update(physics_source_report=str(root/'physics-validation.json'),comparison_only_camera_changed=True,
              solver_camera=report['camera'],camera=metadata)
(out/'physics-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'camera.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
print('SHARED_CAMERA_FIT',json.dumps(metadata,ensure_ascii=False),flush=True)
