"""Shared camera over both evaluated tracks; preview only, no scene writes."""
from pathlib import Path
import argparse,hashlib,json,sys
import bpy,numpy as np
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
p=argparse.ArgumentParser();p.add_argument('--pair-dir',type=Path,required=True);p.add_argument('--case',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--all-frames',action='store_true',help='Diagnostic continuous comparison; default remains five keyframes')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);pair=a.pair_dir.resolve();out=a.output_dir.resolve();case=a.case.resolve()
if not all(x.is_relative_to(ROOT/'output/secondary_motion') for x in [pair,out,case]) or out.exists():raise ValueError('New preview directory only')
plan=json.loads((case/'model-plan.json').read_text(encoding='utf-8'));bindings=plan['bindings'];sources={v:hashlib.file_digest((pair/f'{v}.blend').open('rb'),'sha256').hexdigest() for v in ['before','after']}
tracks=[];bounds=[];rotation=None;frame_range=None
for variant in ['before','after']:
 bpy.ops.wm.open_mainfile(filepath=str(pair/f'{variant}.blend'),load_ui=False,use_scripts=False);scene=bpy.context.scene
 rig=bpy.data.objects[bindings['armature']];range_now=(scene.frame_start,scene.frame_end)
 if frame_range is None:frame_range=range_now
 assert frame_range==range_now
 if rotation is None:rotation=scene.camera.rotation_euler.to_matrix() if scene.camera else Vector((1.7,4.5,.4)).to_track_quat('Z','Y').to_matrix()
 for index,frame in enumerate(range(scene.frame_start,scene.frame_end+1)):
  scene.frame_set(frame);hip=rig.matrix_world@rig.pose.bones['腰'].head;track=np.array([hip.x,hip.y,0.])
  if variant=='before':tracks.append(track)
  else:assert np.array_equal(tracks[index],track),'Body trajectory changed'
  deps=bpy.context.evaluated_depsgraph_get()
  for obj in scene.objects:
   if obj.type!='MESH' or obj.hide_render or not (obj.name==bindings['mesh'] or obj.name.startswith('MB4_RigidAttachment_')):continue
   ev=obj.evaluated_get(deps)
   points=np.array([ev.matrix_world@Vector(corner) for corner in ev.bound_box])-track
   bounds.append(points@np.asarray(rotation))
minimum=np.concatenate(bounds).min(axis=0);maximum=np.concatenate(bounds).max(axis=0)
center=(minimum+maximum)/2;scale=max(maximum[1]-minimum[1],(maximum[0]-minimum[0])/.8)*1.14
max_ndc=float(np.max(np.abs((np.concatenate(bounds)[:,:2]-center[:2])/(np.array([.8,1.])*scale*.5))))
assert max_ndc<1.,'Actual actor bounds escape the preview camera'
frames=list(range(frame_range[0],frame_range[1]+1)) if a.all_frames else sorted(set(np.linspace(*frame_range,5).round().astype(int).tolist()))
out.mkdir();cameras={};clipping={}
for variant in ['before','after']:
 bpy.ops.wm.open_mainfile(filepath=str(pair/f'{variant}.blend'),load_ui=False,use_scripts=False);scene=bpy.context.scene
 camera=scene.camera
 if camera is None:
  camera=bpy.data.objects.new('PreviewCamera',bpy.data.cameras.new('PreviewCamera'));scene.collection.objects.link(camera);scene.camera=camera
 camera.animation_data_clear();camera.constraints.clear() if hasattr(camera.constraints,'clear') else None
 for constraint in list(camera.constraints):camera.constraints.remove(constraint)
 camera.rotation_euler=rotation.to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=scale
 scene.render.engine='BLENDER_WORKBENCH';scene.render.resolution_x=480;scene.render.resolution_y=600;scene.render.resolution_percentage=100
 scene.display.shading.light='STUDIO';scene.display.shading.color_type='TEXTURE';scene.display.shading.show_shadows=True
 scene.display.shading.show_cavity=True;scene.display.shading.background_type='WORLD'
 folder=out/variant;folder.mkdir();cameras[variant]=[]
 for frame in frames:
  scene.frame_set(frame);camera.location=rotation@Vector(center+np.array([0,0,4.]))+Vector(tracks[frame-frame_range[0]])
  bpy.context.view_layer.update();cameras[variant].append([list(row) for row in camera.matrix_world])
  scene.render.filepath=str(folder/f'{frame:04d}.png');bpy.ops.render.render(write_still=True)
 assert sources[variant]==hashlib.file_digest((pair/f'{variant}.blend').open('rb'),'sha256').hexdigest()
assert cameras['before']==cameras['after']
proof={'passed':True,'same_camera':True,'source_files_unchanged':True,'frames':frames,'preview_kind':'all integer frames for continuous diagnostic' if a.all_frames else 'five actual rendered keyframes, not continuous video',
 'fps':scene.render.fps/scene.render.fps_base,
 'all_integer_frame_actor_bounds_fit':True,'maximum_absolute_ndc_xy':max_ndc,
 'bounds_scope':'configured character mesh and rigid attachments; excludes hidden source mannequin',
 'orthographic_scale':float(scale),'physics_quality_approval':False,'source_sha256':sources,
 'renderer_sha256':hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest()}
(out/'preview.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
print('WORKBENCH_PREVIEW_PASSED',flush=True)
