"""World-pose FBX retarget to a new copy of the original configured character."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from solve_secondary_hair_blender import digest_actions,mesh_digest
from motionbridge.blender_secondary_rigid import evaluated_mesh_records
parser=argparse.ArgumentParser();parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--fbx',type=Path,required=True);parser.add_argument('--input-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--take',type=int);parser.add_argument('--render-check',action='store_true')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=args.scene.resolve();fbx=args.fbx.resolve();out=args.output_dir.resolve()
if not source.is_relative_to(ROOT) or not fbx.is_relative_to(ROOT/'output/secondary_motion') or not out.is_relative_to(ROOT/'output/secondary_motion') or out.exists():raise ValueError('Use a new isolated target copy')
source_sha=hashlib.file_digest(source.open('rb'),'sha256').hexdigest();started=time.perf_counter();plan=json.loads((args.input_dir/'model-plan.json').read_text(encoding='utf-8'))
bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False);scene=bpy.context.scene;rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
start=scene.frame_start;end=min(scene.frame_end,start+args.take-1) if args.take else scene.frame_end;frames=np.arange(start,end+.01,.5).tolist();scene.frame_set(start);actions=digest_actions();skin=mesh_digest(mesh)
old_objects=set(bpy.data.objects);bpy.ops.import_scene.fbx(filepath=str(fbx),use_anim=True,anim_offset=0.,automatic_bone_orientation=False)
imported=next(o for o in bpy.data.objects if o not in old_objects and o.type=='ARMATURE');names=[b.name for b in imported.data.bones if b.name in rig.data.bones]
assert all(n in rig.data.bones for n in names),'FBX does not match the target character'
target=rig.copy();target.data=rig.data.copy();target.name='MB5_FBXRetargetRig';scene.collection.objects.link(target);target.parent=None;target.matrix_world=rig.matrix_world.copy();target.animation_data_clear();target.data.animation_data_clear();target.constraints.clear()
for pose in target.pose.bones:
 for constraint in list(pose.constraints):pose.constraints.remove(constraint)
 pose.rotation_mode='QUATERNION'
model=mesh.copy();model.data=mesh.data.copy();model.name='MB5_FBXRetargetMesh';scene.collection.objects.link(model)
if model.parent==rig:model.parent=target
model.matrix_world=mesh.matrix_world.copy();model.hide_render=False;model.hide_set(False)
for modifier in model.modifiers:
 if modifier.type=='ARMATURE' and modifier.object==rig:modifier.object=target
inverse=target.matrix_world.inverted();basis=np.empty((len(frames),len(names),10),dtype=np.float32);last={};maximum_shear=0.
for f,frame in enumerate(frames):
 scene.frame_set(int(frame),subframe=frame-int(frame));desired={n:inverse@(imported.matrix_world@imported.pose.bones[n].matrix) for n in names}
 for i,n in enumerate(names):
  b=target.data.bones[n];kwargs={'parent_matrix':desired[b.parent.name],'parent_matrix_local':b.parent.matrix_local} if b.parent else {}
  local=b.convert_local_to_pose(desired[n],b.matrix_local,invert=True,**kwargs);l,q,s=local.decompose();q.normalize()
  if n in last and q.dot(last[n])<0:q.negate()
  last[n]=q.copy();basis[f,i]=[*l,*q,*s]
  from mathutils import Matrix
  maximum_shear=max(maximum_shear,float(np.max(np.abs(np.asarray(local)-np.asarray(Matrix.LocRotScale(l,q,s))))))
assert maximum_shear<.0002,'Retarget would discard significant local shear'
action=bpy.data.actions.new('MB5_FBXWorldRetarget');slot=action.slots.new(id_type='OBJECT',name=target.name);layer=action.layers.new('World pose mapping');strip=layer.strips.new(type='KEYFRAME');bag=strip.channelbags.new(slot)
for i,n in enumerate(names):
 for path,offset,width in [('location',0,3),('rotation_quaternion',3,4),('scale',7,3)]:
  for c in range(width):
   curve=bag.fcurves.new(data_path='pose.bones['+json.dumps(n,ensure_ascii=False)+'].'+path,index=c);curve.keyframe_points.add(len(frames));curve.keyframe_points.foreach_set('co',np.column_stack((frames,basis[:,i,offset+c])).ravel())
   for k in curve.keyframe_points:k.interpolation='LINEAR'
   curve.update()
target.animation_data_create();target.animation_data.action=action;target.animation_data.action_slot=slot
world_error=0.;vertex_error=0.;sample_indices=set(np.linspace(0,len(frames)-1,min(17,len(frames))).round().astype(int).tolist())
for f,frame in enumerate(frames):
 scene.frame_set(int(frame),subframe=frame-int(frame))
 actual=np.asarray([target.matrix_world@target.pose.bones[n].matrix for n in names]);expected=np.asarray([rig.matrix_world@rig.pose.bones[n].matrix for n in names]);world_error=max(world_error,float(np.max(np.abs(actual-expected))))
 if f in sample_indices:
  a,_,_,_=evaluated_mesh_records(mesh);b,_,_,_=evaluated_mesh_records(model);assert a.shape==b.shape;vertex_error=max(vertex_error,float(np.linalg.norm(a-b,axis=1).max()))
 if f%300==0:print('FBX_CHARACTER_READBACK',f+1,len(frames),flush=True)
scene.frame_set(start);assert digest_actions(set(actions))==actions and mesh_digest(mesh)==skin
assert mesh_digest(model)==skin,'Original skin/81 keys were not copied faithfully'
saved_geometry={}
for frame in sorted(set([float(start),(start+end)//2+.5 if start<end else float(start),float(end)])):
 scene.frame_set(int(frame),subframe=frame-int(frame));saved_geometry[frame]=evaluated_mesh_records(model)[0]
mesh.hide_render=True;mesh.hide_set(True);imported.hide_render=True;imported.hide_set(True)
out.mkdir();scene.frame_end=end;bpy.ops.file.make_paths_absolute();bpy.ops.file.pack_all();bpy.context.preferences.filepaths.save_version=0
for obj in scene.objects:
 for modifier in obj.modifiers:
  if modifier.type=='MESH_CACHE':
   path=Path(bpy.path.abspath(modifier.filepath)).resolve();digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest();dest=out/'cache'/(digest+path.suffix)
   dest.parent.mkdir(exist_ok=True)
   if not dest.exists():shutil.copyfile(path,dest)
   modifier.filepath=str(dest)
bpy.ops.wm.save_as_mainfile(filepath=str(out/'retargeted-character.blend'),relative_remap=False)
for obj in scene.objects:
 for modifier in obj.modifiers:
  if modifier.type=='MESH_CACHE':modifier.filepath='//'+Path(modifier.filepath).relative_to(out).as_posix()
bpy.ops.wm.save_as_mainfile(filepath=str(out/'retargeted-character.blend'),relative_remap=False)
bpy.ops.wm.open_mainfile(filepath=str(out/'retargeted-character.blend'),load_ui=False,use_scripts=False);scene=bpy.context.scene;model=bpy.data.objects['MB5_FBXRetargetMesh']
saved_error=0.
for frame,expected in saved_geometry.items():
 scene.frame_set(int(frame),subframe=frame-int(frame));actual=evaluated_mesh_records(model)[0];saved_error=max(saved_error,float(np.linalg.norm(actual-expected,axis=1).max()))
assert saved_error<1e-5,'Saved retarget project changed or lost cached resources'
if args.render_check:
 shots=out/'shots';shots.mkdir()
 for frame in sorted(set([start,(start+end)//2,min(end,849),end])):
  scene.frame_set(frame);scene.render.filepath=str(shots/f'{frame:04d}.png');bpy.ops.render.render(write_still=True)
passed=world_error<.002 and vertex_error<.0005
report={'passed':passed,'source_file_unchanged':hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==source_sha,'source_sha256':source_sha,'source_original_actions_skin_morphs_unchanged':True,
 'saved_geometry_max_error_m':saved_error,'mapped_bones':len(names),'sample_frames':len(frames),'mesh_sample_indices':sorted(map(int,sample_indices)),'world_matrix_max_error':world_error,'world_vertex_max_error_m':vertex_error,
 'shape_keys_in_copy':len(model.data.shape_keys.key_blocks),'method':'map FBX world poses to the original target bind/rest matrices, bake a new Action on a new rig/mesh copy',
 'existing_vertex_cache_note':'Any target vertex caches remain target resources; FBX does not contain cloth vertices','physics_quality_approval':False,'C4D_runtime_verified':False,'seconds':time.perf_counter()-started}
(out/'fbx-character-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');assert passed,report
print('FBX_CHARACTER_PASSED',world_error,vertex_error,flush=True)
