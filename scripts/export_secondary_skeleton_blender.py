"""Bake evaluated secondary bones into a new independent FBX, then read back.

Source files/actions/skin/morphs are read only. This FBX is skeleton-only;
vertex cloth requires Alembic or the separately preserved Blender package.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import bpy,numpy as np
from mathutils import Matrix,Vector,Quaternion
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from solve_secondary_hair_blender import digest_actions,mesh_digest
from motionbridge.secondary_hair import select_hair
parser=argparse.ArgumentParser();parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--input-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--take',type=int)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=args.scene.resolve();case=args.input_dir.resolve();out=args.output_dir.resolve()
if not source.is_relative_to(ROOT) or not out.is_relative_to(ROOT/'output/secondary_motion') or out.exists():raise ValueError('New isolated export directory required')
plan=json.loads((case/'model-plan.json').read_text(encoding='utf-8'));profile=json.loads((case/'profile.json').read_text(encoding='utf-8'))
source_sha=hashlib.file_digest(source.open('rb'),'sha256').hexdigest();started=time.perf_counter()
bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False);scene=bpy.context.scene
rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
start=scene.frame_start;end=min(scene.frame_end,start+args.take-1) if args.take else scene.frame_end
frames=np.arange(start,end+.01,.5).tolist();scene.frame_set(start)
actions=digest_actions();skin=mesh_digest(mesh);shape_count=len(mesh.data.shape_keys.key_blocks) if mesh.data.shape_keys else 0;source_object_rest=rig.matrix_world.copy()
protected=[plan['bones'][i]['blender_name'] for i in plan['protected_bone_ids']]
hair=[plan['bones'][i]['blender_name'] for i in select_hair(plan,profile['hair_dynamics'])]
weighted={plan['bones'][int(b)]['blender_name'] for row in plan['geometry']['skin_weights'] for b in row}
selected=set(protected+hair)|weighted
for name in list(selected):
 parent=rig.data.bones[name].parent
 while parent:selected.add(parent.name);parent=parent.parent
depth=lambda b:0 if b.parent is None else 1+depth(b.parent)
names=sorted(selected,key=lambda n:(depth(rig.data.bones[n]),n));index={n:i for i,n in enumerate(names)}
world=np.empty((len(frames),len(names),4,4),dtype=np.float32);basis=np.empty((len(frames),len(names),10),dtype=np.float32);root_values=np.empty((len(frames),10),dtype=np.float32)
rest_world={n:source_object_rest@rig.data.bones[n].matrix_local for n in names};maximum_shear=0.;last_quats={}
for f,frame in enumerate(frames):
 scene.frame_set(int(frame),subframe=frame-int(frame));root_matrix=rig.matrix_world.copy();l,q,s=root_matrix.decompose();root_values[f]=[*l,*q,*s]
 for i,name in enumerate(names):
  b=rig.data.bones[name];pose=rig.pose.bones[name].matrix.copy();world[f,i]=np.asarray(root_matrix@pose)
  kwargs={'parent_matrix':rig.pose.bones[b.parent.name].matrix,'parent_matrix_local':b.parent.matrix_local} if b.parent else {}
  local=b.convert_local_to_pose(pose,b.matrix_local,invert=True,**kwargs);l,q,s=local.decompose();q.normalize()
  if name in last_quats and q.dot(last_quats[name])<0:q.negate()
  last_quats[name]=q.copy();basis[f,i]=[*l,*q,*s]
  recomposed=Matrix.LocRotScale(l,q,s);maximum_shear=max(maximum_shear,float(np.max(np.abs(np.asarray(local)-np.asarray(recomposed)))))
 if f%300==0:print('FBX_CAPTURE',f+1,len(frames),flush=True)
if maximum_shear>1e-4:raise ValueError(f'Source requires non-LRS shear; do not silently discard it: {maximum_shear}')
scene.frame_set(start);assert digest_actions(set(actions))==actions and mesh_digest(mesh)==skin
out.mkdir();np.savez(out/'skeleton-reference.npz',frames=frames,world=world,names=np.asarray(names))
# Work only on a new armature; source pose constraints and curves remain intact.
copy=rig.copy();copy.data=rig.data.copy();copy.name='MB5_BakedSkeleton';scene.collection.objects.link(copy)
copy.parent=None;copy.animation_data_clear();copy.data.animation_data_clear();copy.constraints.clear();copy.hide_viewport=False;copy.hide_set(False)
bpy.ops.object.select_all(action='DESELECT');copy.select_set(True);bpy.context.view_layer.objects.active=copy
for pose in copy.pose.bones:
 for constraint in list(pose.constraints):pose.constraints.remove(constraint)
 pose.rotation_mode='QUATERNION'
bpy.ops.object.mode_set(mode='EDIT')
for bone in list(copy.data.edit_bones):
 if bone.name not in selected:copy.data.edit_bones.remove(bone)
bpy.ops.object.mode_set(mode='OBJECT')
for p in copy.pose.bones:p.rotation_mode='QUATERNION'
copy.rotation_mode='QUATERNION'
action=bpy.data.actions.new('MB5_BakedSecondaryMotion');slot=action.slots.new(id_type='OBJECT',name=copy.name);layer=action.layers.new('Evaluated motion');strip=layer.strips.new(type='KEYFRAME');bag=strip.channelbags.new(slot)
def curves(prefix,values):
 for path,offset,width in [('location',0,3),('rotation_quaternion',3,4),('scale',7,3)]:
  for component in range(width):
   curve=bag.fcurves.new(data_path=prefix+path,index=component);curve.keyframe_points.add(len(frames));curve.keyframe_points.foreach_set('co',np.column_stack((frames,values[:,offset+component])).ravel())
   for k in curve.keyframe_points:k.interpolation='LINEAR'
   curve.update()
curves('',root_values)
for i,name in enumerate(names):curves('pose.bones['+json.dumps(name,ensure_ascii=False)+'].',basis[:,i])
copy.animation_data_create();copy.animation_data.action=action;copy.animation_data.action_slot=slot
bake_error=0.
for f,frame in enumerate(frames):
 scene.frame_set(int(frame),subframe=frame-int(frame))
 actual=np.asarray([copy.matrix_world@copy.pose.bones[n].matrix for n in names]);bake_error=max(bake_error,float(np.max(np.abs(actual-world[f]))))
assert bake_error<2e-4,f'Independent bake did not preserve evaluated matrices: {bake_error}'
scene.frame_start=start;scene.frame_end=end;scene.frame_set(start);fps=scene.render.fps/scene.render.fps_base
bpy.ops.export_scene.fbx(filepath=str(out/'skeleton.fbx'),use_selection=True,object_types={'ARMATURE'},add_leaf_bones=False,
 bake_anim=True,bake_anim_use_all_bones=True,bake_anim_use_nla_strips=False,bake_anim_use_all_actions=False,bake_anim_simplify_factor=0.,bake_anim_step=.5,
 axis_forward='-Z',axis_up='Y',apply_unit_scale=True,use_armature_deform_only=False)
scene.frame_set(start);assert digest_actions(set(actions))==actions and mesh_digest(mesh)==skin
source_protected_sha=hashlib.sha256(world[:,[index[n] for n in protected]].tobytes()).hexdigest()
bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene;scene.render.fps=round(fps);scene.render.fps_base=round(fps)/fps
bpy.ops.import_scene.fbx(filepath=str(out/'skeleton.fbx'),use_anim=True,anim_offset=0.,automatic_bone_orientation=False)
imported=next(o for o in scene.objects if o.type=='ARMATURE');assert all(n in imported.pose.bones for n in names)
scene.frame_set(start);correction={n:(imported.matrix_world@imported.data.bones[n].matrix_local).inverted()@rest_world[n] for n in names}
position_error=0.;aligned_error=0.;raw_error=0.
for f,frame in enumerate(frames):
 scene.frame_set(int(frame),subframe=frame-int(frame))
 for i,n in enumerate(names):
  actual=imported.matrix_world@imported.pose.bones[n].matrix;aligned=np.asarray(actual@correction[n]);expected=world[f,i]
  position_error=max(position_error,float(np.linalg.norm(np.asarray(actual.translation)-expected[:3,3])))
  aligned_error=max(aligned_error,float(np.max(np.abs(aligned-expected))));raw_error=max(raw_error,float(np.max(np.abs(np.asarray(actual)-expected))))
 if f%300==0:print('FBX_READBACK',f+1,len(frames),flush=True)
(out/'bone-map.json').write_text(json.dumps({'names':names,'source_rest_world':{n:np.asarray(rest_world[n]).tolist() for n in names},'import_rest_world':{n:np.asarray(imported.matrix_world@imported.data.bones[n].matrix_local).tolist() for n in names},'note':'Retarget using world poses, not blind local Action copy when rest bases differ.'},ensure_ascii=False,indent=2),encoding='utf-8')
passed=position_error<.0005 and raw_error<.002
report={'passed':passed,'format':'FBX skeleton only','source':str(source),'source_sha256':source_sha,'source_file_unchanged':hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==source_sha,
 'integer_frames':end-start+1,'sample_frames':len(frames),'sampling_step':.5,'fps':fps,'exported_bones':len(names),'protected_bones':len(protected),'hair_bones':len(hair),
 'original_actions_skin_morphs_unchanged':True,'shape_keys_retained_in_source':shape_count,'source_protected_matrix_sha256':source_protected_sha,
 'independent_bake_matrix_max_error':bake_error,'source_basis_shear_max_error':maximum_shear,'roundtrip_world_head_error_m':position_error,
 'roundtrip_rest_aligned_matrix_max_error':aligned_error,'roundtrip_raw_matrix_max_error':raw_error,'comparison':'direct world heads and full world pose matrices at all integer/half samples; rest-aligned result diagnostic only',
 'weighted_bones_included':len(weighted),'bind_rest_matches_source':aligned_error<.002,'retarget_note':'Skeleton-only: apply evaluated world poses to an existing character. Do not blindly copy local F-curves if bind/rest bases differ.',
 'physics_quality_approval':False,'cloth_vertices_in_fbx':False,'shape_keys_in_fbx':False,'C4D_runtime_verified':False,'seconds':time.perf_counter()-started}
(out/'fbx-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');assert passed,report
print('SKELETON_FBX_PASSED',json.dumps(report),flush=True)
