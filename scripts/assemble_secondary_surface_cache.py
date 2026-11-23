"""Non-destructive visible surface layer; original skin/morph/actions retained."""
from pathlib import Path
import argparse,hashlib,json,sys
import bpy,numpy as np
from mathutils.kdtree import KDTree
from mathutils import Matrix,Quaternion
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from motionbridge.secondary_surface import new_child_output,MDDWriter,rigid_fit
from motionbridge.secondary_surface_config import surface_settings
from solve_secondary_hair_blender import digest_actions,protected_matrices,mesh_digest
from motionbridge.blender_secondary_rigid import create_rigid_objects,evaluated_mesh_records
from motionbridge.secondary_contact_constraints import bind_surface_points,transport_surface_points,FloorContactGraph,rigid_floor_envelope,mesh_edges,garment_edge_deformation,regularized_attachment_fit
parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True)
parser.add_argument('--cloth-dir',type=Path,required=True);parser.add_argument('--tag',default='surface-result')
parser.add_argument('--hair-scene',type=Path)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=args.input_dir.resolve();cloth_root=args.cloth_dir.resolve();out=new_child_output(root,args.tag)
from motionbridge.secondary_attachment_driver import reject_legacy_assembly
reject_legacy_assembly()  # Retired path: never run a one-way follower using old captured metadata.
if not root.is_relative_to(ROOT/'output/secondary_motion') or not cloth_root.is_relative_to(root) or out.exists():raise ValueError('必须使用新表面产物目录')
meta=json.loads((root/'input.json').read_text(encoding='utf-8'));plan=json.loads((root/'model-plan.json').read_text(encoding='utf-8'))
cfg=surface_settings(json.loads((root/'profile.json').read_text(encoding='utf-8')))
parts=np.load(root/'partitions.npz');source=np.load(root/'source-positions.npy',mmap_mode='r')
cloth=np.load(cloth_root/'cloth-positions.npy',mmap_mode='r');count=len(cloth)
ids=np.load(cloth_root/'cloth-index-map.npz')['source_ids'];rest=np.asarray(plan['geometry']['positions_m'])
out.mkdir();bpy.ops.wm.open_mainfile(filepath=str(root/'before.blend'),load_ui=False,use_scripts=False)
bpy.context.preferences.filepaths.save_version=0
before_scene=bpy.context.scene;before_scene.frame_end=meta['frame_start']+count-1
bpy.ops.wm.save_as_mainfile(filepath=str(out/'before.blend'))
if args.hair_scene:bpy.ops.wm.open_mainfile(filepath=str(args.hair_scene),load_ui=False,use_scripts=False)
scene=bpy.context.scene;scene.frame_start=meta['frame_start'];scene.frame_end=scene.frame_start+count-1
mesh=bpy.data.objects[plan['bindings']['mesh']];rig=bpy.data.objects[plan['bindings']['armature']]
tree=KDTree(len(ids))
for i,v in enumerate(rest[ids]):tree.insert(v,i)
tree.balance();rigid=[];flexible=[]
for part in meta['rigid_components']:
 vertices=np.asarray(part['vertex_ids'],int);near=[];mask=[]
 for v in rest[vertices]:
  _,index,distance=tree.find(v);near.append(index);mask.append(distance<.03)
 # Use a real contact vertex as the attachment anchor, not the empty centroid
 # of a ring or curved decorative piece.
 best=min(vertices,key=lambda v:tree.find(rest[v])[2]);neighbors=tree.find_n(rest[best],6)
 patch=np.array([row[1] for row in neighbors],int) if neighbors and neighbors[0][2]<.03 else np.array([],int)
 rigid.append((vertices,np.asarray(near,int),np.asarray(mask,bool),patch))
attachment_records=[];skin=plan['geometry']['skin_weights'];cloth_skins=[skin[int(i)] for i in ids]
cloth_faces=np.load(cloth_root/'cloth-index-map.npz')['welded_faces']
mapping=np.load(cloth_root/'cloth-index-map.npz');representatives=mapping['representatives'];welded_ids=ids[representatives]
bind_limit=float(np.linalg.norm(np.ptp(rest,axis=0)))*cfg['attachment_bind_scale']
for part in meta['excluded_rigid_components']:
 if part['reason']!='flexible_trim_requires_cloth_follow':continue
 vertices=np.asarray(part['vertex_ids'],int)
 binding=bind_surface_points(rest[vertices],rest[welded_ids],cloth_faces,[skin[int(i)] for i in vertices],[skin[int(i)] for i in welded_ids],max_distance=bind_limit)
 flexible.append((vertices,binding))
 attachment_records.append({'vertices':len(vertices),'bound_vertices':int(binding['bound'].sum()),'max_rest_gap_m':float(binding['distance_m'].max(initial=0))})
protected_vertices={v for r in plan['regions'] if r['role'] in ('body','boots') for v in r['vertex_ids']}
garment_vertices=sorted({v for r in plan['regions'] if r['role']=='cloth' for v in r['vertex_ids']}|{int(v) for vertices,_ in flexible for v in vertices})
garment_vertices=np.asarray(garment_vertices,int);all_faces=np.asarray(plan['geometry']['triangles'],int)
flexible_vertices={int(v) for vertices,_ in flexible for v in vertices}
# The original hard-material list also contains excluded flexible metal trim.
# Restore THEIR real triangles to the edge solve; only retained rigid objects
# stay out. Otherwise isolated trim vertices would get floor clamps but no
# length constraints and would be absent from the deformation quality gate.
flexible_faces=np.all(np.isin(all_faces,list(flexible_vertices)),axis=1)
ground_faces=np.all(np.isin(all_faces,garment_vertices),axis=1)&(~np.isin(np.arange(len(all_faces)),meta['hard_faces'])|flexible_faces)
ground_faces=np.searchsorted(garment_vertices,all_faces[ground_faces])
weld_keys=[tuple(sorted((b,round(w,6)) for b,w in skin[int(i)].items())) for i in garment_vertices]
trim_mask=np.isin(garment_vertices,list(flexible_vertices))
body_mask=np.isin(garment_vertices,list(protected_vertices))
# Solve the garment first. Binding trims before projecting the garment makes
# them follow a surface that will no longer be displayed. The trim phase pins
# the final garment, so its own length/floor constraints cannot move it again.
ground_graph=FloorContactGraph(rest[garment_vertices],ground_faces,body_mask|trim_mask,weld_keys)
trim_graph=FloorContactGraph(rest[garment_vertices],ground_faces,body_mask|~trim_mask,weld_keys)
cloth_shape_edges=mesh_edges(ground_faces[~np.any(trim_mask[ground_faces],axis=1)])
trim_transport_max_correction=0.
garment_edges=mesh_edges(ground_faces);deformation_maximum=0.;deformation_failed_frames=0;deformation_edge_samples=0;maximum_shape_passes=0
writer=MDDWriter(out/'surface-final.mdd',count,len(mesh.data.vertices),meta['fps'])
transforms=np.zeros((count,len(rigid),7),dtype=np.float32);last_quaternions=[None]*len(rigid)
floor_shifts=0;maximum_rigidity_error=0.;maximum_attachment_fit=0.;maximum_attachment_rotation=0.;hard_edges=[];contact_vertices=0;contact_max_shift=0.;contact_max_strain=0.
for part,_,_,_ in rigid:
 # A deterministic spanning set of pair distances verifies transform rigidity.
 hard_edges.append(np.column_stack((np.full(len(part)-1,part[0]),part[1:])))
for index in range(count):
 frame=meta['frame_start']+index;scene.frame_set(frame)
 if args.hair_scene:
  ev=mesh.evaluated_get(bpy.context.evaluated_depsgraph_get());data=ev.to_mesh()
  positions=np.empty(len(data.vertices)*3,dtype=np.float32);data.vertices.foreach_get('co',positions);ev.to_mesh_clear();positions=positions.reshape(-1,3)
 else:positions=source[index].copy()
 positions[ids]=cloth[index]
 incoming=positions[garment_vertices].copy();projected=incoming
 for shape_pass in range(cfg['shape_projection_passes']):
  projected,ground=ground_graph.solve(projected,cfg['floor_z_m'],cfg['body_gap_m'],cfg['shape_projection_iterations'],cfg['contact_max_strain'],reference_positions=source[index,garment_vertices])
  contact_vertices+=ground['contact_vertices']
  shape=garment_edge_deformation(source[index,garment_vertices],projected,cloth_shape_edges,cfg['maximum_extra_edge_strain'])
  if shape['passed']:break
 maximum_shape_passes=max(maximum_shape_passes,shape_pass+1)
 positions[garment_vertices]=projected;contact_max_shift=max(contact_max_shift,float(np.linalg.norm(projected-incoming,axis=1).max()));contact_max_strain=max(contact_max_strain,ground['edge_relative_error'])
 for vertices,binding in flexible:
  positions[vertices]=transport_surface_points(binding,source[index,vertices],source[index,welded_ids],positions[welded_ids],cloth_faces)
 trim_target=positions[garment_vertices].copy();projected=trim_target
 for shape_pass in range(cfg['shape_projection_passes']):
  projected,ground=trim_graph.solve(projected,cfg['floor_z_m'],cfg['body_gap_m'],cfg['shape_projection_iterations'],cfg['contact_max_strain'],reference_positions=source[index,garment_vertices])
  contact_vertices+=ground['contact_vertices']
  shape=garment_edge_deformation(source[index,garment_vertices],projected,garment_edges,cfg['maximum_extra_edge_strain'])
  if shape['passed']:break
 assert np.array_equal(projected[~trim_mask],trim_target[~trim_mask]),'Trim solve moved final garment'
 trim_transport_max_correction=max(trim_transport_max_correction,float(np.linalg.norm(projected-trim_target,axis=1).max()))
 positions[garment_vertices]=projected;maximum_shape_passes=max(maximum_shape_passes,shape_pass+1)
 contact_max_shift=max(contact_max_shift,float(np.linalg.norm(projected-incoming,axis=1).max()));contact_max_strain=max(contact_max_strain,ground['edge_relative_error'])
 deformation_maximum=max(deformation_maximum,shape['maximum_additional_relative_edge_error']);deformation_failed_frames+=int(not shape['passed']);deformation_edge_samples+=shape['edges_over_limit']
 # Attach to the FINAL visible garment, after both floor and shape constraints.
 # Using the native cloth cache here left ornaments following a different mesh
 # from the one rendered, especially during crouches and ground contact.
 attachment_cloth=positions[ids];delta=attachment_cloth-source[index,ids]
 for part_index,((vertices,near,mask,patch),edges) in enumerate(zip(rigid,hard_edges)):
  target=source[index,vertices].copy();target[mask]+=delta[near[mask]]
  fitted,rotation,translation=rigid_fit(rest[vertices],source[index,vertices])
  if len(patch)>=3:
   try:
    dr,dt=regularized_attachment_fit(source[index,ids[patch]],attachment_cloth[patch],cfg['attachment_rotation_prior']);fitted=fitted@dr.T+dt;rotation=dr@rotation;translation=dr@translation+dt
    maximum_attachment_rotation=max(maximum_attachment_rotation,float(np.arccos(np.clip((np.trace(dr)-1)/2,-1,1))))
   except ValueError:
    move=delta[patch].mean(0);fitted+=move;translation+=move
  penetration=max(0.,cfg['floor_z_m']+.001-float(fitted[:,2].min()))
  if penetration:fitted[:,2]+=penetration;translation[2]+=penetration;floor_shifts+=1
  maximum_attachment_fit=max(maximum_attachment_fit,float(np.linalg.norm(fitted-target,axis=1).max()))
  expected=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
  actual=np.linalg.norm(fitted[0]-fitted[1:],axis=1)
  maximum_rigidity_error=max(maximum_rigidity_error,float(np.max(np.abs(actual-expected),initial=0)))
  center=rest[vertices].mean(0);q=Matrix(rotation).to_quaternion().normalized()
  if last_quaternions[part_index] is not None and q.dot(last_quaternions[part_index])<0:q.negate()
  last_quaternions[part_index]=q
  transforms[index,part_index,:3]=rotation@center+translation;transforms[index,part_index,3:]=list(q)
 writer.append(positions)
 if index%90==0:print('SURFACE_ASSEMBLE',index+1,count,flush=True)
writer.close()
rigid_envelope_max=0.;rigid_envelope_keys=0
for part_index,(vertices,_,_,_) in enumerate(rigid):
 rotations=np.array([Quaternion(tuple(q)).normalized().to_matrix() for q in transforms[:,part_index,3:]])
 centers,lift=rigid_floor_envelope(rest[vertices]-rest[vertices].mean(0),rotations,transforms[:,part_index,:3],cfg['floor_z_m'],cfg['body_gap_m'])
 transforms[:,part_index,:3]=centers;rigid_envelope_max=max(rigid_envelope_max,float(lift.max(initial=0)));rigid_envelope_keys+=int(np.count_nonzero(lift))
dynamic=set(garment_vertices.tolist())-protected_vertices
mask=mesh.vertex_groups.new(name='MBP_SurfaceDynamic');mask.add(sorted(dynamic),1.,'REPLACE')
mod=mesh.modifiers.new('MB4_DynamicSurfaceCache','MESH_CACHE');mod.filepath=str(out/'surface-final.mdd');mod.frame_start=meta['frame_start']
mod.cache_format='MDD';mod.deform_mode='OVERWRITE';mod.interpolation='LINEAR'
mod.vertex_group=mask.name
objects,rigid_face_ids=create_rigid_objects(mesh,plan,[part[0] for part in rigid],transforms,
                                          list(range(meta['frame_start'],meta['frame_start']+count)))
rigid_names=[obj.name for obj in objects]
np.save(out/'rigid-transforms.npy',transforms)
mesh['motionbridge_surface_state']='experimental_actual_triangle_cloth_and_rigid_attachments'
scene.frame_set(scene.frame_start);bpy.ops.wm.save_as_mainfile(filepath=str(out/'after.blend'))
mod.filepath='//surface-final.mdd';bpy.ops.wm.save_as_mainfile(filepath=str(out/'after.blend'))
bpy.ops.wm.open_mainfile(filepath=str(out/'after.blend'),load_ui=False,use_scripts=False)
mesh=bpy.data.objects[plan['bindings']['mesh']];rig=bpy.data.objects[plan['bindings']['armature']]
frames=sorted(set(range(meta['frame_start'],meta['frame_start']+count))|{f+.5 for f in range(meta['frame_start'],meta['frame_start']+count-1)})
matrices=protected_matrices(rig,meta['protected_names'],frames)
baseline=np.load(root/'protected-before.npy',mmap_mode='r')[:len(frames)]
assert np.array_equal(matrices,baseline),'主骨骼矩阵变化'
assert digest_actions(set(meta['original_actions']))==meta['original_actions'],'源动作曲线变化'
scene=bpy.context.scene;scene.frame_set(meta['frame_start']);assert mesh_digest(mesh)==meta['skin_morph_sha256'],'蒙皮/形态键变化'
assert Path(bpy.path.abspath(mesh.modifiers['MB4_DynamicSurfaceCache'].filepath)).resolve()==out/'surface-final.mdd'
reference=np.memmap(out/'surface-final.mdd',dtype='>f4',mode='r',offset=8+4*count,shape=(count,len(mesh.data.vertices),3))
cache_error=0.;saved_rigid_error=0.
for index in range(count):
 scene.frame_set(meta['frame_start']+index);points,source_ids,_,_=evaluated_mesh_records(mesh)
 cache_error=max(cache_error,float(np.linalg.norm(points-reference[index,source_ids],axis=1).max()))
assert cache_error<1e-5,f'保存后实际表面没有回读到缓存: {cache_error}'
for frame in frames:
 scene.frame_set(int(frame),subframe=frame-int(frame))
 for name in rigid_names:
  obj=bpy.data.objects[name];points,_,_,_=evaluated_mesh_records(obj)
  expected=np.linalg.norm(np.asarray([v.co[:] for v in obj.data.vertices])[1:]-np.asarray(obj.data.vertices[0].co),axis=1)
  actual=np.linalg.norm(points[1:]-points[0],axis=1)
  saved_rigid_error=max(saved_rigid_error,float(np.max(np.abs(actual-expected),initial=0)))
assert saved_rigid_error<2e-6,'刚性饰品在半帧产生变形'
report={'passed':True,'scope':'motion/data preservation and visible cached geometry; not collision quality',
 'frames':count,'protected_bones':len(meta['protected_names']),'protected_matrix_samples':len(frames),
 'protected_matrix_max_error':0,'original_actions_unchanged':True,'skin_and_shape_keys_unchanged':True,
 'rigid_components':len(rigid),'excluded_rigid_components':meta['excluded_rigid_components'],
 'flexible_trim_components_following_cloth':len(flexible),
 'triangle_attachment_binding':{'method':'rest triangle barycentric plus posed patch rotation; skin-aware local selection','components':attachment_records,
  'bound_vertices':sum(row['bound_vertices'] for row in attachment_records),'unbound_vertices':sum(row['vertices']-row['bound_vertices'] for row in attachment_records),'model_scaled_distance_limit_m':bind_limit},
 'garment_ground_constraints':{'vertex_samples':contact_vertices,'maximum_shift_m':contact_max_shift,'maximum_edge_relative_error':contact_max_strain,
  'plane_interpolation_guarantee':'dynamic vertex cache is linear; all nonprotected garment key positions are above the plane'},
 'garment_deformation_gate':{'passed':deformation_failed_frames==0,'frames_over_budget':deformation_failed_frames,'edge_samples_over_budget':deformation_edge_samples,
  'actual_flexible_trim_triangles':int(flexible_faces.sum()),'actual_garment_edges':len(garment_edges),
  'maximum_additional_relative_edge_error':deformation_maximum,'relative_limit':cfg['maximum_extra_edge_strain'],'absolute_tolerance_m':5e-6,'maximum_projection_passes_used':maximum_shape_passes,
  'scope':'all integer-frame nonrigid garment/trim triangle edges relative to skinned input; not continuous-time shape certification'},
 'rigid_continuous_floor_envelope':{'maximum_extra_lift_m':rigid_envelope_max,'adjusted_keys':rigid_envelope_keys,'method':'linear center plus rotation arc sagitta conservative plane bound'},
 'rigid_shape_distance_max_error_m':maximum_rigidity_error,'rigid_attachment_fit_max_error_m':maximum_attachment_fit,
 'rigid_attachment_mode':'kinematic rigid shape matching, follows local clothing motion; not free 6-DOF rigid-body dynamics',
 'rigid_attachment_driver':'final constrained visible garment, including floor and edge projection',
 'rigid_attachment_orientation':{'method':'scale-normalized source-pose regularized SO3 fit','rotation_prior':cfg['attachment_rotation_prior'],'maximum_extra_rotation_rad':maximum_attachment_rotation},
 'flexible_attachment_driver':{'surface':'final constrained visible garment','garment_fixed_during_trim_constraints':True,'maximum_trim_shape_floor_correction_m':trim_transport_max_correction},
 'rigid_floor_projection_count':floor_shifts,'cache':str(out/'surface-final.mdd'),
 'cache_interpolation':'linear vertex interpolation; half-frame triangle penetration still requires audit',
 'saved_geometry_max_error_m':cache_error,'cache_mask_excludes_all_protected_body_and_boot_vertices':True,
 'separate_rigid_objects':len(rigid_names),'rigid_shape_error_including_half_frames_m':saved_rigid_error,
 'rigid_render_face_ids':rigid_face_ids.tolist(),
 'source':meta['source'],'source_sha256':meta['source_sha256']}
(out/'surface-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('SURFACE_CACHE_PASSED',json.dumps({k:report[k] for k in ['passed','frames','protected_matrix_max_error','saved_geometry_max_error_m','separate_rigid_objects','rigid_shape_error_including_half_frames_m']},ensure_ascii=False),flush=True)
