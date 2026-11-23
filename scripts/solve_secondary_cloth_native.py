"""Native Blender cloth on actual garment triangles and moving body triangles.

Scratch simulation timeline prepends a held pose. Source actions are neither
shifted nor modified. Sampled output is mapped back to original frame numbers.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from motionbridge.secondary_surface import new_child_output,MDDWriter,weld_driver_surface,anchor_unattached_components
from motionbridge.secondary_surface_config import surface_settings
from motionbridge.secondary_contact_constraints import contact_pin_release,FloorContactGraph,prewarm_convergence

parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True)
parser.add_argument('--take',type=int);parser.add_argument('--tag',default='cloth-native')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=args.input_dir.resolve();out=new_child_output(root,args.tag)
if not root.is_relative_to(ROOT/'output/secondary_motion') or out.exists() or not args.tag.replace('-','').isalnum():raise ValueError('非法或已存在的衣物求解目录')
meta=json.loads((root/'input.json').read_text(encoding='utf-8'));plan=json.loads((root/'model-plan.json').read_text(encoding='utf-8'))
parts=np.load(root/'partitions.npz');source=np.load(root/'source-positions.npy',mmap_mode='r')
profile=json.loads((root/'profile.json').read_text(encoding='utf-8'));cfg=surface_settings(profile)
count=min(args.take or meta['frames'],meta['frames']);fps=meta['fps'];warm=max(2,int(round(cfg['warmup_seconds']*fps)))
assert count>0
ids=parts['cloth_ids'];faces=parts['cloth_faces'];pins=parts['cloth_pins']
rest=np.asarray(plan['geometry']['positions_m'])[ids]
skin_keys=[tuple(sorted((b,round(w,6)) for b,w in plan['geometry']['skin_weights'][int(i)].items())) for i in ids]
representatives,welded_faces,welded_pins,inverse=weld_driver_surface(rest,faces,pins,skin_keys)
welded_pins,anchor_records=anchor_unattached_components(rest[representatives],welded_faces,welded_pins,band=cfg['anchor_band_m'])
effective_pins=welded_pins[inverse]
# Small, explicit pose guide keeps freely skinned ribbons attached to model
# styling while allowing the native cloth force/contact solve to move them.
welded_pins=np.maximum(welded_pins,cfg['minimum_pose_guide'])
protected_vertices={v for r in plan['regions'] if r['role'] in ('body','boots') for v in r['vertex_ids']}
protected=np.isin(ids[representatives],list(protected_vertices))
unsafe=np.zeros(len(representatives),bool)
for index in range(count):unsafe|=source[index,ids[representatives],2]<cfg['floor_z_m']+cfg['body_gap_m']
original_solver_pins=welded_pins.copy()
welded_pins,contact_depth=contact_pin_release(welded_pins,welded_faces,unsafe,protected,cfg['contact_pin_collar'])
# The guide itself has already been made floor-feasible. Retain its configured
# soft attraction; remove incompatible HARD pins rather than free whole panels.
welded_pins=np.maximum(welded_pins,cfg['minimum_pose_guide'])
effective_pins=welded_pins[inverse]
contact_graph=FloorContactGraph(rest[representatives],welded_faces,protected)
initial_driver,initial_contact=contact_graph.solve(np.asarray(source[0,ids[representatives]]),cfg['floor_z_m'],cfg['body_gap_m'],cfg['contact_edge_iterations'],cfg['contact_max_strain'])
for index in range(count):
 delta=source[index,ids]-source[index,ids[representatives]][inverse]
 if np.max(np.abs(delta))>2e-6:raise ValueError('UV缝顶点在动画中分离，不允许错误焊接')
cloth_ids=ids[representatives];body_ids=parts['body_ids'];body_faces=parts['body_faces'];out.mkdir()
for label,selected in [('cloth-driver',cloth_ids),('body-driver',body_ids)]:
 writer=MDDWriter(out/f'{label}.mdd',warm+count,len(selected),fps)
 for _ in range(warm):writer.append(initial_driver if label=='cloth-driver' else source[0,selected])
 for index in range(count):
  target=np.asarray(source[index,selected]).copy()
  if label=='cloth-driver':target,_=contact_graph.solve(target,cfg['floor_z_m'],cfg['body_gap_m'],cfg['contact_edge_iterations'],cfg['contact_max_strain'])
  writer.append(target)
 writer.close()
bpy.ops.wm.open_mainfile(filepath=str(root/'before.blend'),load_ui=False,use_scripts=False)
bpy.context.preferences.filepaths.save_version=0
# All point cache paths now belong to this NEW output directory.
bpy.ops.wm.save_as_mainfile(filepath=str(out/'solver-setup.blend'))
scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=warm+count;scene.render.fps=round(fps);scene.render.fps_base=round(fps)/fps
scene.frame_set(1)
colliders=bpy.data.collections.new('MB4_ActualTriangleColliders');scene.collection.children.link(colliders)
sim=bpy.data.collections.new('MB4_SurfaceSolver');scene.collection.children.link(sim)
def mesh_object(name,positions,triangles,collection):
 data=bpy.data.meshes.new(name);data.from_pydata(positions.tolist(),[],triangles.tolist());data.update()
 obj=bpy.data.objects.new(name,data);collection.objects.link(obj);obj.hide_render=True;return obj
def cache(obj,path):
 mod=obj.modifiers.new('MB4_CapturedActualPoses','MESH_CACHE');mod.cache_format='MDD';mod.filepath=str(path)
 mod.frame_start=1;mod.time_mode='FRAME';mod.play_mode='SCENE';mod.deform_mode='OVERWRITE'
 return mod
body=mesh_object('MB4_ActualBodyTriangles',source[0,body_ids],body_faces,colliders)
cache(body,out/'body-driver.mdd');body.modifiers.new('MB4_ActualBodyCollision','COLLISION')
body.collision.thickness_outer=cfg['body_gap_m'];body.collision.thickness_inner=cfg['body_gap_m']
floor=mesh_object('MB4_Floor',np.array([[-20,-20,0],[20,-20,0],[20,20,0],[-20,20,0]],float),np.array([[0,1,2],[0,2,3]]),colliders)
floor.location.z=cfg['floor_z_m']
floor.modifiers.new('MB4_FloorCollision','COLLISION');floor.collision.thickness_outer=cfg['body_gap_m'];floor.collision.thickness_inner=cfg['body_gap_m']
cloth_obj=mesh_object('MB4_ActualGarmentTriangles',initial_driver,welded_faces,sim)
cache(cloth_obj,out/'cloth-driver.mdd')
group=cloth_obj.vertex_groups.new(name='MB4_Pin')
for index,weight in enumerate(welded_pins):
 if weight>0:group.add([index],float(weight),'REPLACE')
cloth=cloth_obj.modifiers.new('MB4_ClothDynamics','CLOTH');settings=cloth.settings
settings.quality=cfg['cloth_quality'];settings.mass=cfg['mass_kg'];settings.tension_stiffness=cfg['tension_stiffness'];settings.compression_stiffness=cfg['compression_stiffness']
settings.shear_stiffness=cfg['shear_stiffness'];settings.bending_stiffness=cfg['bending_stiffness'];settings.air_damping=cfg['air_damping']
settings.tension_damping=cfg['material_damping'];settings.compression_damping=cfg['material_damping'];settings.shear_damping=cfg['material_damping']
settings.vertex_group_mass=group.name;settings.pin_stiffness=cfg['pin_stiffness'];settings.use_dynamic_mesh=True
collision=cloth.collision_settings;collision.use_collision=True;collision.collision_quality=cfg['collision_quality']
collision.distance_min=cfg['body_gap_m'];collision.use_self_collision=True;collision.self_distance_min=cfg['self_gap_m']
collision.self_impulse_clamp=cfg['self_impulse_clamp'];collision.impulse_clamp=cfg['impulse_clamp'];collision.collection=colliders
cloth.point_cache.frame_start=1;cloth.point_cache.frame_end=warm+count;cloth.point_cache.name='MB4_Cloth'
cloth.point_cache.use_disk_cache=True;cloth.point_cache.filepath=str(out/'point-cache')
bpy.ops.wm.save_as_mainfile(filepath=str(out/'solver-setup.blend'))
output=np.lib.format.open_memmap(out/'cloth-positions.npy',mode='w+',dtype=np.float32,shape=(count,len(ids),3))
last=None;warm_speed=None;warm_speeds=[];max_shift=0.;pin_reprojection=0.;start=time.perf_counter();ground_contacts=0;ground_max_shift=0.;ground_max_strain=0.
for simulation_frame in range(1,warm+count+1):
 scene.frame_set(simulation_frame);evaluated=cloth_obj.evaluated_get(bpy.context.evaluated_depsgraph_get());data=evaluated.to_mesh()
 points=np.empty(len(data.vertices)*3,dtype=np.float32);data.vertices.foreach_get('co',points);points=points.reshape(-1,3)
 evaluated.to_mesh_clear()
 if not np.isfinite(points).all():raise ValueError('原生衣物求解产生非有限值')
 if simulation_frame==warm and last is not None:
  warm_speed=np.linalg.norm(points-last,axis=1)*fps
 if max(1,warm-int(round(fps)))<simulation_frame<=warm and last is not None:
  warm_speeds.append(np.linalg.norm(points-last,axis=1)*fps)
 if simulation_frame>warm:
  index=simulation_frame-warm-1;expanded=points[inverse].copy()
  fixed=effective_pins>=.999999
  pin_reprojection=max(pin_reprojection,float(np.max(np.linalg.norm(expanded[fixed]-source[index,ids[fixed]],axis=1),initial=0)))
  expanded[fixed]=source[index,ids[fixed]]
  welded,contact=contact_graph.solve(expanded[representatives],cfg['floor_z_m'],cfg['body_gap_m'],cfg['contact_edge_iterations'],cfg['contact_max_strain'])
  expanded=welded[inverse]
  ground_contacts+=contact['contact_vertices'];ground_max_shift=max(ground_max_shift,contact['maximum_shift_m']);ground_max_strain=max(ground_max_strain,contact['edge_relative_error'])
  max_shift=max(max_shift,float(np.linalg.norm(expanded-source[index,ids],axis=1).max()))
  if max_shift>cfg['max_displacement_m']:raise ValueError('衣物位移超过2米，判为求解爆炸，不发布')
  output[index]=expanded
 last=points.copy()
 if simulation_frame%30==0:
  progress={'simulation_frame':simulation_frame,'total':warm+count,'captured':max(0,simulation_frame-warm),'seconds':time.perf_counter()-start}
  (out/'progress.json').write_text(json.dumps(progress),encoding='utf-8');print('CLOTH_PROGRESS',json.dumps(progress),flush=True)
output.flush()
result={'passed':True,'simulation_performed':True,'method':'Blender native Cloth on real garment triangles, actual posed body triangles, self collision and real floor mesh',
 'frames':count,'fps':fps,'prewarm_seconds':warm/fps,'internal_simulation_frames':warm+count,
 'original_action_keyframes_shifted':False,'source_raw_mesh_or_skin_modified':False,
 'render_cloth_source_vertices':len(ids),'simulation_welded_vertices':len(cloth_ids),'actual_cloth_triangles':len(welded_faces),
 'actual_body_triangles':len(body_faces),'floor_triangles':2,'full_pin_exact_reprojection_max_m':pin_reprojection,
 'model_rest_anchor_components':anchor_records,'minimum_pose_guide':cfg['minimum_pose_guide'],'settings':cfg,
 'spring_rest_source':'contact-corrected animated garment driver; preserves existing pose-following rest policy',
 'native_cache_info':cloth.point_cache.info,'native_gravity':list(scene.gravity),'native_use_gravity':scene.use_gravity,
 'contact_priority':{'unsafe_pose_vertices':int(unsafe.sum()),'relaxed_pin_vertices':int(np.sum(welded_pins<original_solver_pins)),
  'protected_pins_changed':bool(np.any(welded_pins[protected]!=original_solver_pins[protected])),
  'topological_collar_rings':cfg['contact_pin_collar'],'integer_contact_vertex_samples':ground_contacts,
  'maximum_final_contact_projection_m':ground_max_shift,'maximum_contact_edge_relative_error':ground_max_strain,'initial_projection':initial_contact},
 'max_cloth_shift_m':max_shift,'prewarm_vertex_speed_p95_m_s':float(np.percentile(warm_speed,95)),
 'prewarm_convergence':prewarm_convergence(warm_speeds,cfg['prewarm_p95_limit_m_s'],cfg['prewarm_max_limit_m_s']),
 'prewarm_vertex_speed_max_m_s':float(warm_speed.max()),'seconds':time.perf_counter()-start,
 'limitations':['三角面接触已进入原生求解，不是完整连续碰撞/零穿透认证。','原生重叠、自碰撞与强制运动学固定区可能留下残余，必须另做保存表面审核。',
                'UV接缝仅在静态位置/蒙皮相同且全片驱动重合核验后合并；保留原有真实三角形，未低模替代。']}
np.savez(out/'cloth-index-map.npz',source_ids=ids,representatives=representatives,inverse=inverse,welded_faces=welded_faces)
(out/'cloth-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('NATIVE_RIN_CLOTH_COMPUTED',json.dumps(result,ensure_ascii=False),flush=True)
