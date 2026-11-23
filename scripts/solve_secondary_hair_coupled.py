"""Coupled hair with real body triangle queries, native cross joints and floor."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import bpy,numpy as np
from mathutils import Matrix,Euler,Vector
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from motionbridge.secondary_hair import select_hair
from motionbridge.secondary_surface import new_child_output
from motionbridge.secondary_coupling import CoupledHairSolver
from motionbridge.secondary_surface_config import surface_settings
from solve_secondary_hair_blender import make_nodes,capture_drivers,interpolate_drivers,protected_matrices,digest_actions,mesh_digest

parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True)
parser.add_argument('--take',type=int);parser.add_argument('--tag',default='hair-coupled')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=args.input_dir.resolve();out=new_child_output(root,args.tag)
if not root.is_relative_to(ROOT/'output/secondary_motion') or out.exists():raise ValueError('发束耦合必须输出到新的独立目录')
meta=json.loads((root/'input.json').read_text(encoding='utf-8'));plan=json.loads((root/'model-plan.json').read_text(encoding='utf-8'))
profile=json.loads((root/'profile.json').read_text(encoding='utf-8'));settings=profile['hair_dynamics'];cfg=surface_settings(profile)
count=min(args.take or meta['frames'],meta['frames']);frames=list(range(meta['frame_start'],meta['frame_start']+count))
bpy.ops.wm.open_mainfile(filepath=str(root/'before.blend'),load_ui=False,use_scripts=False)
rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
ids=select_hair(plan,settings);nodes,unused=make_nodes(plan,ids,rig,settings);lookup={bone:i for i,bone in enumerate(ids)}
cross=[]
for joint in plan['joints']:
 if joint['id'] not in unused:continue
 a,b=[plan['rigid_bodies'][i]['bone_id'] for i in joint['rigid_ids']]
 if a not in lookup or b not in lookup:continue
 p=joint['blender'];rotation=Euler(p['rotation_rad'],p['rotation_order']).to_matrix()
 ra=rig.data.bones[plan['bones'][a]['blender_name']].matrix_local
 rb=rig.data.bones[plan['bones'][b]['blender_name']].matrix_local
 cross.append({'id':joint['id'],'name':joint['name'],'a':lookup[a],'b':lookup[b],
     'anchor_a':list(ra.inverted()@Vector(p['position_m'])),'anchor_b':list(rb.inverted()@Vector(p['position_m'])),
     'frame_a':[list(row) for row in ra.to_3x3().inverted()@rotation],
     'frame_b':[list(row) for row in rb.to_3x3().inverted()@rotation],
     'linear_min':p['linear_min_m'],'linear_max':p['linear_max_m'],
     'angular_min':p['angular_min_rad'],'angular_max':p['angular_max_rad']})
assert len(cross)==len(unused),'仍有未适配的原生跨束关节'
external_names=sorted({n['external_parent'] for n in nodes if n['parent']<0})
drivers={name:[] for name in external_names}
for frame in frames:
 bpy.context.scene.frame_set(frame)
 for name in external_names:
  m=rig.matrix_world@rig.pose.bones[name].matrix;r=m.to_quaternion().normalized().to_matrix().to_4x4();r.translation=m.translation
  drivers[name].append(np.asarray(r))
drivers={name:np.asarray(values) for name,values in drivers.items()}
rest_heads=np.asarray([rig.data.bones[n['name']].head_local for n in nodes])
rest_tails=np.asarray([rig.data.bones[n['name']].tail_local for n in nodes])
solver=CoupledHairSolver(nodes,settings,cross,rest_heads,rest_tails,floor_z=cfg['floor_z_m'])
parts=np.load(root/'partitions.npz');body_ids=parts['body_ids'];body_faces=parts['body_faces']
source=np.load(root/'source-positions.npy',mmap_mode='r');triangles=body_faces.tolist()
class BodyTriangles:
 def __init__(self,positions):
  self.tree=BVHTree.FromPolygons(positions.tolist(),triangles,all_triangles=True)
 def __call__(self,points,radius):
  result=[]
  for point in points:
   near,normal,face,distance=self.tree.find_nearest(Vector(point),.03)
   if near is None:continue
   n=np.asarray(normal);difference=point-np.asarray(near);signed=float(np.dot(difference,n))
   supported=np.linalg.norm(difference-signed*n)<1e-5
   if supported and signed<radius+.001:
    result.append((point,n,min(.03,radius+.001-signed)))
  return result
out.mkdir();np.save(out/'body-faces.npy',body_faces)
(out/'cross-joints.json').write_text(json.dumps(cross,ensure_ascii=False,indent=2),encoding='utf-8')
dt=1/meta['fps']/settings['substeps'];warm_steps=round(5/dt);fixed=interpolate_drivers(drivers,0,1)
first_body=BodyTriangles(source[0,body_ids]);started=time.perf_counter()
for step in range(warm_steps):
 solver.step(fixed,first_body,dt)
 if step%90==0:print('COUPLED_WARMUP',step,warm_steps,round(time.perf_counter()-started,2),flush=True)
speed=np.linalg.norm(solver.base.velocities,axis=1)
warm={'steps':warm_steps,'max_tip_speed_m_s':float(speed.max()),'p95_tip_speed_m_s':float(np.percentile(speed,95)),
      'residuals':solver.residuals(fixed),'worst_bone':nodes[int(np.argmax(speed))]['name']}
(out/'warmup.json').write_text(json.dumps(warm,ensure_ascii=False,indent=2),encoding='utf-8')
if speed.max()>.04:raise ValueError('耦合发束预热未收敛，不作为成功结果发布')
rotations=[];heads=[];tails=[];angles=[];residuals=[]
for index in range(count):
 previous=max(0,index-1)
 for sub in range(settings['substeps']):
  fraction=(sub+1)/settings['substeps'];drive=interpolate_drivers(drivers,index,fraction)
  positions=source[previous,body_ids]*(1-fraction)+source[index,body_ids]*fraction
  h,r,t,a=solver.step(drive,BodyTriangles(positions),dt)
 heads.append(h);rotations.append(r);tails.append(t);angles.append(a);residuals.append(solver.residuals(drive))
 if index%30==0:
  progress={'frames':index+1,'total':count,'seconds':time.perf_counter()-started}
  (out/'progress.json').write_text(json.dumps(progress),encoding='utf-8');print('COUPLED_HAIR',json.dumps(progress),flush=True)
arrays=dict(heads=np.asarray(heads),rotations=np.asarray(rotations),tails=np.asarray(tails),angles=np.asarray(angles))
np.savez_compressed(out/'hair-coupled.npz',**arrays)
# Add a new result layer; retain every incoming action/target curve unchanged.
collection=bpy.data.collections.new('MB4_CoupledHairTargets');bpy.context.scene.collection.children.link(collection)
for i,node in enumerate(nodes):
 target=bpy.data.objects.new(f'MB4_HairTarget_{node["bone_id"]}',None);collection.objects.link(target)
 target.hide_render=True;target.rotation_mode='QUATERNION';target.empty_display_size=.01
 action=bpy.data.actions.new(f'MB4_HairRotation_{node["bone_id"]}');slot=action.slots.new(id_type='OBJECT',name=target.name)
 layer=action.layers.new('Coupled model hair');strip=layer.strips.new(type='KEYFRAME');bag=strip.channelbags.new(slot)
 quaternions=[];last=None
 for matrix in arrays['rotations'][:,i]:
  q=Matrix(matrix).to_quaternion().normalized()
  if last is not None and q.dot(last)<0:q.negate()
  quaternions.append(list(q));last=q
 for c in range(4):
  curve=bag.fcurves.new(data_path='rotation_quaternion',index=c);curve.keyframe_points.add(count)
  curve.keyframe_points.foreach_set('co',np.column_stack((frames,np.asarray(quaternions)[:,c])).ravel())
  for key in curve.keyframe_points:key.interpolation='LINEAR'
  curve.update()
 target.animation_data_create();target.animation_data.action=action;target.animation_data.action_slot=slot
 constraint=rig.pose.bones[node['name']].constraints.new('COPY_ROTATION');constraint.name='MB4_CoupledHairDynamics'
 constraint.target=target;constraint.owner_space='WORLD';constraint.target_space='WORLD';constraint.mix_mode='REPLACE'
scene=bpy.context.scene;scene.frame_end=frames[-1];scene.frame_set(frames[0])
bpy.context.preferences.filepaths.save_version=0;bpy.ops.wm.save_as_mainfile(filepath=str(out/'hair-after.blend'))
bpy.ops.wm.open_mainfile(filepath=str(out/'hair-after.blend'),load_ui=False,use_scripts=False)
rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
audit_frames=sorted(set(frames+[f+.5 for f in frames[:-1]]));matrices=protected_matrices(rig,meta['protected_names'],audit_frames)
baseline=np.load(root/'protected-before.npy',mmap_mode='r')[:len(audit_frames)]
assert np.array_equal(matrices,baseline),'耦合发束改变了主骨矩阵'
assert digest_actions(set(meta['original_actions']))==meta['original_actions'],'输入动作曲线变化'
bpy.context.scene.frame_set(frames[0]);assert mesh_digest(mesh)==meta['skin_morph_sha256'],'蒙皮/形态键变化'
report={'passed':True,'frames':count,'source_actions_unchanged':True,'protected_bones':len(meta['protected_names']),
 'matrix_samples':len(audit_frames),'matrix_max_error':0,'skin_and_morphs_unchanged':True,'hair_bones':len(nodes),
 'cross_joints_integrated':len(cross),'floor_z_m':cfg['floor_z_m'],'self_collision_candidate_pairs':len(solver.pairs),
 'authored_overlap_exclusions':solver.authored_overlap_exclusions,'self_contact_projections':solver.self_contacts,
 'triangle_contact_projections':solver.triangle_contacts,'floor_contact_projections':solver.floor_contacts,
 'cross_joint_projections':solver.cross_projections,'merit_backoffs':solver.merit_backoffs,
 'weld_locked_child_bones':[nodes[i]['name'] for i in solver.weld_locked_bones],'actual_body_triangles':len(body_faces),'warmup':warm,
 'residual_maxima':{key:max(row[key] for row in residuals) for key in residuals[0]},
 'seconds':time.perf_counter()-started,'limitations':['发束自碰撞为骨段胶囊，非真实发片三角面自碰撞。',
 '原生跨束位移和角范围进入迭代，受自身角限位/固定根和速度上限影响可能有非零残差。',
 '身体查询使用每子步插值后的真实三角面，但不是任意运动的完整连续碰撞认证。']}
(out/'hair-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('COUPLED_HAIR_PASSED',json.dumps(report,ensure_ascii=False),flush=True)
