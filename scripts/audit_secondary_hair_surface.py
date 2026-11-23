"""Independent saved mesh samples and temporal diagnostics, not a collision certificate."""
from pathlib import Path
import argparse,hashlib,json,math,sys
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from solve_secondary_hair_blender import body_specs,colliders_at,protected_matrices,digest_actions,mesh_digest
from motionbridge.secondary_hair import proxy_contacts
parser=argparse.ArgumentParser();parser.add_argument('--pair-dir',type=Path,required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=args.pair_dir.resolve()
if not root.is_relative_to(ROOT/'output/secondary_motion'):raise ValueError('非法输出路径')
destination=root/'surface-audit.json'
if destination.exists():raise ValueError('已有表面审核，拒绝覆盖')
plan=json.loads((root/'model-plan.json').read_text(encoding='utf-8'))
profile=json.loads((root/'profile.json').read_text(encoding='utf-8'))
physics=json.loads((root/'physics-validation.json').read_text(encoding='utf-8'))
nodes=json.loads((root/'hair-nodes.json').read_text(encoding='utf-8'))
hair_region=next(r for r in plan['regions'] if r['role']=='hair');hair=np.array(hair_region['vertex_ids'])
active=hair[np.array(hair_region['pin_weights'])<.999999]
nonhair=np.array(sorted(set(range(len(plan['geometry']['positions_m'])))-set(hair)))
source_faces=np.array([plan['geometry']['triangles'][f] for f in hair_region['face_ids']])
edges=np.unique(np.sort(np.vstack([source_faces[:,[0,1]],source_faces[:,[1,2]],source_faces[:,[2,0]]]),axis=1),axis=0)
frames=None;protection={}
before=[];metrics={};max_nonhair=0.;displacements=[];edge_relative=[];edge_absolute=[];all_lengths=[];worst_edge=None
for variant in ('before','after'):
 bpy.ops.wm.open_mainfile(filepath=str(root/f'{variant}.blend'),load_ui=False,use_scripts=False)
 rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
 scene=bpy.context.scene
 if frames is None:frames=sorted(set(range(scene.frame_start,scene.frame_end+1,15))|{scene.frame_end})
 audit_frames=np.arange(scene.frame_start,scene.frame_end+.01,.5).tolist()
 names=[plan['bones'][i]['blender_name'] for i in plan['protected_bone_ids']]
 matrices=protected_matrices(rig,names,audit_frames)
 scene.frame_set(scene.frame_start)
 protection[variant]={'matrix_sha256':hashlib.sha256(matrices.tobytes()).hexdigest(),
                      'actions':digest_actions(set(physics['source_actions'])),
                      'skin_morph_sha256':mesh_digest(mesh),'frames':len(audit_frames),'bones':len(names)}
 specs=body_specs(plan,rig,profile['hair_dynamics'],[n['bone_id'] for n in nodes])
 contact_count=0;worst=0.;sum_depth=0.;contact_frames=0
 for index,frame in enumerate(frames):
  bpy.context.scene.frame_set(frame);deps=bpy.context.evaluated_depsgraph_get();evaluated=mesh.evaluated_get(deps)
  data=evaluated.to_mesh();positions=np.empty(len(data.vertices)*3,dtype=np.float32);data.vertices.foreach_get('co',positions)
  positions=positions.reshape(-1,3);world=np.asarray(mesh.matrix_world);positions=positions@world[:3,:3].T+world[:3,3]
  evaluated.to_mesh_clear()
  drivers={s['bone']:np.asarray(rig.matrix_world@rig.pose.bones[s['bone']].matrix) for s in specs}
  body=colliders_at(specs,drivers);distance,_=proxy_contacts(positions[active],**body)
  depth=np.maximum(-distance.min(axis=1),0)
  contact_count+=int(np.sum(depth>.001));contact_frames+=int(np.any(depth>.001));worst=max(worst,float(depth.max()));sum_depth+=float(depth.sum())
  lengths=np.linalg.norm(positions[edges[:,0]]-positions[edges[:,1]],axis=1)
  if variant=='before':before.append(positions.copy());all_lengths.append(lengths)
  else:
   max_nonhair=max(max_nonhair,float(np.max(np.linalg.norm(positions[nonhair]-before[index][nonhair],axis=1))))
   displacements.extend(np.linalg.norm(positions[hair]-before[index][hair],axis=1).tolist())
   valid=all_lengths[index]>.0001
   edge_relative.extend((np.abs(lengths[valid]-all_lengths[index][valid])/all_lengths[index][valid]).tolist())
   differences=np.abs(lengths-all_lengths[index]);edge_absolute.extend(differences.tolist())
   largest=int(np.argmax(differences))
   if worst_edge is None or differences[largest]>worst_edge['absolute_change_m']:
    worst_edge={'frame':frame,'vertices':edges[largest].tolist(),'before_length_m':float(all_lengths[index][largest]),
                'after_length_m':float(lengths[largest]),'absolute_change_m':float(differences[largest])}
 metrics[variant]={'active_hair_vertex_proxy_contacts_deeper_than_1mm':contact_count,'sampled_frames_with_contact':contact_frames,
                   'maximum_proxy_depth_m':worst,'summed_proxy_depth_m':sum_depth,
                   'sampled_frames':len(frames),'active_hair_vertices':len(active)}
sim=np.load(root/'hair-simulation.npz');angles=sim['angles'];rot=sim['rotations'];fps=bpy.context.scene.render.fps/bpy.context.scene.render.fps_base
speed=np.linalg.norm(np.diff(angles,axis=0),axis=2)*fps
adj=np.einsum('fnji,fnjk->fnik',rot[:-1],rot[1:]);trace=np.trace(adj,axis1=2,axis2=3)
world_angles=np.arccos(np.clip((trace-1)/2,-1,1));max_where=np.unravel_index(np.argmax(world_angles),world_angles.shape)
report={'passed':max_nonhair<2e-6 and protection['before']==protection['after'],
        'independent_saved_protection':protection,
        'method':'Actual evaluated mesh, every 15 frames plus final; analytic authored body proxies',
        'sampled_frames':frames,'nonhair_vertex_count':len(nonhair),'nonhair_max_world_change_m':max_nonhair,
        'hair_vertex_change_p95_m':float(np.percentile(displacements,95)), 'hair_vertex_change_max_m':float(max(displacements)),
        'hair_edge_relative_length_change_p95':float(np.percentile(edge_relative,95)),
        'hair_edge_relative_length_change_max':float(max(edge_relative)),
        'hair_edge_absolute_length_change_p95_m':float(np.percentile(edge_absolute,95)),
        'hair_edge_absolute_length_change_max_m':float(max(edge_absolute)),'worst_absolute_edge_change':worst_edge,
        'proxy_surface_before':metrics['before'],'proxy_surface_after':metrics['after'],
        'joint_euler_speed_p95_rad_s':float(np.percentile(speed,95)),'joint_euler_speed_max_rad_s':float(speed.max()),
        'world_hair_adjacent_rotation_p95_deg':float(np.degrees(np.percentile(world_angles,95))),
        'world_hair_adjacent_rotation_max_deg':float(np.degrees(world_angles.max())),
        'world_hair_adjacent_rotation_max_frame':int(max_where[0]+2),'world_hair_adjacent_rotation_max_bone':nodes[max_where[1]]['name'],
        'limitations':['采样表面审核，非全帧真实身体三角面连续碰撞认证。',
                       '代理接触包含源模型已有重叠；没有把原生重叠扣除后当成零穿插。',
                       '骨段长度保持不等于混合蒙皮网格不拉伸；边长变化按实测保留。']}
destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('HAIR_SURFACE_AUDIT',json.dumps(report,ensure_ascii=False),flush=True)
