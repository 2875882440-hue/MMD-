"""Saved actual triangles, floor and protected surface audit. No proxy scores."""
from pathlib import Path
import argparse,json,sys,hashlib
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from motionbridge.blender_secondary_rigid import evaluated_mesh_records
from motionbridge.blender_triangle_contacts import TriangleSurface
from solve_secondary_hair_blender import protected_matrices,digest_actions,mesh_digest
parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--pair-dir',type=Path,required=True)
parser.add_argument('--stride',type=int,default=1,help='1 means every integer and half frame')
parser.add_argument('--reuse-before',type=Path,help='Reuse validated immutable before audit only for same captured source')
parser.add_argument('--report-path',type=Path,help='New diagnostic report under validation; never overwrite original audit')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=args.input_dir.resolve();pair=args.pair_dir.resolve()
report_path=args.report_path.resolve() if args.report_path else pair/'surface-audit.json'
if args.report_path and not report_path.is_relative_to(ROOT/'validation'):raise ValueError('诊断报告只写validation')
if not pair.is_relative_to(ROOT/'output/secondary_motion') or report_path.exists():raise ValueError('审核只写新报告')
meta=json.loads((root/'input.json').read_text(encoding='utf-8'));plan=json.loads((root/'model-plan.json').read_text(encoding='utf-8'))
profile=json.loads((root/'profile.json').read_text(encoding='utf-8'));floor=profile.get('surface_dynamics',{}).get('floor_z_m',0.)
body_faces=set(meta['body_source_face_ids']);cloth_faces={f for r in plan['regions'] if r['role']=='cloth' for f in r['face_ids']}
hair_faces={f for r in plan['regions'] if r['role']=='hair' for f in r['face_ids']}
accessory_faces={f for r in plan['regions'] if r['role']=='accessory' for f in r['face_ids']}
body_vertices=np.array(sorted({v for r in plan['regions'] if r['role'] in ('body','boots') for v in r['vertex_ids']}),int)
results={};protection={};before_body=None;max_body_error=0.;frame_list=None
if args.reuse_before:
 baseline_path=args.reuse_before.resolve()
 if not baseline_path.is_relative_to(root):raise ValueError('只允许同一捕获源的基线')
 previous=json.loads(baseline_path.read_text(encoding='utf-8'))
 if previous.get('attachment_audit_required') is not True:raise ValueError('旧基线缺少饰品碰撞审核，必须重新采样before')
 previous_surface=json.loads((baseline_path.parent/'surface-validation.json').read_text(encoding='utf-8'))
 assert previous['protection_passed'] and previous_surface['source_sha256']==meta['source_sha256']
 assert previous['protection']['before']['matrix_sha256']==meta['protected_matrix_sha256']
 results['before']=previous['before'];protection['before']=previous['protection']['before']
 frame_list=[r['frame'] for r in previous['before']['per_frame']]
 before_body=np.load(baseline_path.parent/'audit-body-before.npy',mmap_mode='r')
 assert before_body.shape==(len(frame_list),len(body_vertices),3)
for variant in (('after',) if args.reuse_before else ('before','after')):
 bpy.ops.wm.open_mainfile(filepath=str(pair/f'{variant}.blend'),load_ui=False,use_scripts=False)
 scene=bpy.context.scene;rig=bpy.data.objects[plan['bindings']['armature']];mesh=bpy.data.objects[plan['bindings']['mesh']]
 frames=sorted(set(np.arange(scene.frame_start,scene.frame_end+.01,.5*args.stride).tolist())|{scene.frame_end})
 if frame_list is None:
  frame_list=frames;before_body=np.lib.format.open_memmap(report_path.parent/'audit-body-before.npy',mode='w+',dtype=np.float32,shape=(len(frames),len(body_vertices),3))
 else:assert frames==frame_list
 matrices=protected_matrices(rig,meta['protected_names'],frames)
 indices=np.rint((np.asarray(frames)-meta['frame_start'])*2).astype(int)
 baseline=np.load(root/'protected-before.npy',mmap_mode='r')[indices]
 assert np.array_equal(matrices,baseline),'主骨矩阵改变'
 scene.frame_set(scene.frame_start)
 assert digest_actions(set(meta['original_actions']))==meta['original_actions']
 assert mesh_digest(mesh)==meta['skin_morph_sha256']
 protection[variant]={'matrix_sha256':hashlib.sha256(matrices.tobytes()).hexdigest(),'samples':len(frames),'bones':48,'source_actions':len(meta['original_actions']),'all_source_actions_unchanged':True,'original_skin_and_shape_keys_unchanged':True}
 row={'samples':len(frames),'body_cloth_crossing_pairs':0,'body_cloth_crossing_frames':0,'body_cloth_max_pairs':0,
      'body_hair_crossing_pairs':0,'body_hair_crossing_frames':0,'body_hair_max_pairs':0,
      'cloth_hair_crossing_pairs':0,'cloth_hair_crossing_frames':0,'cloth_hair_max_pairs':0,
      'dynamic_floor_frames_over_1mm':0,'dynamic_floor_vertex_samples_over_1mm':0,'dynamic_floor_max_depth_m':0.,
      'protected_floor_frames_over_1mm':0,'protected_floor_max_depth_m':0.,'per_frame':[]}
 for channel in ('accessory_body','accessory_cloth','accessory_hair'):
  for suffix in ('crossing_pairs','crossing_frames','max_pairs'):row[channel+'_'+suffix]=0
 actor=[o for o in scene.objects if o.type=='MESH' and not o.hide_render and (o.name==mesh.name or o.name.startswith('MB4_RigidAttachment_'))]
 for index,frame in enumerate(frames):
  scene.frame_set(int(frame),subframe=frame-int(frame));points=[];faces=[];source_faces=[];offset=0
  main_points,main_ids,main_faces,main_fids=evaluated_mesh_records(mesh)
  lookup=np.full(len(plan['geometry']['positions_m']),-1,int);lookup[main_ids]=np.arange(len(main_ids))
  assert np.all(lookup[body_vertices]>=0),'保护表面缺失'
  body=main_points[lookup[body_vertices]]
  if variant=='before':before_body[index]=body
  else:max_body_error=max(max_body_error,float(np.linalg.norm(body-before_body[index],axis=1).max()))
  for obj in actor:
   p,ids,f,fids=evaluated_mesh_records(obj);points.append(p);faces.append(f+offset);source_faces.append(fids);offset+=len(p)
  p=np.concatenate(points);f=np.concatenate(faces);fids=np.concatenate(source_faces)
  bt=TriangleSurface(p,f[np.isin(fids,list(body_faces))]);ct=TriangleSurface(p,f[np.isin(fids,list(cloth_faces))]);ht=TriangleSurface(p,f[np.isin(fids,list(hair_faces))])
  counts={'body_cloth':len(bt.crossings(ct)),'body_hair':len(bt.crossings(ht)),'cloth_hair':len(ct.crossings(ht))}
  accessory_mask=np.isin(fids,list(accessory_faces))
  at=TriangleSurface(p,f[accessory_mask]) if accessory_mask.any() else None
  counts.update({name:len(at.crossings(target)) if at else 0 for name,target in [('accessory_body',bt),('accessory_cloth',ct),('accessory_hair',ht)]})
  for key,n in counts.items():
   row[key+'_crossing_pairs']+=n;row[key+'_crossing_frames']+=int(n>0);row[key+'_max_pairs']=max(row[key+'_max_pairs'],n)
  dynamic_ids=np.unique(f[~np.isin(fids,list(body_faces))]);depth=np.maximum(floor-p[dynamic_ids,2],0)
  protected_depth=np.maximum(floor-body[:,2],0)
  row['dynamic_floor_frames_over_1mm']+=int(np.any(depth>.001));row['dynamic_floor_vertex_samples_over_1mm']+=int(np.sum(depth>.001))
  row['dynamic_floor_max_depth_m']=max(row['dynamic_floor_max_depth_m'],float(depth.max(initial=0)))
  row['protected_floor_frames_over_1mm']+=int(np.any(protected_depth>.001));row['protected_floor_max_depth_m']=max(row['protected_floor_max_depth_m'],float(protected_depth.max(initial=0)))
  row['per_frame'].append({'frame':float(frame),**counts,'dynamic_floor_depth_m':float(depth.max(initial=0))})
  if index%120==0:print('ACTUAL_TRIANGLE_AUDIT',variant,index+1,len(frames),flush=True)
 if variant=='before':before_body.flush()
 for label,selected in [('integer',[r for r in row['per_frame'] if r['frame'].is_integer()]),('half',[r for r in row['per_frame'] if not r['frame'].is_integer()])]:
  row[label]={'samples':len(selected),'dynamic_floor_frames_over_1mm':sum(r['dynamic_floor_depth_m']>.001 for r in selected),'dynamic_floor_max_depth_m':max((r['dynamic_floor_depth_m'] for r in selected),default=0.)}
  for key in ('body_cloth','body_hair','cloth_hair','accessory_body','accessory_cloth','accessory_hair'):
   row[label][key+'_crossing_pairs']=sum(r[key] for r in selected)
   row[label][key+'_crossing_frames']=sum(r[key]>0 for r in selected)
 results[variant]=row
assert max_body_error<2e-6,'保护身体/靴子表面改变'
report={'protection_passed':True,'collision_quality_passed':False,'collision_quality_reason':'Residual counts retained; no zero-crossing claim',
 'attachment_audit_required':True,'accessory_source_triangles':len(accessory_faces),
 'method':'Actual saved visible mesh proper triangle crossings (no proxy), floor vertex depths; integer and half samples',
 'baseline_reused_from':str(args.reuse_before) if args.reuse_before else None,
 'frame_stride':args.stride,'protection':protection,'protected_body_boot_vertices':len(body_vertices),'protected_body_boot_max_world_change_m':max_body_error,
 'before':results['before'],'after':results['after'],
 'limitations':['逐样本真实三角审核，不是任意时刻连续CCD认证。','计数保留模型已有重叠，没有扣掉后声称零穿插。',
 '身体为非封闭服装模型，proper crossing计数不是准确穿透体积或深度。','发束自碰撞求解使用骨段胶囊；本审核未穷举发片内部自交。']}
report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('ACTUAL_SURFACE_AUDIT_DONE',json.dumps({k:report[k] for k in ['protection_passed','protected_body_boot_max_world_change_m']}),flush=True)
