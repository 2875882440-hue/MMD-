"""Whole-sequence cloth-component backoff from sampled actual triangles.

Proposals only: the assembled result must still pass the full integer/half
actual-surface gate. No frame-dependent switching, no changes to body or hair.
"""
from pathlib import Path
import argparse,json,shutil,sys
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from motionbridge.secondary_surface import new_child_output
from motionbridge.secondary_surface_config import surface_settings
from motionbridge.blender_secondary_rigid import evaluated_mesh_records
from motionbridge.blender_triangle_contacts import TriangleSurface
parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--pair-dir',type=Path,required=True)
parser.add_argument('--cloth-dir',type=Path,required=True);parser.add_argument('--tag',default='cloth-guarded')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=args.input_dir.resolve();pair=args.pair_dir.resolve();cloth=args.cloth_dir.resolve();out=new_child_output(root,args.tag)
if not root.is_relative_to(ROOT/'output/secondary_motion') or not pair.is_relative_to(root) or not cloth.is_relative_to(root) or out.exists():raise ValueError('使用同源新目录')
meta=json.loads((root/'input.json').read_text(encoding='utf-8'));plan=json.loads((root/'model-plan.json').read_text(encoding='utf-8'))
mapping=np.load(cloth/'cloth-index-map.npz');faces=mapping['welded_faces'];inverse=mapping['inverse'];ids=mapping['source_ids']
parent=np.arange(len(mapping['representatives']))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=int(parent[i])
 return i
for face in faces:
 a=find(int(face[0]))
 for vertex in face[1:]:parent[find(int(vertex))]=a
labels=np.asarray([find(i) for i in range(len(parent))]);_,labels=np.unique(labels,return_inverse=True);source_labels=labels[inverse];n=int(labels.max()+1)
vertex_labels=np.full(len(plan['geometry']['positions_m']),-1,int);vertex_labels[ids]=source_labels
face_labels=np.max(vertex_labels[np.asarray(plan['geometry']['triangles'],int)],axis=1)
body_ids=set(meta['body_source_face_ids']);cloth_ids={f for r in plan['regions'] if r['role']=='cloth' for f in r['face_ids']};hair_ids={f for r in plan['regions'] if r['role']=='hair' for f in r['face_ids']}
frames=sorted(set(range(meta['frame_start'],meta['frame_end']+1,15))|{meta['frame_end']});metrics={}
for variant in ('before','after'):
 bpy.ops.wm.open_mainfile(filepath=str(pair/f'{variant}.blend'),load_ui=False,use_scripts=False)
 scene=bpy.context.scene;actor=[o for o in scene.objects if o.type=='MESH' and not o.hide_render and (o.name==plan['bindings']['mesh'] or o.name.startswith('MB4_RigidAttachment_'))]
 body_count=np.zeros(n,int);hair_count=np.zeros(n,int)
 for index,frame in enumerate(frames):
  scene.frame_set(frame);ps=[];fs=[];fids=[];offset=0
  for obj in actor:
   p,_,f,source=evaluated_mesh_records(obj);ps.append(p);fs.append(f+offset);fids.append(source);offset+=len(p)
  p=np.concatenate(ps);f=np.concatenate(fs);fids=np.concatenate(fids)
  mask=np.isin(fids,list(cloth_ids));cloth_labels=face_labels[fids[mask]]
  ct=TriangleSurface(p,f[mask]);bt=TriangleSurface(p,f[np.isin(fids,list(body_ids))]);ht=TriangleSurface(p,f[np.isin(fids,list(hair_ids))])
  for target,total in [(bt,body_count),(ht,hair_count)]:
   pairs=target.crossings(ct)
   if pairs:
    components=cloth_labels[np.asarray(list(pairs),int)[:,1]];components=components[components>=0]
    total+=np.bincount(components,minlength=n)
  if index%15==0:print('COMPONENT_GUARD_SAMPLE',variant,index+1,len(frames),flush=True)
 metrics[variant]={'body':body_count,'hair':hair_count}
accepted=(metrics['after']['body']<=metrics['before']['body'])&(metrics['after']['hair']<=metrics['before']['hair'])
source=np.load(root/'source-positions.npy',mmap_mode='r');cfg=surface_settings(json.loads((root/'profile.json').read_text(encoding='utf-8')))
floor=cfg['floor_z_m']+cfg['body_gap_m'];mandatory=np.zeros(n,bool)
protected_vertices={v for r in plan['regions'] if r['role'] in ('body','boots') for v in r['vertex_ids']};movable=~np.isin(ids,list(protected_vertices))
for frame in source:
 bad=(frame[ids,2]<floor)&movable
 if bad.any():mandatory[np.unique(source_labels[bad])]=True
quality_accepted=accepted.copy();accepted|=mandatory
records=[{'component':i,'accepted':bool(accepted[i]),'sample_quality_accepted':bool(quality_accepted[i]),'mandatory_floor':bool(mandatory[i]),'vertices':int(np.sum(source_labels==i)),
          'body_before':int(metrics['before']['body'][i]),'body_after':int(metrics['after']['body'][i]),
          'hair_before':int(metrics['before']['hair'][i]),'hair_after':int(metrics['after']['hair'][i])} for i in range(n)]
out.mkdir();candidate=np.load(cloth/'cloth-positions.npy',mmap_mode='r');source=np.load(root/'source-positions.npy',mmap_mode='r')
output=np.lib.format.open_memmap(out/'cloth-positions.npy',mode='w+',dtype=np.float32,shape=candidate.shape)
keep=accepted[source_labels]
for index in range(len(candidate)):
 output[index]=source[index,ids];output[index,keep]=candidate[index,keep]
output.flush();shutil.copy2(cloth/'cloth-index-map.npz',out/'cloth-index-map.npz')
report={'scope':'sampled proposal only; final full integer/half gate required','sample_frames':frames,'whole_track_decisions':True,
 'quality_selected_components':int(quality_accepted.sum()),'mandatory_floor_components':np.flatnonzero(mandatory).tolist(),
 'floor_priority_note':'Never restore a garment component to an underground kinematic target; final collision gate still applies',
 'no_per_frame_switching':True,'accepted_components':int(accepted.sum()),'rejected_components':int((~accepted).sum()),'records':records,'source_cloth':str(cloth)}
(out/'component-guard.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
original=json.loads((cloth/'cloth-validation.json').read_text(encoding='utf-8'));original['component_guard']=report
(out/'cloth-validation.json').write_text(json.dumps(original,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPONENT_GUARD_DONE',int(accepted.sum()),n,flush=True)
