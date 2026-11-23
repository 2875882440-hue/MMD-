from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
records=[]
for job in json.loads((ROOT/'research/v065/queue/results.json').read_text()):
    if job['status']!='completed':continue
    old=json.loads((ROOT/'data/jobs'/job['reused_inference_from']/'outputs/motion.json').read_text(encoding='utf-8'))
    new=json.loads((ROOT/'research/v065/queue/jobs'/job['job_id']/'outputs/motion.json').read_text(encoding='utf-8'))
    a=np.array([f['positions'] for f in old['frames']]);b=np.array([f['positions'] for f in new['frames']])
    records.append({'name':job['name'],'job_id':job['job_id'],'max_all_position_change_m':float(abs(a-b).max()),
        'max_all_rotation_change':float(abs(np.array(old['world_rotations'])-new['world_rotations']).max()),'head_collision':new['head_collision']})
(ROOT/'research/v065/metrics.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
before=ROOT/'research/v065/target-before.json';after=ROOT/'research/v065/target-after.json'
if before.exists() and after.exists():
    old=json.loads(before.read_text());new=json.loads(after.read_text());summary={}
    for side in ['left','right']:
        a=old['sides'][side];b=new['sides'][side];p=np.array(a['points']);q=np.array(b['points']);d=np.linalg.norm(p-q,axis=-1)
        row={'before_head':a['head'],'after_head':b['head'],'before_torso':a['torso'],'after_torso':b['torso'],
            'max_joint_adjustment_m':float(d.max()),'max_changed_frame':int(d.max(axis=1).argmax()+1),
            'new_head_penetrating_frames':int(((np.array(a['minimum_head_per_frame_m'])>=-.001)&(np.array(b['minimum_head_per_frame_m'])<-.001)).sum())}
        for label,x in [('old',p),('new',q)]:
            accel=np.linalg.norm(np.diff(x[:,2]-x[:,0],n=2,axis=0)*900,axis=-1)
            row[label+'_wrist_accel_p95']=float(np.percentile(accel,95));row[label+'_wrist_accel_max']=float(accel.max())
        summary[side]=row
    (ROOT/'research/v065/target-comparison-metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
