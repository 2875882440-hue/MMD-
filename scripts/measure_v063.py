from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
records=[]
for job in json.loads((ROOT/'research/v063/queue/results.json').read_text()):
    if job['status']!='completed':continue
    old=json.loads((ROOT/'data/jobs'/job['reused_inference_from']/'outputs/motion.json').read_text(encoding='utf-8'))
    new=json.loads((ROOT/'research/v063/queue/jobs'/job['job_id']/'outputs/motion.json').read_text(encoding='utf-8'))
    before=np.array([f['positions'] for f in old['frames']]);after=np.array([f['positions'] for f in new['frames']])
    a=np.array(old['world_rotations']);b=np.array(new['world_rotations'])
    report={'name':job['name'],'job_id':job['job_id'],'root_horizontal_max_change_m':float(abs(before[:,0][:,[0,2]]-after[:,0][:,[0,2]]).max()),
        'upper_body_rotation_max_change':float(abs(a[:,[0,3,6,9,12,13,14,15,16,17,18,19,20,21]]-b[:,[0,3,6,9,12,13,14,15,16,17,18,19,20,21]]).max()),
        'root_height_max_change_m':float(abs(before[:,0,1]-after[:,0,1]).max()),
        'collision':{k:v for k,v in new['floor_collision'].items() if k!='root_lift_m'}}
    for label,w in [('old',a),('new',b)]:
        d=w[:-1].swapaxes(-1,-2)@w[1:]
        speed=np.rad2deg(Rotation.from_matrix(d.reshape(-1,3,3)).magnitude()).reshape(-1,22)*new['fps']
        report[label+'_leg_peak_deg_s']=float(speed[:,[1,2,4,5]].max())
    records.append(report)
(ROOT/'research/v063/metrics.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False,indent=2))
