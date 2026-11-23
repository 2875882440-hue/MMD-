from pathlib import Path
import sys,json,numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motionbridge.floor_collision import foot_proxy_heights
records=[]
for job in json.loads((ROOT/'research/v064/queue/results.json').read_text()):
    if job['status']!='completed':continue
    old=json.loads((ROOT/'data/jobs'/job['reused_inference_from']/'outputs/motion.json').read_text(encoding='utf-8'))
    new=json.loads((ROOT/'research/v064/queue/jobs'/job['job_id']/'outputs/motion.json').read_text(encoding='utf-8'))
    before=np.array([f['positions'] for f in old['frames']]);after=np.array([f['positions'] for f in new['frames']])
    a=np.array(old['world_rotations']);b=np.array(new['world_rotations'])
    report={'name':job['name'],'job_id':job['job_id'],'root_max_change_m':float(abs(before[:,0]-after[:,0]).max()),
        'torso_legs_rotation_max_change':float(abs(a[:,:16]-b[:,:16]).max()),
        'feet_max_change_m':float(abs(before[:,[7,8,10,11]]-after[:,[7,8,10,11]]).max()),
        'floor_proxy_min_m':float(foot_proxy_heights(after,b,new['rest_positions']).min()),
        'collision':new['body_collision']}
    for label,w in [('old',a),('new',b)]:
        d=w[:-1].swapaxes(-1,-2)@w[1:]
        speed=np.rad2deg(Rotation.from_matrix(d.reshape(-1,3,3)).magnitude()).reshape(-1,22)*new['fps']
        report[label+'_arm_peak_deg_s']=float(speed[:,16:].max())
        report[label+'_arm_p95_deg_s']=float(np.percentile(speed[:,16:],95))
    records.append(report)
(ROOT/'research/v064/metrics.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False,indent=2))
