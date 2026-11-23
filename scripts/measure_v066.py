"""Observation agreement, motion preservation and continuity; NOT accuracy."""
from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
records=[]
for job in json.loads((ROOT/'research/v066/queue/results.json').read_text()):
    olddir=ROOT/'data/jobs'/job['reused_inference_from']/'outputs'
    newdir=ROOT/'research/v066/queue/jobs'/job['job_id']/'outputs'
    old=json.loads((olddir/'motion.json').read_text(encoding='utf-8'));new=json.loads((newdir/'motion.json').read_text(encoding='utf-8'))
    with np.load(newdir/'gvhmr/motion_world.npz',allow_pickle=False) as a:
        wc=Rotation.from_rotvec(a['camera_global_orient']).as_matrix()@Rotation.from_rotvec(a['world_global_orient']).as_matrix().swapaxes(-1,-2)
        root_camera=a['joints_camera'][:,:1];K=a['K']
    kp=np.load(newdir/'gvhmr/keypoints_2d.npz',allow_pickle=False)['keypoints']
    report={'name':job['name'],'image_refinement':new['image_arm_refinement'],'accuracy_measured':False}
    for label,m in [('old',old),('new',new)]:
        p=np.array([f['positions'] for f in m['frames']]);w=np.array(m['world_rotations']);fps=m['fps']
        cam=np.einsum('fij,fkj->fki',wc,p-p[:,:1])+root_camera;h=np.einsum('fij,fkj->fki',K,cam);px=h[:,:,:2]/h[:,:,2:]
        arm_errors=[]
        for arm,obs in [([16,18,20],[5,7,9]),([17,19,21],[6,8,10])]:
            desired=px[:,arm[0],None]+kp[:,obs[1:],:2]-kp[:,obs[0],None,:2]
            trusted=np.min(kp[:,obs,2],axis=1)>.7
            arm_errors.extend(np.linalg.norm(px[:,arm[1:]]-desired,axis=-1)[trusted].ravel())
        vel=Rotation.from_matrix((w[:-1].swapaxes(-1,-2)@w[1:]).reshape(-1,3,3)).magnitude().reshape(-1,22)*fps*180/np.pi
        acceleration=np.linalg.norm(np.diff(p[:,[20,21]]-p[:,9:10],n=2,axis=0)*fps**2,axis=-1)
        report[label]={'mean_shoulder_relative_error_px':float(np.mean(arm_errors)),
            'arm_peak_deg_s':float(vel[:,16:].max()),'arm_p95_deg_s':float(np.percentile(vel[:,16:],95)),
            'wrist_accel_p95_m_s2':float(np.percentile(acceleration,95)),'wrist_accel_max_m_s2':float(acceleration.max())}
    a=np.array([f['positions'] for f in old['frames']]);b=np.array([f['positions'] for f in new['frames']])
    report['root_max_change_m']=float(abs(a[:,0]-b[:,0]).max())
    report['feet_max_change_m']=float(abs(a[:,[7,8,10,11]]-b[:,[7,8,10,11]]).max())
    report['torso_head_legs_rotation_max_change']=float(abs(np.array(old['world_rotations'])[:,:16]-np.array(new['world_rotations'])[:,:16]).max())
    records.append(report)
(ROOT/'research/v066/metrics.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False,indent=2))
