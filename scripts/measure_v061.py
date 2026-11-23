"""Ablation diagnostics on fixed development clips, never capture accuracy."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motionbridge.world_contacts import refine_world_contacts
results=[]
for name,ident in [('dance','f2bdc1313b3343c39f14a55dcb8a35d5'),('tennis','21c5391351334cb5bcbf13cfdcfbf963')]:
    original=ROOT/'data/jobs'/ident/'outputs';trial=ROOT/'research/v061'/name
    old=json.loads((original/'motion.json').read_text(encoding='utf-8'))
    raw=json.loads((trial/'motion_raw.json').read_text(encoding='utf-8'))
    kp=np.load(trial/'keypoints_2d.npz')['keypoints']
    refined=refine_world_contacts(raw,kp)
    record={'name':name,'accuracy_measured':False,'ablation':{},'contact_report':refined['world_contact_refinement']}
    for label,m in [('v060',old),('arm_correction',raw),('arm_and_contacts',refined)]:
        w=np.array(m['world_rotations']);p=np.array([f['positions'] for f in m['frames']])
        speed=np.rad2deg(Rotation.from_matrix((w[:-1].swapaxes(-1,-2)@w[1:]).reshape(-1,3,3)).magnitude()).reshape(-1,22)*m['fps']
        forward=w[:,0,:,2];dot=forward@forward[0]
        record['ablation'][label]={'arm_max_deg_s':float(speed[:,[16,17,18,19]].max()),
            'arm_p95_deg_s':float(np.percentile(speed[:,[16,17,18,19]],95)),
            'leg_max_deg_s':float(speed[:,[1,2,4,5]].max()),
            'root_range_m':np.ptp(p[:,0],axis=0).tolist(),
            'toe_height_first_30_m':np.median(p[:30,[10,11],1],axis=0).tolist(),
            'toe_height_last_30_m':np.median(p[-30:,[10,11],1],axis=0).tolist(),
            'facing_dot_initial_at_14s':float(dot[420]) if len(dot)>420 else None}
    repaired=json.loads((trial/'arm_report.json').read_text())
    record['arm_report']=repaired
    # Residual to the SAME unmodified image observations on each side; still no truth.
    old_kp=np.load(original/'gvhmr/keypoints_2d.npz')['keypoints']
    unchanged=np.max(abs(kp[:,[7,8,9,10],:2]-old_kp[:,[7,8,9,10],:2]),axis=-1)<.001
    reliable=unchanged&(old_kp[:,[7,8,9,10],2]>=.7)
    record['unchanged_observation_reprojection']={}
    for label,path in [('v060',original/'gvhmr/motion_world.npz'),('new',trial/'motion_world.npz')]:
        a=np.load(path);cam=a['joints_camera'][:,[18,19,20,21]]
        projected=np.einsum('fij,fkj->fki',a['K'],cam);pixel=projected[...,:2]/projected[...,2:]
        error=np.linalg.norm(pixel-old_kp[:,[7,8,9,10],:2],axis=-1)
        record['unchanged_observation_reprojection'][label]={'mean_px':float(error[reliable].mean()),
            'median_px':float(np.median(error[reliable])),'count':int(reliable.sum())}
    results.append(record)
out=ROOT/'research/v061/ablation_metrics.json'
out.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
for record in results:
    print(record['name'],json.dumps(record['ablation']),record['unchanged_observation_reprojection'],flush=True)
