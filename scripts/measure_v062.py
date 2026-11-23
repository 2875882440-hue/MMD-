"""Compare preserved v0.6.1 with v0.6.2; all values are diagnostics, not truth."""
from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
records=[]
for name,jid in [('dance','78d52a9ac4204f99bdf61469cb6495ee'),('tennis','bd1897dc5789411286ca6089679748a0')]:
    olddir=ROOT/'data/jobs'/jid/'outputs';newdir=ROOT/'research/v062'/name
    old=json.loads((olddir/'motion.json').read_text(encoding='utf-8'));new=json.loads((newdir/'motion.json').read_text(encoding='utf-8'))
    a=np.load(olddir/'gvhmr/motion_world.npz');kp=np.load(olddir/'gvhmr/keypoints_2d.npz')['keypoints']
    wc=Rotation.from_rotvec(a['camera_global_orient']).as_matrix()@Rotation.from_rotvec(a['world_global_orient']).as_matrix().swapaxes(-1,-2)
    residual=np.einsum('fij,fkj->fki',wc,a['joints_world']-a['joints_world'][:,:1])+a['joints_camera'][:,:1]-a['joints_camera']
    rec={'name':name,'accuracy_measured':False,'world_to_camera_consistency_max_m':float(np.max(abs(residual)))}
    for label,m in [('v061',old),('v062',new)]:
        p=np.array([f['positions'] for f in m['frames']]);w=np.array(m['world_rotations']);fps=m['fps']
        delta=w[:-1].swapaxes(-1,-2)@w[1:]
        speed=np.rad2deg(Rotation.from_matrix(delta.reshape(-1,3,3)).magnitude()).reshape(-1,22)*fps
        wrists=p[:,[20,21]]-p[:,9:10];acc=np.linalg.norm(np.diff(wrists,n=2,axis=0)*fps*fps,axis=-1)
        cam=np.einsum('fij,fkj->fki',wc,p-p[:,:1])+a['joints_camera'][:,:1]
        projected=np.einsum('fij,fkj->fki',a['K'],cam[:,[18,19,20,21]])
        px=projected[...,:2]/projected[...,2:];valid=kp[:,[7,8,9,10],2]>=.7
        err=np.linalg.norm(px-kp[:,[7,8,9,10],:2],axis=-1)
        rec[label]={'arm_peak_world_deg_s':float(speed[:,[16,17,18,19]].max()),
            'arm_p95_world_deg_s':float(np.percentile(speed[:,[16,17,18,19]],95)),
            'wrist_relative_acceleration_p95_m_s2':float(np.percentile(acc,95)),
            'wrist_relative_axis_ranges_m':np.ptp(wrists,axis=0).tolist(),
            'root_aligned_arm_projection_mean_px':float(err[valid].mean()),
            'grounded_anchor_segments':m['world_contact_refinement'].get('grounded_anchor_segments',0)}
    op=np.array([f['positions'] for f in old['frames']]);np_=np.array([f['positions'] for f in new['frames']])
    rec['source_root_positions_max_difference_m']=float(np.max(abs(op[:,0]-np_[:,0])))
    rec['source_root_rotations_max_difference']=float(np.max(abs(np.array(old['world_rotations'])[:,0]-np.array(new['world_rotations'])[:,0])))
    rec['contact_weight_max_difference']=float(np.max(abs(np.array(old['foot_contacts']['weights'])-np.array(new['foot_contacts']['weights']))))
    records.append(rec)
targets={}
for version in ['v061','v062']:
    audit=json.loads((ROOT/'research'/version/'nahida/saved_project_audit.json').read_text(encoding='utf-8'))
    floor=np.array([f['min'][2] for f in audit['mesh_bounds_samples']])
    targets[version]={'max_frame_rotation':audit['maximum_adjacent_world_rotation_change'],
        'sampled_mesh_floor_min_m':float(floor.min()),'sampled_mesh_floor_max_m':float(floor.max()),
        'sampled_mesh_floor_mean_absolute_m':float(abs(floor).mean()),'sample_count':len(floor)}
report={'clips':records,'target':targets,'note':'No ground truth; root-aligned projection uses the same corrected 2D observations. Mesh bounds are sparse samples, not sole contacts.'}
(ROOT/'research/v062/metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
