from pathlib import Path
import json, sys
import numpy as np
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parents[1]
dest=root/'data/release-v051'; dest.mkdir(exist_ok=True)
def metrics(world,points,fps,names):
    delta=Rotation.from_matrix((world[1:]@np.swapaxes(world[:-1],-1,-2)).reshape(-1,3,3)).as_rotvec().reshape(len(world)-1,-1,3)
    speeds=np.linalg.norm(delta,axis=-1)*fps*180/np.pi
    accel=np.linalg.norm(np.diff(delta,axis=0),axis=-1)*fps*fps*180/np.pi
    reversal=(np.sum(delta[1:]*delta[:-1],axis=-1)<0)&(np.minimum(np.linalg.norm(delta[1:],axis=-1),np.linalg.norm(delta[:-1],axis=-1))>np.deg2rad(2))
    out={}
    for j,name in enumerate(names):
        out[name]={'speed_p95':float(np.percentile(speeds[:,j],95)), 'accel_p95':float(np.percentile(accel[:,j],95)),
                   'reversals':int(reversal[:,j].sum()),'position_second_diff_p95_m':float(np.percentile(np.linalg.norm(np.diff(points[:,j],n=2,axis=0),axis=-1),95))}
    return out
if __name__=='__main__':
    report={}
    for version,ident in [('v040','b6f9d511e8214257b7df4fab6df2196f'),('v050','47b709e3e44540c5aa96c76f664c244c')]:
        m=json.loads((root/'data/jobs'/ident/'outputs/motion.json').read_text())
        report[version]=metrics(np.array(m['world_rotations']),np.array([f['positions'] for f in m['frames']]),m['fps'],m['joint_names'])
    (dest/'source_baseline_jitter.json').write_text(json.dumps(report,indent=2))
    for n in ['Chest','LeftShoulder','LeftArm','LeftForeArm','LeftHand','RightArm','RightForeArm','RightHand','LeftUpLeg']:
        print(n, {v:r[n] for v,r in report.items()})
