from pathlib import Path
import json, sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from motionbridge.exporters import landmarks_to_motion
from diagnose_jitter import metrics
root=Path(__file__).resolve().parents[1]
for name,ident in [('dance','47b709e3e44540c5aa96c76f664c244c'),('tennis','b6b39f9009e442de882865a2c6310872')]:
    folder=root/'data/jobs'/ident/'outputs'
    old=json.loads((folder/'motion.json').read_text())
    a=np.load(folder/'observations.npz')
    new=landmarks_to_motion(a['world'],a['image'],a['confidence'],old['fps'],old['source'],old['settings'],hand_support=a['hand_support'])
    dest=root/'data/release-v051'/('probe-'+name); dest.mkdir(exist_ok=True)
    (dest/'motion.json').write_text(json.dumps(new))
    report=metrics(np.array(new['world_rotations']),np.array([f['positions'] for f in new['frames']]),new['fps'],new['joint_names'])
    (dest/'jitter.json').write_text(json.dumps(report,indent=2))
    print(name,{n:report[n] for n in ['Chest','LeftShoulder','LeftArm','LeftForeArm','LeftHand','RightHand']},flush=True)
