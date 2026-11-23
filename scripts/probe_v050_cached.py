"""Fast diagnostic on existing observations, without re-running inference."""
from pathlib import Path
import json, sys, time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from motionbridge.exporters import landmarks_to_motion, export_bvh
root=Path(__file__).resolve().parents[1]
for name,ident in [('dance','b6f9d511e8214257b7df4fab6df2196f'),('tennis','288450f7d88848609abd9d7c52351f8e')]:
    source=root/'data/jobs'/ident/'outputs'
    old=json.loads((source/'motion.json').read_text())
    a=np.load(source/'observations.npz'); begin=time.monotonic()
    new=landmarks_to_motion(a['world'],a['image'],a['confidence'],old['fps'],old['source'],old['settings'],hand_support=a['hand_support'])
    out=root/'data/release-v050'/('cached-'+name)
    out.mkdir(parents=True,exist_ok=True)
    (out/'motion.json').write_text(json.dumps(new),encoding='utf-8')
    export_bvh(new,out/'motion.bvh')
    print(name,round(time.monotonic()-begin,2),new['projection_refinement'],flush=True)
    for t in ([1,4,7,10,16,28] if name=='dance' else [1,4,7]):
        f=round(t*new['fps'])
        p=np.array(new['frames'][f]['positions']); q=np.array(old['frames'][f]['positions'])
        print(t,'old wrist-head', (q[[8,12]]-q[4]).round(3).tolist(),'new',(p[[8,12]]-p[4]).round(3).tolist(),flush=True)
