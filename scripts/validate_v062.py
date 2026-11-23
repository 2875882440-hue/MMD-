"""Reuse unchanged v0.6.1 model inference for isolated postprocessing regression."""
from pathlib import Path
import sys,json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motionbridge.world_stability import stabilize_world_upper_body
from motionbridge.world_contacts import refine_world_contacts
from motionbridge.exporters import export_bvh,export_blender
from motionbridge.retarget import export_character
blender=r'D:\STEAM\steamapps\common\Blender\blender.exe'
for label,jid in [('dance','78d52a9ac4204f99bdf61469cb6495ee'),('tennis','bd1897dc5789411286ca6089679748a0')]:
    source=ROOT/'data/jobs'/jid/'outputs';out=ROOT/'research/v062'/label
    out.mkdir(parents=True,exist_ok=True)
    raw=json.loads((source/'motion_raw.json').read_text(encoding='utf-8'))
    kp=np.load(source/'gvhmr/keypoints_2d.npz')['keypoints']
    stable=stabilize_world_upper_body(raw)
    final=refine_world_contacts(stable,kp)
    for name,m in [('motion_raw.json',raw),('motion_stable.json',stable),('motion.json',final)]:
        (out/name).write_text(json.dumps(m,ensure_ascii=False),encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps({'source_job':jid,'reused':'unchanged GVHMR inference and corrected 2D keypoints','new':'local rotation stability and grounded stance anchors'},indent=2))
    export_bvh(final,out/'motion.bvh')
    export_blender(out/'motion.json',out,blender)
    print(label,final['world_stability'],flush=True)
config=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
(ROOT/'research/v062/nahida').mkdir(exist_ok=True)
export_character(ROOT,blender,config['characters']['nahida'],ROOT/'research/v062/dance/motion.json',ROOT/'research/v062/nahida',lambda:False,lambda p,s:print(p,s,flush=True))
