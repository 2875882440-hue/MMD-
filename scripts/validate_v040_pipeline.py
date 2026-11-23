"""Real-video v0.4 candidate regression, independent outputs and CPU inference."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from motionbridge.pipeline import process_video
from motionbridge.exporters import export_blender, export_bvh

root=Path(__file__).resolve().parents[1]
for name,ident,fps in [('dance','0b1e96ce61304e9b84d29cee9d025245',30),('tennis','f8585fecb46c42b5b171b5328c6d64ce',24)]:
    output=root/'data/release-v040'/name/'outputs'
    output.mkdir(parents=True,exist_ok=True)
    if not (output/'motion.json').exists():
        result=process_video(root/'data/jobs'/ident/'input/source.mp4', output,
             {'sample_fps':fps,'smoothing':.35,'model':'full','refine_motion':True,'stabilize_rotations':True,'enhance_motion':True},
             lambda p,s:print(name,round(p),s,flush=True), lambda:False)
        export_bvh(result['motion'],output/'motion.bvh')
    if not (output/'blender_verification.json').exists():
        export_blender(output/'motion.json',output,r'D:\STEAM\steamapps\common\Blender\blender.exe')
    m=json.loads((output/'motion.json').read_text(encoding='utf-8'))
    print(name,json.dumps({key:m.get(key) for key in ['hand_assistance','rotation_stabilization']}),flush=True)
    print(name,json.dumps({k:v for k,v in m.get('foot_contacts',{}).items() if k not in ['weights','segments']}),flush=True)
