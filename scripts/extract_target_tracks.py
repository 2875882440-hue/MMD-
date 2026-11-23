"""Read the saved evaluated Blender rig for temporal comparison, without saving it."""
from pathlib import Path
import sys,json
import numpy as np
import bpy
folder,dest=map(Path,sys.argv[sys.argv.index('--')+1:])
report=json.loads((folder/'retarget_report.json').read_text(encoding='utf-8'))
bpy.ops.wm.open_mainfile(filepath=str(folder/'Nahida_MotionBridge_Dance.blend'),load_ui=False)
rig=bpy.data.objects['Nahida_arm']; scene=bpy.context.scene
names=list(report['mapped_bones']); rotations=[]; positions=[]
for f in range(scene.frame_start,scene.frame_end+1):
    scene.frame_set(f)
    rotations.append([np.asarray(rig.pose.bones[n].matrix.to_quaternion().to_matrix()) for n in names])
    positions.append([list(rig.pose.bones[n].head) for n in names])
np.savez_compressed(dest,world=np.asarray(rotations),positions=np.asarray(positions),names=names,fps=scene.render.fps/scene.render.fps_base)
print('TRACKS_EXTRACTED',dest,flush=True)
