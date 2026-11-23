"""Read-only, same-calibration head/torso proxy comparison for Nahida scenes."""
from pathlib import Path
import sys,json
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motionbridge.body_collision import torso_space,arm_clearance
from motionbridge.head_collision import clearance_report
from motionbridge.blender_head import calibrate_face_proxy,animated_head_proxy
blend,destination=map(Path,sys.argv[sys.argv.index('--')+1:])
bpy.ops.wm.open_mainfile(filepath=str(blend.resolve()),load_ui=False)
scene=bpy.context.scene;rig=bpy.data.objects['Nahida_arm']
profile=calibrate_face_proxy(rig,bpy.data.objects['Nahida_mesh'],['颜'],'頭')
names=['腰','左腕','右腕','左ひじ','右ひじ','左手首','右手首'];tracks={n:[] for n in names};matrices=[]
for f in range(scene.frame_start,scene.frame_end+1):
    scene.frame_set(f)
    for n in names:tracks[n].append(list(rig.pose.bones[n].head))
    matrices.append(np.array(rig.pose.bones['頭'].matrix))
tracks={n:np.array(v) for n,v in tracks.items()}
def rest(n):return np.array(rig.data.bones[n].head_local)
torso=torso_space(tracks['腰'],tracks['左腕'],tracks['右腕'],rest('腰'),rest('左腕'),rest('右腕'))
hc,hb,hr=animated_head_proxy(matrices,np.array(rig.data.bones['頭'].matrix_local),profile)
radius=float(np.linalg.norm(rest('左腕')-rest('右腕'))*.12)
result={'scene':str(blend.resolve()),'profile':profile,'arm_radius_m':radius,'sides':{},
        'note':'Head/torso geometric proxies, not evaluated face mesh or motion accuracy.'}
for side,label in [('左','left'),('右','right')]:
    points=np.stack([tracks[side+n] for n in ['腕','ひじ','手首']],axis=1)
    head=arm_clearance(points,hc,hb,hr,radius);body=arm_clearance(points,*torso)
    result['sides'][label]={'head':clearance_report(head),'torso':clearance_report(body),
        'minimum_head_per_frame_m':head.min(axis=1).tolist(),'points':points.tolist()}
destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
