"""Read-only Nahida arm/torso proxy audit for before/after comparisons."""
from pathlib import Path
import sys,json
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motionbridge.body_collision import torso_space,arm_clearance
blend,destination=map(Path,sys.argv[sys.argv.index('--')+1:])
bpy.ops.wm.open_mainfile(filepath=str(blend.resolve()),load_ui=False)
scene=bpy.context.scene;rig=bpy.data.objects['Nahida_arm']
names=['腰','左腕','右腕','左ひじ','右ひじ','左手首','右手首'];tracks={n:[] for n in names}
for f in range(scene.frame_start,scene.frame_end+1):
    scene.frame_set(f)
    for n in names:tracks[n].append(list(rig.pose.bones[n].head))
tracks={n:np.array(v) for n,v in tracks.items()}
def rest(n):return np.array(rig.data.bones[n].head_local)
center,basis,radii,radius=torso_space(tracks['腰'],tracks['左腕'],tracks['右腕'],rest('腰'),rest('左腕'),rest('右腕'))
result={'scene':str(blend.resolve()),'radii_m':radii.tolist(),'arm_radius_m':radius,'sides':{},
        'note':'Proportion-based sampled bone proxy audit, not body mesh/cloth intersection or capture accuracy.'}
for side,label in [('左','left'),('右','right')]:
    points=np.stack([tracks[side+n] for n in ['腕','ひじ','手首']],axis=1)
    clearance=arm_clearance(points,center,basis,radii,radius)
    result['sides'][label]={'penetrating_frames':int((clearance.min(axis=1)<-.001).sum()),
        'min_clearance_m':float(clearance.min()),'total_proxy_deficit_m':float(np.maximum(-clearance,0).sum()),
        'minimum_per_frame_m':clearance.min(axis=1).tolist(),'points':points.tolist()}
destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
