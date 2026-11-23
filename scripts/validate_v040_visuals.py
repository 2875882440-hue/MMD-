from pathlib import Path
import sys
import json
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from motionbridge.retarget import character_profiles, export_character, run_script
root=Path(__file__).resolve().parents[1]
dest=root/'data/release-v040'
blender=r'D:\STEAM\steamapps\common\Blender\blender.exe'
for name,ident,times,fps in [('dance','0b1e96ce61304e9b84d29cee9d025245',[1,4,7,10,13,16,19,23,28],30),('tennis','f8585fecb46c42b5b171b5328c6d64ce',[1,4,7],24)]:
    folder=dest/(name+'-review')
    folder.mkdir(exist_ok=True)
    output=dest/name/'outputs'
    motion=json.loads((output/'motion.json').read_text(encoding='utf-8'))
    source=root/'data/jobs'/ident/'input/source.mp4'
    (folder/'assessment.json').write_text(json.dumps({'outputs':str(output),'source_filename':str(source),'sample_times':times,'source_duration':motion['num_frames']/fps}),encoding='utf-8')
    for phase in ['probe','samples']:
        run_script(blender,root/'scripts/render_dance_review.py',[folder,phase],folder/(phase+'.log'),lambda:False)
    subprocess.run([sys.executable,str(root/'scripts/compose_dance_review.py'),str(folder),str(source),'--sheets-only'],check=True)
target=dest/'character'
target.mkdir(exist_ok=True)
report=export_character(root,blender,character_profiles(root,blender)[0],dest/'dance/outputs/motion.json',target,lambda:False,lambda p,s:print(p,s,flush=True))
print('TARGET_AUDIT',report['passed'],flush=True)
run_script(blender,root/'scripts/render_retarget_review.py',[target,'samples'],target/'samples.log',lambda:False)
subprocess.run([sys.executable,str(root/'scripts/compose_retarget_review.py'),str(target),'--source',str(root/'data/jobs/0b1e96ce61304e9b84d29cee9d025245/input/source.mp4'),'--sheets-only'],check=True)
