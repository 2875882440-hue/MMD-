"""Render already-completed production jobs serially; never re-run recognition."""
from pathlib import Path
import json
import sys
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from motionbridge.retarget import run_script
root=Path(__file__).resolve().parents[1]
release=root/'data/release-v040'
ids=json.loads((release/'final_jobs.json').read_text())
blender=r'D:\STEAM\steamapps\common\Blender\blender.exe'
for name,times in [('dance',[1,4,7,10,13,16,19,23,28]),('tennis',[1,4,7])]:
    job=root/'data/jobs'/ids[name]
    output=job/'outputs'
    folder=release/(name+'-final-review')
    folder.mkdir(exist_ok=True)
    motion=json.loads((output/'motion.json').read_text(encoding='utf-8'))
    (folder/'assessment.json').write_text(json.dumps({'outputs':str(output),'source_filename':str(job/'input/source.mp4'),
        'sample_times':times,'source_duration':motion['num_frames']/motion['fps']}),encoding='utf-8')
    for phase in ['probe','samples']:
        run_script(blender,root/'scripts/render_dance_review.py',[folder,phase],folder/(phase+'.log'),lambda:False)
    subprocess.run([sys.executable,str(root/'scripts/compose_dance_review.py'),str(folder),str(job/'input/source.mp4'),'--sheets-only'],check=True)
target=root/'data/jobs'/ids['character']/'outputs'
for phase in ['samples','preview']:
    run_script(blender,root/'scripts/render_retarget_review.py',[target,phase],release/('character-'+phase+'.log'),lambda:False,timeout=900)
    print('character rendered',phase,flush=True)
subprocess.run([sys.executable,str(root/'scripts/compose_retarget_review.py'),str(target),'--source',str(root/'data/jobs'/ids['dance']/'input/source.mp4')],check=True)
