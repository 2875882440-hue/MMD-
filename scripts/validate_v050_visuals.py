"""Serial visual review of completed production outputs; preserve older jobs."""
from pathlib import Path
import sys, json, subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from motionbridge.retarget import run_script
root=Path(__file__).resolve().parents[1]; dest=root/'data/release-v050'
ids=json.loads((dest/'final_jobs.json').read_text())
blender=r'D:\STEAM\steamapps\common\Blender\blender.exe'
for name,times in [('dance',[1,4,7,10,13,16,19,23,28]),('tennis',[1,4,7])]:
    job=root/'data/jobs'/ids[name]; output=job/'outputs'
    assert json.loads((job/'job.json').read_text())['status']=='completed'
    folder=dest/(name+'-review'); folder.mkdir(exist_ok=True)
    m=json.loads((output/'motion.json').read_text())
    (folder/'assessment.json').write_text(json.dumps({'outputs':str(output),'source_filename':str(job/'input/source.mp4'),
        'sample_times':times,'source_duration':m['num_frames']/m['fps']}),encoding='utf-8')
    for phase in ['probe','samples']:
        run_script(blender,root/'scripts/render_dance_review.py',[folder,phase],folder/(phase+'.log'),lambda:False)
    subprocess.run([sys.executable,str(root/'scripts/compose_dance_review.py'),str(folder),str(job/'input/source.mp4'),'--sheets-only'],check=True)
target=root/'data/jobs'/ids['character']/'outputs'
assert json.loads((target.parent/'job.json').read_text())['status']=='completed'
for phase in ['samples','preview']:
    run_script(blender,root/'scripts/render_retarget_review.py',[target,phase],dest/('character-'+phase+'.log'),lambda:False,timeout=900)
    print('Rendered',phase,flush=True)
subprocess.run([sys.executable,str(root/'scripts/compose_retarget_review.py'),str(target),'--source',str(root/'data/jobs'/ids['dance']/'input/source.mp4')],check=True)
import imageio_ffmpeg
r=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(target/'Nahida_Dance_Comparison.mp4'),
    '-map','0:v:0','-map','0:a:0','-f','null','-'],capture_output=True,text=True,timeout=120)
(target/'av_decode_verification.json').write_text(json.dumps({'passed':r.returncode==0 and not r.stderr.strip(),'returncode':r.returncode,'stderr':r.stderr}),encoding='utf-8')
assert r.returncode==0 and not r.stderr.strip()
print('VISUAL_AND_AV_COMPLETE',flush=True)
