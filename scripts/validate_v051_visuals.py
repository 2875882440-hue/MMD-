from pathlib import Path
import sys,json,subprocess,hashlib
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from motionbridge.retarget import run_script
from diagnose_jitter import metrics
dest=root/'data/release-v051'; ids=json.loads((dest/'final_jobs.json').read_text())
blender=r'D:\STEAM\steamapps\common\Blender\blender.exe'
report={}
for version,ident in [('v040','57317bd57b2d41f7963825d970a35ba6'),('v050','340f020f5cb8476eb1ea00b193aaf021'),('v051',ids['character'])]:
    folder=root/'data/jobs'/ident/'outputs'; tracks=dest/f'{version}-target-tracks.npz'
    if not tracks.exists(): run_script(blender,root/'scripts/extract_target_tracks.py',[folder,tracks],dest/f'{version}-target-tracks.log',lambda:False)
    with np.load(tracks) as a:
        report[version]=metrics(a['world'],a['positions'],float(a['fps']),a['names'].tolist())
(dest/'target_jitter_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('TARGET_JITTER',json.dumps({v:{n:m[n] for n in ['左腕','左ひじ','左手首','右手首']} for v,m in report.items()},ensure_ascii=False),flush=True)
checks={}
for name in ['dance','tennis']:
    folder=root/'data/jobs'/ids[name]/'outputs'
    audit=json.loads((folder/'blender_verification.json').read_text()); assert audit['passed']; checks[name]=audit['all_frames']
    m=json.loads((folder/'motion.json').read_text())
    (dest/f'{name}-final-jitter.json').write_text(json.dumps(metrics(np.array(m['world_rotations']),np.array([f['positions'] for f in m['frames']]),m['fps'],m['joint_names']),indent=2))
    review=dest/f'{name}-review'; review.mkdir(exist_ok=True)
    (review/'assessment.json').write_text(json.dumps({'outputs':str(folder),'source_filename':str(folder.parent/'input/source.mp4'),'sample_times':[1,4,7,10,13,16,19,23,28] if name=='dance' else [1,4,7],'source_duration':m['num_frames']/m['fps']}))
    for phase in ['probe','samples']:
        run_script(blender,root/'scripts/render_dance_review.py',[review,phase],review/f'{phase}.log',lambda:False)
    subprocess.run([sys.executable,str(root/'scripts/compose_dance_review.py'),str(review),str(folder.parent/'input/source.mp4'),'--sheets-only'],check=True)
target=root/'data/jobs'/ids['character']/'outputs'
audit=json.loads((target/'saved_project_audit.json').read_text());assert audit['passed'];checks['character']=audit['maximum_adjacent_world_rotation_change']
(dest/'final_audits.json').write_text(json.dumps(checks,indent=2))
run_script(blender,root/'scripts/render_retarget_review.py',[target,'samples'],dest/'character-samples.log',lambda:False)
subprocess.run([sys.executable,str(root/'scripts/compose_retarget_review.py'),str(target),'--source',str(root/'data/jobs'/ids['dance']/'input/source.mp4'),'--sheets-only'],check=True)
subprocess.run([sys.executable,str(root/'scripts/resume_v050_preview.py'),'--release-dir',str(dest),'--frame-step','1'],check=True)
print('V051_VISUALS_COMPLETE',flush=True)
