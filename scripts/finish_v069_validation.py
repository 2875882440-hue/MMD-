"""Post-validation only: wait for existing new jobs, never submit duplicates."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'research/v069';STATE=DEST/'post-validation-state.json'
assert not STATE.exists(),'Inspect existing post-validation before restarting'
state={'phase':'waiting_for_existing_characters','completed':[]}
def save():STATE.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
save();started=time.monotonic()
try:
    while True:
        progress=json.loads((DEST/'validation-state.json').read_text(encoding='utf-8'))
        if progress['phase']=='failed':raise RuntimeError('Validation failed; no duplicate jobs or renders started')
        manifest=DEST/'character-jobs.json'
        records=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else []
        if len(records)==2 and all(r['status']=='completed' and len(r.get('downloads',{}))==4 for r in records):break
        if time.monotonic()-started>15600:raise TimeoutError('Inspect existing jobs; no duplicates started')
        time.sleep(3)
    for folder in ['source-dance','source-tennis']:(DEST/folder).mkdir(exist_ok=True)
    blender='D:/STEAM/steamapps/common/Blender/blender.exe'
    common=[blender,'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','23','--python']
    camera='research/global-motion-regression/nahida-world/Nahida_MotionBridge_Dance.blend'
    steps=[
        ('source-dance-fbx',common+['scripts/blender_export.py','--','data/jobs/7c2fb9899dc64d6ea83baa33548997cb/outputs/motion.json','research/v069/source-dance'],600),
        ('source-tennis-fbx',common+['scripts/blender_export.py','--','data/jobs/a47fc9bd9b164d7b8867b5d77ae3b590/outputs/motion.json','research/v069/source-tennis'],600),
        ('dance-rig',common+['scripts/audit_character_motion_unchanged.py','--','research/v068/nahida/Nahida_MotionBridge_Dance.blend','research/v069/nahida/Nahida_MotionBridge_Dance.blend','research/v069/rig-unchanged.json'],180),
        ('tennis-rig',common+['scripts/audit_character_motion_unchanged.py','--','research/v068/tennis/Nahida_MotionBridge_Dance.blend','research/v069/tennis/Nahida_MotionBridge_Dance.blend','research/v069/tennis-rig-unchanged.json'],180),
        ('dance-render',common+['scripts/render_retarget_review.py','--','research/v069/nahida','preview','--frame-step','1','--threads','4','--reference-camera',camera],1800),
        ('tennis-render',common+['scripts/render_retarget_review.py','--','research/v069/tennis','samples','--sample-times','0,3,6,9','--threads','2','--diagnostic-stage'],240),
        ('comparison',[sys.executable,'-X','utf8','scripts/compose_v069_comparison.py'],300),
        ('verification',[sys.executable,'-X','utf8','scripts/verify_v069_release.py'],180)]
    env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    for name,command,timeout in steps:
        state['phase']=name;save();print('START',name,flush=True);start=time.monotonic()
        with (DEST/(name+'.log')).open('wb') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            state['child_pid']=process.pid;save()
            try:code=process.wait(timeout=timeout)
            except BaseException:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
            if code:raise RuntimeError(name+' failed; inspect log before retry')
        state['completed'].append({'stage':name,'seconds':round(time.monotonic()-start,2)})
        state.pop('child_pid',None);save();print('DONE',name,flush=True)
    state['phase']='ready_for_visual_review';save();print('V069_ARTIFACTS_READY_FOR_VISUAL_REVIEW',flush=True)
except BaseException as exc:
    state.update(error=repr(exc),failed_phase=state['phase'],phase='failed');save();raise
