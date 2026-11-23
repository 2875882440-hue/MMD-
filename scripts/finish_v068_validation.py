"""Finish this validation run serially after its existing HTTP jobs complete.

No new jobs, source inference or recurring task. Never rerun an existing state.
"""
from pathlib import Path
import json, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'research/v068'
STATE=DEST/'post-validation-state.json'
assert not STATE.exists(),'Read existing state and resume only the missing stage explicitly'
state={'phase':'waiting_for_existing_characters','completed':[]};start=time.monotonic()
def save():
    STATE.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
save()
try:
    while True:
        records=json.loads((DEST/'character-jobs.json').read_text(encoding='utf-8'))
        if any(r['status']=='failed' for r in records):raise RuntimeError('Character regression failed; inspect before continuing')
        if len(records)==2 and all(r['status']=='completed' and r.get('surface_clearance') and len(r.get('downloads',{}))==4 for r in records):break
        if time.monotonic()-start>4000:raise TimeoutError('Existing character jobs are not ready; no duplicates started')
        time.sleep(3)
    py=str(ROOT/'.venv/Scripts/python.exe')
    blender='D:/STEAM/steamapps/common/Blender/blender.exe'
    common=[blender,'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','23','--python']
    camera='research/global-motion-regression/nahida-world/Nahida_MotionBridge_Dance.blend'
    steps=[
        ('report-notes',[py,'-X','utf8','scripts/normalize_v068_report_notes.py'],60),
        ('dance-rig',common+['scripts/audit_character_motion_unchanged.py','--','research/v067/nahida/Nahida_MotionBridge_Dance.blend','research/v068/nahida/Nahida_MotionBridge_Dance.blend','research/v068/rig-unchanged.json'],180),
        ('tennis-rig',common+['scripts/audit_character_motion_unchanged.py','--','research/v067/tennis/Nahida_MotionBridge_Dance.blend','research/v068/tennis/Nahida_MotionBridge_Dance.blend','research/v068/tennis-rig-unchanged.json'],180),
        ('dance-render',common+['scripts/render_retarget_review.py','--','research/v068/nahida','preview','--frame-step','1','--threads','4','--reference-camera',camera],1800),
        ('tennis-render',common+['scripts/render_retarget_review.py','--','research/v068/tennis','samples','--sample-times','0,3,6,9','--threads','2','--diagnostic-stage'],240),
        ('comparison',[py,'-X','utf8','scripts/compose_v068_comparison.py'],300),
        ('verification',[py,'-X','utf8','scripts/verify_v068_release.py'],120)]
    env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    for name,command,timeout in steps:
        state['phase']=name;save();print('START',name,flush=True);step_start=time.monotonic()
        with (DEST/(name+'.log')).open('wb') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            state['child_pid']=process.pid;save()
            try:code=process.wait(timeout=timeout)
            except BaseException:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
            if code:raise RuntimeError(name+' failed: '+str(code))
        state['completed'].append({'stage':name,'seconds':round(time.monotonic()-step_start,2)})
        state.pop('child_pid',None);save();print('DONE',name,flush=True)
    state['phase']='ready_for_visual_review';save();print('ARTIFACTS_READY_FOR_VISUAL_REVIEW',flush=True)
except BaseException as exc:
    state['error']=repr(exc);state['failed_phase']=state['phase'];state['phase']='failed';save();raise
