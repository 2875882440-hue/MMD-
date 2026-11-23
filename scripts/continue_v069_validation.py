"""Wait for the one existing geometry experiment, then run new jobs serially."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'research/v069';STATE=DEST/'validation-state.json'
assert not STATE.exists(),'Inspect existing validation driver before starting another'
state={'phase':'waiting_for_support_candidate','date':'2026-10-31'}
def save():STATE.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
save();start=time.monotonic();report_path=DEST/'mesh-proper-crossing-final-report.json'
try:
    while not report_path.exists():
        if time.monotonic()-start>1800:raise TimeoutError('Inspect current geometry experiment; no new jobs submitted')
        time.sleep(3)
    report=json.loads(report_path.read_text(encoding='utf-8'));a=report['before'];b=report['after']
    assert report['triangle_pair_method']=='proper_noncoplanar_crossing_v2'
    keys=['body_samples_over_1mm','new_body_triangle_pairs','new_layer_triangle_pairs','body_depth_sum_m','layer_depth_sum_m']
    assert all(b[k]<=a[k]+(1e-3 if k.endswith('_m') else 0) for k in keys), 'Actual geometry counters regressed'
    assert any(b[k]<a[k]-(1e-3 if k.endswith('_m') else 0) for k in keys), 'No actual geometry improvement: do not publish unchanged fallback'
    assert report['combined_guard']['accepted']
    state.update(phase='serial_character_regression',experiment_changes={k:[a[k],b[k]] for k in keys});save()
    time.sleep(2)
    with (DEST/'characters.log').open('wb') as log:
        process=subprocess.Popen([sys.executable,'-X','utf8','scripts/validate_v069_characters.py'],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        state['driver_pid']=process.pid;save()
        code=process.wait()
        if code:raise RuntimeError('Character regression failed; inspect existing jobs, do not duplicate')
    state.update(phase='characters_complete_needs_visual_and_release_checks');save()
    print('V069_CHARACTER_REGRESSION_COMPLETE',flush=True)
except BaseException as exc:
    state.update(phase='failed',error=repr(exc));save();raise
