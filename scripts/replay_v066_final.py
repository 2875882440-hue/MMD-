"""Rerun isolated v066 jobs with the final confidence transition, then character."""
from pathlib import Path
import os,sys,json,shutil,threading
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
dest=ROOT/'research/v066/queue'
os.environ['MOTIONBRIDGE_DATA']=str(dest/'jobs')
from motionbridge import app as server
from motionbridge.retarget import export_character,character_profiles
results=json.loads((dest/'results.json').read_text())
trial=ROOT/'research/v066/initial-alignment-trial';trial.mkdir(exist_ok=True)
for item in results:
    jid=item['job_id'];jobdir=server.DATA/jid
    saved=trial/item['name']
    if saved.exists():raise RuntimeError('Final replay already attempted; inspect first')
    saved.mkdir()
    for filename in ['motion.json','quality.json']:
        src=jobdir/'outputs'/filename
        if src.exists():shutil.copy2(src,saved/filename)
    server.cancel_events[jid]=threading.Event()
    print('START',item['name'],flush=True)
    server.run_job(jid,jobdir/'input/source.mp4')
    done=server.get_job(jid);item.update(status=done['status'],meta=done['meta'],error=done['error'])
    (dest/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    if done['status']!='completed':raise RuntimeError(done['error'])
    assert json.loads((jobdir/'outputs/blender_verification.json').read_text())['passed']
    print('DONE',item['name'],flush=True)
out=ROOT/'research/v066/nahida-final';out.mkdir(exist_ok=False)
blender=json.loads((ROOT/'config.json').read_text())['blender_path']
export_character(ROOT,blender,character_profiles(ROOT,blender)[0],
    server.DATA/results[0]['job_id']/'outputs/motion.json',out,lambda:False,lambda *a:print(a,flush=True))
print('TARGET_DONE',flush=True)
