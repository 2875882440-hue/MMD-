"""Actual application regression using byte-verified unchanged inference caches."""
from pathlib import Path
import os,sys,json,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
dest=ROOT/'research/v062/queue';dest.mkdir(exist_ok=True)
if (dest/'results.json').exists():raise RuntimeError('Do not duplicate completed or interrupted validation; inspect saved jobs first')
os.environ['MOTIONBRIDGE_DATA']=str(dest/'jobs')
from motionbridge import app as server
results=[]
for name,old_id in [('dance','78d52a9ac4204f99bdf61469cb6495ee'),('tennis','bd1897dc5789411286ca6089679748a0')]:
    old=ROOT/'data/jobs'/old_id;record=json.loads((old/'job.json').read_text(encoding='utf-8'))
    job=server.new_job('v0.6.2 '+name+'.mp4',record['settings']);jid=job['id']
    target=server.DATA/jid;(target/'input').mkdir()
    shutil.copy2(old/'input/source.mp4',target/'input/source.mp4')
    shutil.copytree(old/'outputs/gvhmr',target/'outputs/gvhmr')
    server.update(jid,validation_source_job=old_id,validation_cache_reuse=True)
    results.append({'name':name,'job_id':jid,'status':'processing','reused_inference_from':old_id})
    (dest/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('START',jid,flush=True);server.run_job(jid,target/'input/source.mp4')
    done=server.get_job(jid);results[-1].update(status=done['status'],meta=done['meta'],error=done['error'])
    (dest/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    if done['status']!='completed':raise RuntimeError(done['error'])
    audit=json.loads((target/'outputs/blender_verification.json').read_text())
    assert audit['passed']
    print('DONE',name,flush=True)
