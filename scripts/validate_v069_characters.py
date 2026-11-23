"""Serial HTTP regression, restart-safe without duplicating submitted jobs."""
from pathlib import Path
import hashlib,json,time,urllib.request,urllib.parse,shutil
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'research/v069';RESULT=DEST/'character-jobs.json'
BASE='http://127.0.0.1:8765'
def request(path,data=None):
    with urllib.request.urlopen(BASE+path,data=data,timeout=60) as response:return response.read()
def save():RESULT.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
assert json.loads(request('/api/health'))['version']=='0.6.9'
records=json.loads(RESULT.read_text(encoding='utf-8')) if RESULT.exists() else []
targets=[('dance','7c2fb9899dc64d6ea83baa33548997cb','nahida'),('tennis','a47fc9bd9b164d7b8867b5d77ae3b590','tennis')]
assert all(r.get('version')=='0.6.9' for r in records)
for name,source,folder in targets:
    record=next((r for r in records if r['name']==name),None)
    if record is None:
        assert not any(j['status'] in ['queued','processing'] for j in json.loads(request('/api/jobs'))['jobs']), 'Inspect active job before submitting'
        job=json.loads(request('/api/jobs/'+source+'/retarget',urllib.parse.urlencode({'character':'nahida','surface_clearance':'true'}).encode()))
        record={'name':name,'version':'0.6.9','source_job':source,'job_id':job['id'],'status':job['status']};records.append(record);save()
        print('START',name,job['id'],flush=True)
    assert record['source_job']==source
    jid=record['job_id'];job=json.loads(request('/api/jobs/'+jid));deadline=time.monotonic()+7800
    while job['status'] in ['queued','processing']:
        if time.monotonic()>deadline:raise TimeoutError('Inspect existing job; do not submit a duplicate')
        time.sleep(3);job=json.loads(request('/api/jobs/'+jid))
    record['status']=job['status'];record['error']=job.get('error');save()
    if job['status']!='completed':raise RuntimeError('Existing job failed; preserve it and inspect: '+str(job.get('error')))
    output=ROOT/'data/jobs'/jid/'outputs';dst=DEST/folder;dst.mkdir(exist_ok=True)
    audit=json.loads((output/'saved_project_audit.json').read_text(encoding='utf-8'))
    assert audit['passed'] and audit['surface_clearance']['passed']
    assert all(o['triangle_consistent'] for o in audit['surface_clearance']['objects'])
    record['downloads']={}
    for filename in ['Nahida_MotionBridge_Dance.blend','retarget_report.json','saved_project_audit.json','character_preview.png']:
        remote=request('/api/jobs/'+jid+'/files/'+filename);digest=hashlib.sha256(remote).hexdigest()
        assert digest==hashlib.sha256((output/filename).read_bytes()).hexdigest()
        record['downloads'][filename]=digest;shutil.copy2(output/filename,dst/filename)
    reference=output/'surface_solver_reference.npz'
    if reference.exists():
        shutil.copy2(reference,dst/reference.name)
        record['solver_reference_sha256']=hashlib.sha256(reference.read_bytes()).hexdigest()
    record['surface_clearance']=audit['surface_clearance'];save();print('DONE',name,jid,flush=True)
print('V069_CHARACTERS_COMPLETE',flush=True)
