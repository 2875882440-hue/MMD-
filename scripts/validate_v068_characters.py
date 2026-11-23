"""Serial real HTTP character regression; never duplicate an existing run."""
from pathlib import Path
import json,time,urllib.request,urllib.parse,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'research/v068';RESULT=DEST/'character-jobs.json'
BASE='http://127.0.0.1:8765'
def request(path,data=None):
    with urllib.request.urlopen(BASE+path,data=data,timeout=60) as response:return response.read()
assert not RESULT.exists(),'Inspect existing validation before retry'
assert json.loads(request('/api/health'))['version']=='0.6.8'
assert not any(j['status'] in ['queued','processing'] for j in json.loads(request('/api/jobs'))['jobs'])
records=[]
for name,source,folder in [('dance','7c2fb9899dc64d6ea83baa33548997cb','nahida'),('tennis','a47fc9bd9b164d7b8867b5d77ae3b590','tennis')]:
    job=json.loads(request('/api/jobs/'+source+'/retarget',urllib.parse.urlencode({'character':'nahida','surface_clearance':'true'}).encode()))
    jid=job['id'];record={'name':name,'job_id':jid,'source_job':source,'status':job['status']};records.append(record)
    RESULT.write_text(json.dumps(records,indent=2),encoding='utf-8');print('START',name,jid,flush=True)
    deadline=time.monotonic()+2400
    while job['status'] in ['queued','processing']:
        if time.monotonic()>deadline:raise RuntimeError('Polling deadline exceeded; inspect job before restarting')
        time.sleep(2);job=json.loads(request('/api/jobs/'+jid))
    record['status']=job['status'];RESULT.write_text(json.dumps(records,indent=2),encoding='utf-8')
    if job['status']!='completed':raise RuntimeError(job.get('error'))
    output=ROOT/'data/jobs'/jid/'outputs';dst=DEST/folder;dst.mkdir(exist_ok=False)
    audit=json.loads((output/'saved_project_audit.json').read_text(encoding='utf-8'))
    assert audit['passed'] and audit['surface_clearance']['passed']
    record['downloads']={}
    for filename in ['Nahida_MotionBridge_Dance.blend','retarget_report.json','saved_project_audit.json','character_preview.png']:
        remote=request('/api/jobs/'+jid+'/files/'+filename);digest=hashlib.sha256(remote).hexdigest()
        assert digest==hashlib.sha256((output/filename).read_bytes()).hexdigest()
        record['downloads'][filename]=digest;shutil.copy2(output/filename,dst/filename)
    record['surface_clearance']=audit['surface_clearance']
    RESULT.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8');print('DONE',name,flush=True)
