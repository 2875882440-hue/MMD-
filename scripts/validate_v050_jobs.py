"""Serial production API regression; reuse recorded jobs after interruptions."""
from pathlib import Path
import json, time, hashlib
import httpx
root=Path(__file__).resolve().parents[1]
dest=root/'data/release-v050'; dest.mkdir(exist_ok=True)
manifest=dest/'final_jobs.json'
ids=json.loads(manifest.read_text()) if manifest.exists() else {}
protected=dest/'protected_hashes.json'
if not protected.exists():
    files=[]
    for ident in ['b6f9d511e8214257b7df4fab6df2196f','288450f7d88848609abd9d7c52351f8e','57317bd57b2d41f7963825d970a35ba6']:
        files.extend(p for p in (root/'data/jobs'/ident).rglob('*') if p.is_file())
    old=json.loads((root/'data/release-v040/preexisting_hashes.json').read_text())
    for p in files:
        with p.open('rb') as f: old[str(p)]=hashlib.file_digest(f,'sha256').hexdigest()
    protected.write_text(json.dumps(old),encoding='utf-8')
with httpx.Client(base_url='http://127.0.0.1:8765',trust_env=False,timeout=30) as client:
    assert client.get('/api/health').json()['version']=='0.5.0'
    for name,source in [('dance','b6f9d511e8214257b7df4fab6df2196f'),('tennis','288450f7d88848609abd9d7c52351f8e'),('character',None)]:
        if name not in ids:
            suffix=f'/api/jobs/{source}/reprocess' if source else f'/api/jobs/{ids["dance"]}/retarget'
            r=client.post(suffix); r.raise_for_status()
            ids[name]=r.json()['id']; manifest.write_text(json.dumps(ids),encoding='utf-8')
        deadline=time.monotonic()+1200
        last=None
        while time.monotonic()<deadline:
            r=client.get(f'/api/jobs/{ids[name]}'); r.raise_for_status(); job=r.json()
            state=(job['status'],job['stage'])
            if state!=last:
                print(name,ids[name],state,job.get('error'),flush=True); last=state
            if job['status'] in ['completed','failed','cancelled']:
                (dest/(name+'_job.json')).write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
                assert job['status']=='completed', job.get('error')
                break
            time.sleep(3)
        else: raise TimeoutError(name)
print('ALL_FINAL_JOBS_COMPLETED',ids,flush=True)
