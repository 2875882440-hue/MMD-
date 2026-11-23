"""Reprocess final regression clips through the production API, retaining every old task."""
from pathlib import Path
import json,time
import httpx
root=Path(__file__).resolve().parents[1]; dest=root/'data/release-v051'
manifest=dest/'final_jobs.json'
ids=json.loads(manifest.read_text()) if manifest.exists() else {}
with httpx.Client(base_url='http://127.0.0.1:8765',trust_env=False,timeout=30) as client:
    assert client.get('/api/health').json()['version']=='0.5.1'
    for name,source in [('dance','47b709e3e44540c5aa96c76f664c244c'),('tennis','b6b39f9009e442de882865a2c6310872'),('character',None)]:
        if name not in ids:
            url=f'/api/jobs/{source}/reprocess' if source else f'/api/jobs/{ids["dance"]}/retarget'
            response=client.post(url);response.raise_for_status();ids[name]=response.json()['id']
            manifest.write_text(json.dumps(ids))
        deadline=time.monotonic()+1500;last=None
        while time.monotonic()<deadline:
            response=client.get(f'/api/jobs/{ids[name]}');response.raise_for_status();job=response.json()
            state=(job['status'],job['stage'])
            if state!=last: print(name,ids[name],state,job.get('error'),flush=True);last=state
            if job['status'] in ['completed','failed','cancelled']:
                (dest/f'{name}_job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
                assert job['status']=='completed',job.get('error')
                break
            time.sleep(3)
        else: raise TimeoutError(name)
print('ALL_FINAL_JOBS_COMPLETED',ids,flush=True)
