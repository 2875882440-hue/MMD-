"""Verify the actual browser-created job and bytes served by the local app."""
from pathlib import Path
import json,hashlib,urllib.request
ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:8765'
ID='36443e983a494de2b5dc19bc8c3aaa9b'
def get(path):
    with urllib.request.urlopen(BASE+path,timeout=60) as response:return response.read()
job=json.loads(get('/api/jobs/'+ID))
assert job['status']=='completed',job['status']
assert job['software_version']=='0.6.6' and job['settings']['surface_clearance'] is True
assert job['source_job_id']=='7c2fb9899dc64d6ea83baa33548997cb'
out=ROOT/'data/jobs'/ID/'outputs';checks={}
for filename in ['Nahida_MotionBridge_Dance.blend','character_preview.png','retarget_report.json','saved_project_audit.json']:
    remote=get('/api/jobs/'+ID+'/files/'+filename)
    actual=hashlib.sha256(remote).hexdigest();expected=hashlib.sha256((out/filename).read_bytes()).hexdigest()
    assert actual==expected
    checks[filename]={'bytes':len(remote),'sha256':actual,'download_matches':True}
audit=json.loads((out/'saved_project_audit.json').read_text(encoding='utf-8'))
assert audit['passed'] and audit['original_unchanged'] and audit['surface_clearance']['passed']
reference=json.loads((ROOT/'research/v066/nahida-final/retarget_report.json').read_text(encoding='utf-8'))
production=json.loads((out/'retarget_report.json').read_text(encoding='utf-8'))
same={k:production[k]==reference[k] for k in ['surface_clearance','target_accessory_collision','expected_root_positions']}
assert all(same.values())
assert audit['surface_clearance']==json.loads((ROOT/'research/v066/nahida-final/saved_project_audit.json').read_text(encoding='utf-8'))['surface_clearance']
result={'job_id':ID,'version':job['software_version'],'settings':job['settings'],'download_checks':checks,
        'research_production_same':same,'audit_passed':True,'source_unchanged':audit['original_unchanged']}
(ROOT/'research/v066/production-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)
