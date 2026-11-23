"""Correct legacy explanatory text in the first validation run, never geometry.

The first baking process loaded the module before its note was clarified.
Preserve its original report and download the corrected report from the app.
Future exports already write this explanation in blender_surfaces.py.
"""
from pathlib import Path
import json, hashlib, urllib.request
ROOT=Path(__file__).resolve().parents[1];dest=ROOT/'research/v068'
path=dest/'character-jobs.json';records=json.loads(path.read_text(encoding='utf-8'))
assert len(records)==2 and all(r['status']=='completed' for r in records)
note='Final edge metrics include coherent contact. Proposal/backoff/extra-shift fields describe the v067 baseline pass only; see baseline_topology and coherent_contact for their separate budgets and decisions.'
for r in records:
 output=ROOT/'data/jobs'/r['job_id']/'outputs'/'retarget_report.json'
 folder=dest/('nahida' if r['name']=='dance' else 'tennis')
 original=output.read_bytes();report=json.loads(original);changed=False
 for obj in report['surface_clearance']['objects']:
  if obj['topology']['note']!=note:obj['topology']['note']=note;changed=True
 if not changed:continue
 backup=folder/'retarget_report_before_note_correction.json'
 assert not backup.exists(),'Inspect previous correction before retry'
 backup.write_bytes(original)
 output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 with urllib.request.urlopen('http://127.0.0.1:8765/api/jobs/'+r['job_id']+'/files/retarget_report.json',timeout=30) as response:download=response.read()
 assert download==output.read_bytes()
 (folder/'retarget_report.json').write_bytes(download)
 digest=hashlib.sha256(download).hexdigest()
 r['metadata_note_correction']={'previous_sha256':hashlib.sha256(original).hexdigest(),'current_sha256':digest,'geometry_changed':False}
 r['downloads']['retarget_report.json']=digest
path.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print('REPORT_NOTES_VERIFIED')
