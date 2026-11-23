from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]; dest=root/'data/release-v050'
ids=json.loads((dest/'final_jobs.json').read_text())
protected=json.loads((dest/'protected_hashes.json').read_text())
changed=[]
for p,h in protected.items():
    with Path(p).open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=h: changed.append(p)
report={'protected_files':len(protected),'changed':changed,'jobs':{}}
for name,ident in ids.items():
    folder=root/'data/jobs'/ident; output=folder/'outputs'
    job=json.loads((folder/'job.json').read_text())
    check=json.loads((output/('saved_project_audit.json' if name=='character' else 'blender_verification.json')).read_text())
    assert job['status']=='completed' and check['passed']
    report['jobs'][name]={'id':ident,'passed':check['passed'],'frames':check['frames'],'fps':check['fps'],
        'metrics':check.get('all_frames',check.get('maximum_adjacent_world_rotation_change'))}
    if name!='character':
        m=json.loads((output/'motion.json').read_text())
        report['jobs'][name].update(projection=m['projection_refinement'],hands=m['hand_assistance'])
    else:
        r=json.loads((output/'retarget_report.json').read_text())
        report['jobs'][name].update(references=r['reference_landmarks'],target_ik=r['target_endpoint_ik'])
assert not changed
(dest/'release_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
