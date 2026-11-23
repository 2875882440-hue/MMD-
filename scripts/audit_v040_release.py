from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
dest=root/'data/release-v040'
ids=json.loads((dest/'final_jobs.json').read_text())
def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
baseline=json.loads((dest/'preexisting_hashes.json').read_text())
changed=[p for p,h in baseline.items() if digest(p)!=h]
result={'protected_count':len(baseline),'changed':changed,'jobs':{}}
for name,ident in ids.items():
    folder=root/'data/jobs'/ident
    job=json.loads((folder/'job.json').read_text(encoding='utf-8'))
    assert job['status']=='completed'
    if name=='character':
        report=json.loads((folder/'outputs/saved_project_audit.json').read_text(encoding='utf-8'))
        result['download_matches']=digest(folder/'outputs/Nahida_MotionBridge_Dance.blend')==digest(root/'output/playwright/v040-nahida.blend')
        result['target_ik']=json.loads((folder/'outputs/retarget_report.json').read_text(encoding='utf-8'))['target_endpoint_ik']
    else:
        report=json.loads((folder/'outputs/blender_verification.json').read_text(encoding='utf-8'))
    result['jobs'][name]={'id':ident,'passed':report['passed'],'frames':report['frames'],'fps':report['fps'],
                        'metrics':report.get('all_frames',report.get('maximum_adjacent_world_rotation_change'))}
    assert report['passed']
assert not changed and result['download_matches']
(dest/'release_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
