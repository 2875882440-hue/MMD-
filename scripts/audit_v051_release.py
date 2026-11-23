from pathlib import Path
import hashlib,json
import httpx
root=Path(__file__).resolve().parents[1]; dest=root/'data/release-v051'
ids=json.loads((dest/'final_jobs.json').read_text())
protected=json.loads((root/'data/release-v050/protected_hashes.json').read_text())
changed=[]
backups={}
for p,digest in protected.items():
    with Path(p).open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=digest:
            changed.append(p)
            backup=Path(p+'1')
            if backup.exists():
                with backup.open('rb') as b:
                    backups[p]={'path':str(backup),'matches_recorded_original':hashlib.file_digest(b,'sha256').hexdigest()==digest,
                                'current_modified_time':Path(p).stat().st_mtime}
old=json.loads((root/'data/release-v050/browser-download.json').read_text())
with (root/'data/jobs'/old['job']/'outputs/Nahida_MotionBridge_Dance.blend').open('rb') as f:
    old_match=hashlib.file_digest(f,'sha256').hexdigest()==old['sha256']
with httpx.Client(base_url='http://127.0.0.1:8765',trust_env=False,timeout=60) as client:
    health=client.get('/api/health').json();assert health['version']=='0.5.1'
    digest=hashlib.sha256(); size=0
    with client.stream('GET',f'/api/jobs/{ids["character"]}/files/Nahida_MotionBridge_Dance.blend') as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():digest.update(chunk);size+=len(chunk)
with (root/'data/jobs'/ids['character']/'outputs/Nahida_MotionBridge_Dance.blend').open('rb') as f:
    download_match=hashlib.file_digest(f,'sha256').hexdigest()==digest.hexdigest()
result={'protected_files':len(protected),'changed':changed,'v050_original_blend_unchanged':old_match,
        'changed_file_backups':backups,
        'download_matches':download_match,'download_bytes':size,'download_sha256':digest.hexdigest(),'health':health}
(dest/'release_audit.json').write_text(json.dumps(result,indent=2))
assert old_match and download_match
assert all(backups.get(p,{}).get('matches_recorded_original') for p in changed), 'Changed original without matching retained backup'
print('DOWNLOAD_AND_PRESERVATION_CHECKED',len(protected),'changed_with_original_backup',len(changed),size)
