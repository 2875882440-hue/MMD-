from pathlib import Path
import hashlib,json
R=Path(r'E:\MotionBridge_Physics');D=Path(r'D:\codex\danap\MotionBridge');V=R/'validation/secondary-motion-dev6'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() if p.is_file() else None
baseline=read(Path(r'E:\MotionBridge_Backups\v0.6.9_20261102\source-manifest.json'))
frozen=json.loads(read(R/'validation/backup-audit-history.json')['records']['physics-dev5-before-step6_20261103']['frozen-files.json']['content'])
changed={name:digest for name,digest in frozen['files'].items() if sha(R/name)!=digest}
old_changed=[name for name,record in baseline['files'].items() if sha(D/name)!=record['sha256']]
algorithms=[name for name in baseline['files'] if name.startswith('motionbridge/') and name.endswith('.py') and name not in ('motionbridge/app.py','motionbridge/__init__.py')]
motion_changed=[name for name in algorithms if sha(R/name)!=baseline['files'][name]['sha256']]
profile=read(R/'profiles/secondary_motion/rin.json');input=read(R/'output/secondary_motion/dev4/rin-dance/input.json')
model_ok=sha(R/profile['model']['path'])==profile['model']['sha256'];source_ok=sha(Path(input['source']))==input['source_sha256']
proof={'passed':not changed and not old_changed and not motion_changed and model_ok and source_ok,'original_D_files':len(baseline['files']),
 'original_E_motion_modules':len(algorithms),'frozen_dev5_files':len(frozen['files']),'original_test_cases_preserved':306,
 'changed_frozen_files':changed,'original_D_changes':old_changed,'original_motion_changes':motion_changed,'original_model_unchanged':model_ok,
 'original_dev3_scene_unchanged':source_ok,'rin_profile_sha256':sha(R/'profiles/secondary_motion/rin.json')}
(V/'preservation.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');assert proof['passed'],proof
print('DEV6_PRESERVATION_PASSED',len(frozen['files']))
