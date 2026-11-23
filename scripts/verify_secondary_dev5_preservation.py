"""Compare all frozen baselines without writing to the preserved project."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];old=Path(r'D:\codex\danap\MotionBridge')
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() if p.is_file() else None
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
baseline=read(Path(r'E:\MotionBridge_Backups\v0.6.9_20261102\source-manifest.json'))
frozen=json.loads(read(ROOT/'validation/backup-audit-history.json')['records']['physics-dev4-before-step5_20261103']['frozen-files.json']['content'])
algorithms=[n for n in baseline['files'] if n.startswith('motionbridge/') and n.endswith('.py') and n not in ('motionbridge/app.py','motionbridge/__init__.py')]
changes={'original_D':[n for n,r in baseline['files'].items() if sha(old/n)!=r['sha256']],
         'original_E_motion':[n for n in algorithms if sha(ROOT/n)!=baseline['files'][n]['sha256']],
         'frozen_dev4_tests_and_physics':[n for n,h in frozen['files'].items() if sha(ROOT/n)!=h]}
case=ROOT/'output/secondary_motion/dev4/rin-dance';meta=read(case/'input.json');profile=read(ROOT/'profiles/secondary_motion/rin.json')
source_ok=sha(Path(meta['source']))==meta['source_sha256'];model_ok=sha(ROOT/profile['model']['path'])==profile['model']['sha256']
report={'passed':not any(changes.values()) and source_ok and model_ok,'original_D_files':len(baseline['files']),
        'original_E_motion_modules':len(algorithms),'dev4_frozen_files':len(frozen['files']),'changes':changes,
        'source_dev3_unchanged':source_ok,'model_unchanged':model_ok,'dev4_test_cases_preserved':frozen['test_cases']}
(ROOT/'validation/secondary-motion-dev5/preservation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
assert report['passed'],report
print(json.dumps(report,indent=2))
