"""Small source-only backup and final integrity check; never copy user media."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(r'E:\MotionBridge_Physics');VAL=ROOT/'validation/secondary-motion-dev5'
backup=Path(r'E:\MotionBridge_Backups\physics-dev5_20261103');backup.mkdir(exist_ok=True)
archive=backup/'source-config.zip'
assert not archive.exists(),'Preserve an existing snapshot rather than replacing it'
extensions={'.py','.js','.css','.html','.md','.json','.txt','.bat','.ps1','.ini','.yaml','.yml','.toml'}
dirs={'motionbridge','scripts','static','templates','tests','docs','profiles'}
files=[]
for p in ROOT.iterdir():
 if p.is_file() and (p.suffix in extensions or p.name=='.gitignore'):files.append(p)
 elif p.is_dir() and p.name in dirs:
  files.extend(q for q in p.rglob('*') if q.is_file() and q.suffix in extensions and '__pycache__' not in q.parts)
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
records={p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)}
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for name,r in records.items():assert hashlib.sha256(z.read(name)).hexdigest()==r['sha256']
record={'version':'0.7.0-dev.5','source_only':True,'archive':str(archive),'archive_bytes':archive.stat().st_size,
        'archive_sha256':sha(archive),'files':records,'zip_crc_and_member_sha_verified':True}
(backup/'source-manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
latest=json.loads((ROOT/'output/secondary_motion/dev5/managed-export/latest.json').read_text(encoding='utf-8'))
delivery=Path(latest['delivery']);manifest=json.loads((delivery/'manifest.json').read_text(encoding='utf-8'))
for n,r in manifest['files'].items():
 p=delivery/n;assert p.resolve().is_relative_to(delivery) and p.is_file()
 assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],n
verification=json.loads((VAL/'verification.json').read_text(encoding='utf-8'))
verification['source_backup']={k:v for k,v in record.items() if k!='files'}
verification['source_backup']['files']=len(records)
verification['delivery_manifest_files_verified']=len(manifest['files'])
verification['delivery_bytes']=sum(r['bytes'] for r in manifest['files'].values())
(VAL/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print('BACKUP_AND_DELIVERY_MANIFEST_PASSED',len(records),len(manifest['files']),archive.stat().st_size,flush=True)
