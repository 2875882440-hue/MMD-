"""Verify actual browser downloads and unpack a disposable QA copy on D."""
from pathlib import Path
import hashlib,json,zipfile
R=Path(r'E:\MotionBridge_Physics');base=Path(r'D:\codex\danap\output\playwright');records=[]
for filename in ['dev6-browser-package.zip','dev6-cloth-package.zip']:
 with zipfile.ZipFile(base/filename) as archive:
  assert archive.testzip() is None
  manifest=json.loads(archive.read('manifest.json'))
  assert manifest['format_validation_passed'] is True and manifest['physics_quality_approved'] is False
  for name,record in manifest['files'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==record['sha256']
  records.append({'file':filename,'entries':len(archive.namelist()),'passed':True,'crc_and_manifest_sha_verified':True,'physics_quality_not_upgraded':True})
  if filename=='dev6-cloth-package.zip':
   dest=base/'dev6-unpacked-cloth'
   if dest.exists():raise ValueError('Keep the prior QA copy')
   for member in archive.infolist():assert (dest/member.filename).resolve().is_relative_to(dest.resolve())
   archive.extractall(dest)
(R/'validation/secondary-motion-dev6/browser-download-verification.json').write_text(json.dumps({'passed':True,'archives':records},indent=2),encoding='utf-8')
print('BROWSER_DOWNLOADS_PASSED',records)
