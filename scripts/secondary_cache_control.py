"""List, checksum-verify, or restore cache snapshots; no deletion operations."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from motionbridge.secondary_cache import ArtifactCache
parser=argparse.ArgumentParser();parser.add_argument('--stage');parser.add_argument('--key');parser.add_argument('--verify',action='store_true');parser.add_argument('--restore',type=Path)
args=parser.parse_args();cache=ArtifactCache(ROOT/'data/secondary-cache')
if args.restore:
 if not args.stage or not args.key:raise ValueError('restore requires stage and key')
 target=args.restore.resolve()
 if not target.is_relative_to(ROOT/'output/secondary_motion'):raise ValueError('Restore only to a new project output directory')
 result=cache.restore(args.stage,args.key,target)
 if result is None:raise ValueError('Missing or damaged cache; nothing restored')
 print(json.dumps({'restored':True,'output':str(target),'physics_approval':False},ensure_ascii=False));raise SystemExit(0)
entries=[]
for path in sorted((cache.root/'objects').glob('*/*/manifest.json')):
 record=json.loads(path.read_text(encoding='utf-8'))
 if args.stage and record['stage']!=args.stage:continue
 if args.key and record['key']!=args.key:continue
 entries.append({'stage':record['stage'],'key':record['key'],'files':len(record['files']),'bytes':sum(f['bytes'] for f in record['files'].values()),'checksum_verified':cache.verify(record['stage'],record['key']) is not None if args.verify else None,'meaning':record['meaning'],'provenance':record['provenance']})
print(json.dumps({'entries':entries,'automatic_deletion':False},ensure_ascii=False,indent=2))
