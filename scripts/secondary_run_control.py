"""Read a managed run or request cooperative cancellation, without app startup."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from motionbridge.secondary_tasks import RunSession
from motionbridge.secondary_cache import read_json
parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--cancel',action='store_true')
args=parser.parse_args();root=args.run_dir.resolve()
if not root.is_relative_to(ROOT/'output/secondary_motion') or not (root/'run.json').is_file():raise ValueError('只接受本副本内的已管理运行')
if args.cancel:RunSession.cancel(root)
state=read_json(root/'run.json')
print(json.dumps({'run_id':state['run_id'],'status':state['status'],'cancel_requested':args.cancel,'stages':[{k:v for k,v in row.items() if k in ('name','status','output','last_access','log','child_pid','child_finished')} for row in state['stages'].values()]},ensure_ascii=False,indent=2))
