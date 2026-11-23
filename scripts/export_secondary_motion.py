"""Managed, cancellable format-export entry point; no physics recomputation."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
from motionbridge.secondary_export_jobs import export_package
parser=argparse.ArgumentParser(description='独立导出缓存/骨骼/衣物，格式通过不代表物理通过')
parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--formats',default='blender,skeleton,rigid,alembic');parser.add_argument('--experimental',action='store_true')
parser.add_argument('--resume',action='store_true');parser.add_argument('--no-cache',action='store_true');parser.add_argument('--take',type=int);parser.add_argument('--render-check',action='store_true')
parser.add_argument('--adopt',action='append',default=[],help='format=already-validated-directory; requires matching source SHA')
args=parser.parse_args();adopt={}
for item in args.adopt:
 name,path=item.split('=',1);adopt[name]=Path(path)
result=export_package(ROOT,args.scene,args.input_dir,args.output_dir,formats=tuple(args.formats.split(',')),experimental=args.experimental,resume=args.resume,use_cache=not args.no_cache,take=args.take,render_check=args.render_check,adopt=adopt)
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
