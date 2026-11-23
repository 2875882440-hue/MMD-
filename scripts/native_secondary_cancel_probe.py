"""A real Blender scene read loop for process cancellation, not simulation."""
from pathlib import Path
import argparse,json,sys,time,hashlib
import bpy
parser=argparse.ArgumentParser();parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(__file__).resolve().parents[1];out=args.output_dir.resolve()
if not out.is_relative_to(root/'output/secondary_motion') or out.exists():raise ValueError('New owned probe directory required')
bpy.ops.wm.open_mainfile(filepath=str(args.scene),load_ui=False,use_scripts=False);out.mkdir(parents=True)
(out/'native-started.json').write_text(json.dumps({'source_sha256':hashlib.file_digest(args.scene.open('rb'),'sha256').hexdigest(),'pid':__import__('os').getpid()}),encoding='utf-8')
for index in range(50):bpy.context.scene.frame_set(index+1);time.sleep(.1)
(out/'complete.json').write_text(json.dumps({'passed':True}),encoding='utf-8')
