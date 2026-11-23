"""Extra ground-reference stills; never save or change the verified pair."""
from pathlib import Path
import argparse,hashlib,json
import bpy
parser=argparse.ArgumentParser();parser.add_argument('--pair-dir',type=Path,required=True)
args=parser.parse_args(__import__('sys').argv[__import__('sys').argv.index('--')+1:]);pair=args.pair_dir.resolve();out=pair/'ground-diagnostic'
if out.exists() or not pair.is_relative_to(Path(__file__).resolve().parents[1]/'output/secondary_motion'):raise ValueError('使用新的诊断目录')
out.mkdir();records=[]
for variant in ('before','after'):
 path=pair/f'{variant}.blend';sha=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
 bpy.ops.wm.open_mainfile(filepath=str(path),load_ui=False,use_scripts=False)
 scene=bpy.context.scene;scene.frame_set(849);rig=bpy.data.objects['TohsakaRin_arm'];center=rig.matrix_world@rig.pose.bones['腰'].head
 material=bpy.data.materials.new('MBD_GroundReference');material.diffuse_color=(.2,.65,.9,1)
 for index in range(-4,5):
  for direction in range(2):
   points=[(center.x+index*.2,center.y-.8,0),(center.x+index*.2,center.y+.8,0)] if direction==0 else [(center.x-.8,center.y+index*.2,0),(center.x+.8,center.y+index*.2,0)]
   data=bpy.data.curves.new('MBD_GroundGrid','CURVE');data.dimensions='3D';data.bevel_depth=.002;data.bevel_resolution=0
   spline=data.splines.new('POLY');spline.points.add(1)
   for point,co in zip(spline.points,points):point.co=(*co,1)
   data.materials.append(material);obj=bpy.data.objects.new('MBD_GroundGrid',data);scene.collection.objects.link(obj)
 scene.render.filepath=str(out/f'{variant}-0849.png');bpy.ops.render.render(write_still=True)
 assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==sha
 records.append({'variant':variant,'frame':849,'ground_z_m':0,'source_sha256':sha,'source_pair_unchanged':True})
(out/'verification.json').write_text(json.dumps({'scope':'diagnostic stills only; cyan wire grid at actual Z=0, no actor or camera changes','records':records},indent=2),encoding='utf-8')
