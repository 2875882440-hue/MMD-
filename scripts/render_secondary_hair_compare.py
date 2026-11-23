"""Render before/after saved scenes with their shared camera; resume hash-checked frames."""
from pathlib import Path
import argparse,hashlib,json,sys,struct
import bpy
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--pair-dir',type=Path,required=True)
parser.add_argument('--frames',help='Optional comma-separated absolute frame numbers')
parser.add_argument('--resume',action='store_true')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=args.pair_dir.resolve()
if not root.is_relative_to(ROOT/'output/secondary_motion'):raise ValueError('非法对照目录')
target=root/('shots' if args.frames else 'frames');target.mkdir(exist_ok=True)
signatures={v:hashlib.sha256((root/f'{v}.blend').read_bytes()).hexdigest() for v in ('before','after')}
manifest=target/'render-manifest.json'
if manifest.exists():
    old=json.loads(manifest.read_text(encoding='utf-8'))
    if not args.resume or old['scene_sha256']!=signatures:raise ValueError('已有渲染输出或源工程变化，拒绝覆盖')
else:manifest.write_text(json.dumps({'scene_sha256':signatures,'same_camera':True}),encoding='utf-8')
cameras={}
camera_hashes={}
for variant in ('before','after'):
    bpy.ops.wm.open_mainfile(filepath=str(root/f'{variant}.blend'),load_ui=False,use_scripts=False)
    scene=bpy.context.scene;camera=scene.camera
    cameras[variant]={'matrix':[list(row) for row in camera.matrix_world],'scale':camera.data.ortho_scale,
                      'engine':scene.render.engine,'size':[scene.render.resolution_x,scene.render.resolution_y],
                      'fps':scene.render.fps/scene.render.fps_base,'frame_start':scene.frame_start,'frame_end':scene.frame_end}
    if variant=='after' and cameras['before']!=cameras['after']:raise ValueError('两版相机或渲染配置不一致')
    folder=target/variant;folder.mkdir(exist_ok=True)
    frames=[int(n) for n in args.frames.split(',')] if args.frames else range(scene.frame_start,scene.frame_end+1)
    camera_hash=hashlib.sha256()
    for frame in frames:
        if not scene.frame_start<=frame<=scene.frame_end:raise ValueError('帧超出范围')
        output=folder/f'{frame:04d}.png'
        scene.frame_set(frame)
        camera_hash.update(struct.pack('<16f',*[v for row in camera.matrix_world for v in row]))
        camera_hash.update(struct.pack('<f',camera.data.ortho_scale))
        if output.exists() and args.resume:continue
        if output.exists():raise ValueError('已有帧，拒绝覆盖')
        scene.frame_set(frame);scene.render.filepath=str(output)
        bpy.ops.render.render(write_still=True)
        if frame%30==1:print('HAIR_RENDER_PROGRESS',variant,frame,flush=True)
    camera_hashes[variant]=camera_hash.hexdigest()
    if variant=='after' and camera_hashes['before']!=camera_hashes['after']:raise ValueError('前后相机逐帧矩阵不同')
(target/'render-complete.json').write_text(json.dumps({'passed':True,'cameras':cameras,'same_camera':True,
    'fps':cameras['before']['fps'],'camera_frame_sha256':camera_hashes},indent=2),encoding='utf-8')
