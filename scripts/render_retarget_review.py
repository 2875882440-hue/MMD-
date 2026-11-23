"""Reopen the saved retargeted project and render reproducible CPU review frames."""
from pathlib import Path
import json
import sys
import time
import math
import argparse
import bpy

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=Path)
parser.add_argument('phase', choices=['samples', 'preview'])
parser.add_argument('--start-index', type=int, default=0)
parser.add_argument('--end-index', type=int)
parser.add_argument('--skip-frames', default='')
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--frame-step', type=int, choices=[1,2], default=2)
parser.add_argument('--reference-camera',type=Path)
parser.add_argument('--sample-times',default='1,4,7,10,13,16,19,23,28')
parser.add_argument('--diagnostic-stage',action='store_true',help='Review only: hide decorative stage, add plane and follow root with a fixed-angle camera. Never save the scene.')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
folder = args.folder.resolve()
phase = args.phase
bpy.ops.wm.open_mainfile(filepath=str(folder / 'Nahida_MotionBridge_Dance.blend'), load_ui=False)
scene = bpy.context.scene
if args.diagnostic_stage:
    if args.reference_camera:raise ValueError('Diagnostic view cannot be a same-camera beauty comparison')
    from mathutils import Vector
    for obj in scene.objects:
        if obj.type in {'MESH','CURVE','FONT'} and not obj.name.startswith('Nahida'):obj.hide_render=True
    bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,0));ground=bpy.context.object
    material=bpy.data.materials.new('Diagnostic ground');material.diffuse_color=(.12,.14,.16,1);ground.data.materials.append(material)
    data=bpy.data.cameras.new('Diagnostic camera');camera=bpy.data.objects.new('Diagnostic camera',data);scene.collection.objects.link(camera)
    data.type='ORTHO';data.ortho_scale=1.65;scene.camera=camera
if args.reference_camera:
    camera_name=scene.camera.name
    with bpy.data.libraries.load(str(args.reference_camera.resolve()),link=False) as (src,dst):
        if camera_name not in src.objects:raise ValueError('Reference camera missing')
        dst.objects=[camera_name]
    reference=dst.objects[0]
    scene.collection.objects.link(reference)
    scene.camera=reference
scene.cycles.device = 'CPU'
scene.render.threads_mode = 'FIXED'
scene.render.threads = args.threads
scene.render.image_settings.file_format = 'PNG'
scene.render.use_sequencer = False
scene.render.use_compositing = False
if phase == 'samples':
    frames = [round(float(t) * scene.render.fps / scene.render.fps_base) + 1 for t in args.sample_times.split(',')]
    if any(f<scene.frame_start or f>scene.frame_end for f in frames):raise ValueError('Sample times must lie in the saved animation')
    dest = folder / 'samples'
    scene.render.resolution_x, scene.render.resolution_y = 540, 720
else:
    frames = range(1, scene.frame_end + 1, args.frame_step)
    dest = folder / 'preview_frames'
    scene.render.resolution_x, scene.render.resolution_y = 360, 480
scene.render.resolution_percentage = 100
scene.cycles.samples = 8
frames = list(frames)[args.start_index:args.end_index]
skip_frames = {int(f) for f in args.skip_frames.split(',') if f}
dest.mkdir(exist_ok=True)
start = time.monotonic()
report = {'phase': phase, 'frames': [], 'saved_blend_reopened': True, 'device': 'CPU', 'threads': scene.render.threads}
report['reference_camera']=str(args.reference_camera.resolve()) if args.reference_camera else None
report['diagnostic_stage']=args.diagnostic_stage
report['scene_saved']=False
rig = bpy.data.objects['Nahida_arm']
for i, frame in enumerate(frames):
    scene.frame_set(frame)
    if args.diagnostic_stage:
        center=rig.pose.bones['全ての親'].matrix.translation+Vector((0,0,.60))
        scene.camera.location=center+Vector((0,-4,.7))
        scene.camera.rotation_euler=(center-scene.camera.location).to_track_quat('-Z','Y').to_euler()
    report['frames'].append({'frame': frame, 'joints': {n: list(rig.pose.bones[n].head) for n in ['左手首', '右手首', '左足首', '右足首', '頭']}})
    scene.render.filepath = str(dest / f'{frame:04d}.png')
    if frame not in skip_frames:
        print('REVIEW_RENDER_FRAME', frame, flush=True)
        bpy.ops.render.render(write_still=True)
    if i % 30 == 0:
        print('REVIEW_PROGRESS', phase, i + 1, '/', len(frames), round(time.monotonic() - start, 1), flush=True)
report['elapsed_s'] = round(time.monotonic() - start, 3)
suffix = f'_{args.start_index}_{args.end_index}' if args.end_index is not None else ''
(folder / f'{phase}_render_report{suffix}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('REVIEW_COMPLETE', phase, report['elapsed_s'], flush=True)
