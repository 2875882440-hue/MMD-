"""Blender-side CPU render of the saved world-motion export, every frame."""
from pathlib import Path
import argparse
import json
import sys
import time
import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("--samples-only", action="store_true")
parser.add_argument("--sample-times", default="0,1,2,3,4,5")
args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
folder = args.folder.resolve()
bpy.ops.wm.open_mainfile(filepath=str(folder / "motion.blend"), load_ui=False)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 8
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 2
scene.render.threads_mode = "FIXED"
scene.render.threads = 4
scene.render.resolution_x, scene.render.resolution_y = 360, 640
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.use_sequencer = False
scene.render.use_compositing = False
rig = bpy.data.objects["MotionBridge_Rig"]
points = []
for f in range(scene.frame_start, scene.frame_end+1):
    scene.frame_set(f)
    points.extend(rig.matrix_world @ bone.head for bone in rig.pose.bones)
lo = Vector(tuple(min(p[i] for p in points) for i in range(3)))
hi = Vector(tuple(max(p[i] for p in points) for i in range(3)))
center = (lo+hi)/2
camera = scene.camera
motion = json.loads((folder / 'motion.json').read_text(encoding='utf-8'))
from mathutils import Matrix
first_front = Matrix(motion['world_rotations'][0][0]) @ Vector((0,0,1))
front = Vector((first_front.x,-first_front.z,0)).normalized()
camera.location = center + front*5 + Vector((0,0,.15))
camera.rotation_euler = (center-camera.location).to_track_quat("-Z","Y").to_euler()
right = front.cross(Vector((0,0,1)))
horizontal = [p.dot(right) for p in points]
camera.data.ortho_scale = max((hi.z-lo.z)*1.2, (max(horizontal)-min(horizontal))*640/360*1.2,2.0)
dest = folder / "world_frames_initial_view"
dest.mkdir(exist_ok=True)
fps = scene.render.fps / scene.render.fps_base
selected = {min(scene.frame_end,max(scene.frame_start,round(float(i)*fps)+1)) for i in args.sample_times.split(',')}
started = time.monotonic()
rendered = []
for f in range(scene.frame_start,scene.frame_end+1):
    if args.samples_only and f not in selected:
        continue
    path = dest / f"{f:05d}.png"
    if path.exists():
        continue
    scene.frame_set(f)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    rendered.append(f)
report = {"rendered_frames":rendered,"seconds":time.monotonic()-started,
          "cpu_threads":4,"saved_blend_reopened":True,"full_fps":fps,
          "camera":"fixed camera facing initial predicted body heading, fitted to all positions; no per-frame reframing or pose edits"}
(folder / ("world_render_samples.json" if args.samples_only else "world_render_full.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
