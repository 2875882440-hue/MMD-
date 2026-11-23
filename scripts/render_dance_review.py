"""Blender-side assessment: import the actual FBX, save an editable review, render."""
from pathlib import Path
import json
import math
import sys
import time
import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
folder = Path(args[0]).resolve()
phase = args[1] if len(args) > 1 else "probe"
assessment = json.loads((folder / "assessment.json").read_text(encoding="utf-8"))
outputs = Path(assessment["outputs"])
motion = json.loads((outputs / "motion.json").read_text(encoding="utf-8"))
fps, count = motion["fps"], motion["num_frames"]

if phase == "probe":
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(outputs / "motion.fbx"), use_anim=True, anim_offset=0,
                             automatic_bone_orientation=False)
    scene = bpy.context.scene
    rig = next(obj for obj in scene.objects if obj.type == "ARMATURE")
    scene.frame_start, scene.frame_end = 1, count
    scene.render.fps, scene.render.fps_base = round(fps), round(fps) / fps
    scene.unit_settings.system = "METRIC"
    scene.frame_set(1)
    # Full-body fixed camera: no frame-specific reframing or motion correction.
    coords = []
    foot_heights = []
    snapshots = {}
    for frame in range(1, count + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        positions = {b.name: list(rig.matrix_world @ b.head) for b in rig.pose.bones}
        coords.extend(positions.values())
        foot_heights.append({"time": (frame - 1) / fps,
            "left_ankle_z": positions["LeftFoot"][2], "right_ankle_z": positions["RightFoot"][2],
            "left_toe_z": positions["LeftToeBase"][2], "right_toe_z": positions["RightToeBase"][2]})
        if frame in [round(t * fps) + 1 for t in assessment["sample_times"]]:
            snapshots[str(frame)] = positions
    low = min(p[2] for p in coords) - .18
    high = max(p[2] for p in coords) + .24
    half_width = max(abs(p[0]) for p in coords) + .15
    height = max(2.35, high - low, half_width * 2 * 640 / 360)
    center = (high + low) * .5
    bpy.ops.object.camera_add(location=(0, -5, center))
    camera = bpy.context.object
    camera.name = "Assessment_Front_Camera"
    camera.rotation_euler = (Vector((0, 0, center)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = height
    scene.camera = camera
    bpy.ops.object.camera_add(location=(4.5, -4.5, center + .3))
    side = bpy.context.object
    side.name = "Assessment_ThreeQuarter_Camera"
    side.rotation_euler = (Vector((0, 0, center)) - side.location).to_track_quat("-Z", "Y").to_euler()
    side.data.type, side.data.ortho_scale = "ORTHO", height
    bpy.ops.mesh.primitive_plane_add(size=12)
    ground = bpy.context.object
    ground.name = "Assessment_Ground"
    mat = bpy.data.materials.new("Assessment_Ground_Material")
    mat.diffuse_color = (.055, .07, .085, 1)
    mat.use_nodes = True
    mat.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = mat.diffuse_color
    ground.data.materials.append(mat)
    for location, power, size in [((2, -4, 5), 650, 4), ((-3, -2, 3), 500, 4)]:
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.data.energy, light.data.size = power, size
        light.rotation_euler = (Vector((0, 0, 1)) - light.location).to_track_quat("-Z", "Y").to_euler()
    world = bpy.data.worlds.new("Assessment_World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (.12, .15, .19, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .7
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
    scene.render.film_transparent = False
    scene.render.use_sequencer = False
    scene.view_settings.view_transform = "AgX"
    for sec in assessment["sample_times"]:
        scene.timeline_markers.new(f"Compare {sec:.1f}s", frame=round(sec * fps) + 1)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
                area.spaces.active.shading.color_type = "MATERIAL"
    rig.show_in_front = True
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    scene["assessment_source"] = assessment["source_filename"]
    scene["assessment_notes"] = f"Actual FBX import; {fps:g}fps original keys; root stays fixed. No manual pose cleanup. Preview samples every 2 frames at {fps/2:g}fps."
    scene.frame_set(min(count, round(fps) + 1))
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "Dance_Tracking_Review.blend"))
    (folder / "blender_import_assessment.json").write_text(json.dumps({
        "blender_version": bpy.app.version_string, "imported_fbx": str(outputs / "motion.fbx"),
        "bones": len(rig.pose.bones), "has_animation": bool(rig.animation_data and rig.animation_data.action),
        "frames": count, "fps": fps, "camera_ortho_height": height,
        "snapshots": snapshots, "foot_heights": foot_heights}, indent=2), encoding="utf-8")
    scene.render.filepath = str(folder / "blender_probe.png")
    started = time.monotonic()
    bpy.ops.render.render(write_still=True)
    print(f"PROBE_COMPLETE seconds={time.monotonic()-started:.3f}", flush=True)
else:
    bpy.ops.wm.open_mainfile(filepath=str(folder / "Dance_Tracking_Review.blend"))
    scene = bpy.context.scene
    frames_dir = folder / "blender_frames"
    frames_dir.mkdir(exist_ok=True)
    # All original animation keys remain in the .blend; preview samples at 15fps.
    started = time.monotonic()
    selected = {round(t * fps / 2) for t in assessment["sample_times"]}
    rendered_count = 0
    for idx, frame in enumerate(range(1, count + 1, 2)):
        if phase == "samples" and idx not in selected:
            continue
        rendered_count += 1
        output = frames_dir / f"{idx:05d}.png"
        if output.is_file():
            continue
        scene.frame_set(frame)
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        if idx % 20 == 0:
            print(f"REVIEW_PROGRESS {idx+1}/{math.ceil(count/2)} seconds={time.monotonic()-started:.2f}", flush=True)
    print(f"REVIEW_RENDER_COMPLETE frames={rendered_count} seconds={time.monotonic()-started:.2f}", flush=True)
