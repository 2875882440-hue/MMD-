"""Run with Blender --background --factory-startup --python this_file -- motion.json outdir."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix, Vector
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from motionbridge.exporters import solve_rotations, validate_motion
from motionbridge import __version__


C = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def material(name, color, metallic=.0, roughness=.65):
    item = bpy.data.materials.new(name)
    item.diffuse_color = (*color, 1)
    item.use_nodes = True
    principled = item.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (*color, 1)
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    return item


def build_mannequin(armature, names, parents, rest, shift):
    materials = [material("Torso • slate", (.15, .25, .32)),
                 material("Left • amber", (.95, .47, .11)),
                 material("Right • cyan", (.06, .63, .75)),
                 material("Joints • graphite", (.035, .055, .075)),
                 material("Head • ivory", (.8, .84, .82))]
    vertices, faces, face_materials, weights = [], [], [], []

    def ellipsoid(center, scales, joint, mat, orientation=None):
        center = np.asarray(center, float)
        orientation = np.eye(3) if orientation is None else np.asarray(orientation)
        first = len(vertices)
        rings, segments = 8, 12
        for i in range(rings + 1):
            phi = math.pi * i / rings
            for j in range(segments):
                theta = 2 * math.pi * j / segments
                vec = np.array([math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)])
                vertices.append((center + orientation @ (vec * scales)).tolist())
                weights.append(joint)
        for i in range(rings):
            for j in range(segments):
                faces.append((first + i * segments + j, first + i * segments + (j + 1) % segments,
                              first + (i + 1) * segments + (j + 1) % segments, first + (i + 1) * segments + j))
                face_materials.append(mat)

    positions = (C @ rest.T).T + shift
    hip = positions[0]
    ellipsoid(hip, [.16, .10, .10], 0, 0)
    chest = positions[names.index("Chest")]
    spine = positions[names.index("Spine")]
    ellipsoid((spine + chest) * .5, [.14, .09, max(.14, np.linalg.norm(chest - spine) * .8)], names.index("Spine"), 0)
    ellipsoid(chest + [0, 0, .055], [.19, .095, .14], names.index("Chest"), 0)
    head = positions[names.index("Head")]
    ellipsoid(head + [0, 0, .055], [.095, .085, .12], names.index("Head"), 4)
    # Small dark visor makes facing direction immediately legible.
    ellipsoid(head + [0, -.077, .075], [.065, .012, .025], names.index("Head"), 3)
    for joint, name in enumerate(names):
        if not (name.startswith("Left") or name.startswith("Right")):
            continue
        mat = 1 if name.startswith("Left") else 2
        children = [i for i, parent in enumerate(parents) if parent == joint]
        start = positions[joint]
        if children:
            end = positions[children[0]]
            length = np.linalg.norm(end - start)
            quat = Vector((end - start).tolist()).to_track_quat("Z", "Y")
            basis = np.array(quat.to_matrix())
            radius = .058 if "UpLeg" in name else .044 if "Leg" in name else .038
            ellipsoid((start + end) * .5, [radius, radius, max(.04, length * .52)], joint, mat, basis)
        else:
            if "Toe" in name:
                ellipsoid(start + [0, -.025, .01], [.055, .11, .045], joint, mat)
            elif "Hand" in name:
                ellipsoid(start, [.06, .032, .04], joint, mat)
        ellipsoid(start, [.045] * 3, joint, 3)
    mesh = bpy.data.meshes.new("MotionBridge_MannequinMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("MotionBridge_Mannequin", mesh)
    bpy.context.collection.objects.link(obj)
    for mat in materials:
        obj.data.materials.append(mat)
    for polygon, mat in zip(obj.data.polygons, face_materials):
        polygon.material_index = mat
        polygon.use_smooth = True
    for joint, name in enumerate(names):
        group = obj.vertex_groups.new(name=name)
        indices = [i for i, weight in enumerate(weights) if weight == joint]
        if indices:
            group.add(indices, 1., "REPLACE")
    obj.parent = armature
    modifier = obj.modifiers.new("Body skeleton deformation", "ARMATURE")
    modifier.object = armature
    return obj


def pose_snapshot(armature, frames, names):
    snapshots = {}
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        snapshots[str(frame)] = {name: list(armature.matrix_world @ armature.pose.bones[name].head) for name in names}
    return snapshots


def main():
    arguments = sys.argv[sys.argv.index("--") + 1:]
    motion_path, output_dir = Path(arguments[0]), Path(arguments[1])
    motion = json.loads(motion_path.read_text(encoding="utf-8"))
    names, parents, rest, poses, fps = validate_motion(motion)
    local, world, solved = solve_rotations(motion)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = round(fps)
    scene.render.fps_base = round(fps) / fps
    scene.frame_start, scene.frame_end = 1, len(poses)
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.
    # World-motion sources already carry a grounded global root trajectory.
    world_motion = motion.get("root_mode") == "estimated_world_translation"
    floor_height = 0.0 if world_motion else float(-np.min(rest[:, 1]) + .045)
    shift = np.array([0., 0., floor_height])
    armature_data = bpy.data.armatures.new("MotionBridge_Body")
    armature = bpy.data.objects.new("MotionBridge_Rig", armature_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    armature.show_in_front = True
    armature.data.display_type = "OCTAHEDRAL"
    armature["source_coordinate_system"] = motion.get("coordinate_system")
    armature["root_mode"] = motion.get("root_mode", "in_place")
    armature["floor_offset_meters"] = floor_height
    armature["source_is_synthetic_demo"] = bool(motion.get("is_demo"))
    bpy.ops.object.mode_set(mode="EDIT")
    for joint, name in enumerate(names):
        bone = armature_data.edit_bones.new(name)
        head = C @ rest[joint] + shift
        children = [i for i, p in enumerate(parents) if p == joint]
        if children:
            # Prefer spine for the branching pelvis/chest.
            spine_children = [i for i in children if names[i] in ("Spine", "Spine2", "Chest", "Neck")]
            tail = C @ rest[(spine_children or children)[0]] + shift
        else:
            delta = rest[joint] - rest[parents[joint]]
            tail = head + C @ (delta / np.linalg.norm(delta) * (.12 if name == "Head" else .07))
        bone.head, bone.tail = head.tolist(), tail.tolist()
        if parents[joint] >= 0:
            bone.parent = armature_data.edit_bones[names[parents[joint]]]
        bone.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    mannequin = build_mannequin(armature, names, parents, rest, shift)
    rest_bases = [np.array(armature.data.bones[name].matrix_local.to_3x3()) for name in names]
    for joint, name in enumerate(names):
        armature.pose.bones[name].rotation_mode = "QUATERNION"
    previous_quaternions = {}
    for frame_index in range(len(poses)):
        for joint, name in enumerate(names):
            pb = armature.pose.bones[name]
            basis = rest_bases[joint]
            rotation = basis.T @ C @ local[frame_index, joint] @ C.T @ basis
            quaternion = Matrix(rotation.tolist()).to_quaternion()
            if name in previous_quaternions and quaternion.dot(previous_quaternions[name]) < 0:
                quaternion.negate()
            previous_quaternions[name] = quaternion.copy()
            pb.rotation_quaternion = quaternion
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_index + 1, group=name)
            if joint == 0:
                pb.location = (basis.T @ C @ (poses[frame_index, 0] - rest[0])).tolist()
                pb.keyframe_insert(data_path="location", frame=frame_index + 1, group=name)
    if armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        action.name = "MotionBridge_SyntheticDemo" if motion.get("is_demo") else "MotionBridge_VideoCapture"
        # Blender 4.4+ uses layered actions; the legacy collection may still be available.
        try:
            curves = action.fcurves
        except AttributeError:
            curves = []
            for layer in action.layers:
                for strip in layer.strips:
                    for slot in action.slots:
                        channelbag = strip.channelbag(slot)
                        if channelbag:
                            curves.extend(channelbag.fcurves)
        for curve in curves:
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "LINEAR"

    # A neutral studio scene is useful when opening the editable file.
    ground_mat = material("Stage • dark neutral", (.028, .04, .05))
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "Preview_Ground"
    ground.data.materials.append(ground_mat)
    bpy.ops.object.camera_add(location=(2.8, -4.5, 2.2))
    camera = bpy.context.object
    camera.name = "Preview_Camera"
    camera.rotation_euler = (Vector((0, 0, floor_height + .1)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type, camera.data.ortho_scale = "ORTHO", 2.8
    if world_motion:
        points = solved.reshape(-1, 3) @ C.T + shift
        center = (points.min(axis=0) + points.max(axis=0)) * .5
        camera.location = Vector(center + np.array([2.8, -4.5, 1.2]))
        camera.rotation_euler = (Vector(center) - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera.data.ortho_scale = max(2.8, float(np.linalg.norm(np.ptp(points, axis=0))) * 1.2)
    scene.camera = camera
    for label, location, energy, size in [("Key", (3, -4, 5), 700, 5), ("Fill", (-3, -1, 3), 400, 4)]:
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.name = "Preview_" + label
        light.data.energy, light.data.shape, light.data.size = energy, "DISK", size
        light.rotation_euler = (Vector((0, 0, 1)) - light.location).to_track_quat("-Z", "Y").to_euler()
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Preview_World")
    scene.world.color = (.15, .15, .15)
    scene.render.resolution_x, scene.render.resolution_y = 720, 720
    scene.render.resolution_percentage = 100
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"  # Blender 5.2 renamed the identifier.
    scene.frame_set(1)
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    mannequin.select_set(True)
    bpy.context.view_layer.objects.active = armature
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_distance = 3.1
                area.spaces.active.region_3d.view_location = Vector(center) if world_motion else Vector((0, 0, floor_height))
                area.spaces.active.shading.color_type = "MATERIAL"
    scene["MotionBridge_notes"] = (
        "GVHMR 研究结果，包含估计的世界朝向和位移；未做真实尺度标定。彩色角色为内置蒙皮示例，可编辑骨骼关键帧。"
        if world_motion else "基础单目身体动捕；根骨固定。彩色角色为内置刚性蒙皮示例，可编辑骨骼关键帧。"
    )
    scene["MotionBridge_version"] = __version__
    scene["MotionBridge_solver"] = motion.get("solver", "direction_v1")
    blend_path, fbx_path = output_dir / "motion.blend", output_dir / "motion.fbx"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    sample_frames = sorted(set([1, max(1, len(poses) // 3), max(1, 2 * len(poses) // 3), len(poses)]))
    original = pose_snapshot(armature, sample_frames, names)
    # Verify our quaternion/rest-basis conversion before involving FBX.
    solver_error = 0.
    for frame in sample_frames:
        expected = (C @ solved[frame - 1].T).T + shift
        actual = np.array([original[str(frame)][name] for name in names])
        solver_error = max(solver_error, float(np.max(np.linalg.norm(expected - actual, axis=-1))))
    scene.frame_set(1)
    bpy.ops.export_scene.fbx(filepath=str(fbx_path), use_selection=True, object_types={"ARMATURE", "MESH"},
        add_leaf_bones=False, bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False, bake_anim_simplify_factor=0., bake_anim_step=1.,
        axis_forward="-Z", axis_up="Y", use_armature_deform_only=False, apply_unit_scale=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = round(fps)
    bpy.context.scene.render.fps_base = round(fps) / fps
    bpy.ops.import_scene.fbx(filepath=str(fbx_path), use_anim=True, anim_offset=0., automatic_bone_orientation=False)
    rigs = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    imported = rigs[0] if rigs else None
    all_bones = imported is not None and all(name in imported.pose.bones for name in names)
    errors = []
    if all_bones:
        roundtrip = pose_snapshot(imported, sample_frames, names)
        for frame in sample_frames:
            a = np.array([original[str(frame)][name] for name in names])
            b = np.array([roundtrip[str(frame)][name] for name in names])
            errors.append(float(np.max(np.linalg.norm(a - b, axis=-1))))
    max_error = max(errors) if errors else None
    has_animation = bool(imported and imported.animation_data and imported.animation_data.action)
    has_skin = any(any(mod.type == "ARMATURE" and mod.object == imported for mod in obj.modifiers) for obj in meshes)
    # A file can round-trip positions while retaining abrupt axial rotations.
    # Check every exported frame, including explicit stabilized orientation tracks.
    all_frame_position_error, all_frame_rotation_error, max_world_speed = 0., 0., 0.
    floor_heights=[]
    if all_bones and motion.get('floor_collision',{}).get('enabled'):
        from motionbridge.blender_feet import foot_vertex_groups,evaluated_foot_minimum
        foot_samples=foot_vertex_groups(bpy.context.scene,imported,tokens=('Foot','Toe'))
    if all_bones:
        object_rotation = np.array(imported.matrix_world.to_quaternion().to_matrix())
        inverse_rest = {name: np.array(imported.data.bones[name].matrix_local.to_3x3()).T for name in names}
        previous = None
        for f in range(len(poses)):
            bpy.context.scene.frame_set(f + 1)
            if motion.get('floor_collision',{}).get('enabled'):
                floor_heights.append(evaluated_foot_minimum(foot_samples,bpy.context.evaluated_depsgraph_get()))
            actual_positions = np.array([imported.matrix_world @ imported.pose.bones[n].head for n in names])
            expected_positions = (C @ solved[f].T).T + shift
            all_frame_position_error = max(all_frame_position_error, float(np.linalg.norm(actual_positions - expected_positions, axis=-1).max()))
            actual_rotations = np.array([object_rotation @ np.array(imported.pose.bones[n].matrix.to_3x3()) @ inverse_rest[n] @ object_rotation.T for n in names])
            expected_rotations = C @ world[f] @ C.T
            # Frobenius error is stable for tiny floating-point rotation errors.
            all_frame_rotation_error = max(all_frame_rotation_error, float(np.linalg.norm(actual_rotations - expected_rotations, axis=(-1,-2)).max()))
            if previous is not None:
                relative = np.swapaxes(previous, -1, -2) @ actual_rotations
                angles = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1., 1.)))
                max_world_speed = max(max_world_speed, float(angles.max() * fps))
            previous = actual_rotations
    stability = motion.get('rotation_stabilization', {})
    continuity_passed = not stability.get('enabled') or max_world_speed <= max(stability['torso_head_speed_limit_deg_s'], stability['limb_speed_limit_deg_s']) + 1.
    report = {
        "passed": bool(all_bones and has_animation and has_skin and solver_error < .005 and max_error is not None and max_error < .005 and all_frame_position_error < .005 and all_frame_rotation_error < .002 and continuity_passed),
        "blender_version": bpy.app.version_string, "joint_count": len(names), "frames": len(poses), "fps": fps,
        "coordinate_conversion": "source (x,y,z) -> Blender (x,-z,y), meters",
        "root_mode": motion.get("root_mode","in_place"), "floor_offset_meters": floor_height, "is_demo": bool(motion.get("is_demo")),
        "solver_max_joint_error_meters": solver_error,
        "motionbridge_version": __version__, "solver": motion.get("solver", "direction_v1"),
        "all_frames": {"count": len(poses), "max_position_error_m": all_frame_position_error,
                       "max_rotation_matrix_error": all_frame_rotation_error, "max_world_speed_deg_s": max_world_speed,
                       "continuity_limit_applicable": bool(stability.get('enabled')),
                       "continuity_limit_passed": continuity_passed if stability.get('enabled') else None},
        "rotation_stabilization": stability,
        "fbx_roundtrip": {"armatures": len(rigs), "meshes": len(meshes), "all_bones_present": bool(all_bones),
                          "animation_present": has_animation, "armature_skin_present": has_skin,
                          "sample_frames": sample_frames, "import_animation_offset_frames": 0,
                          "max_joint_error_meters": max_error},
        "files": {"motion.blend": blend_path.stat().st_size, "motion.fbx": fbx_path.stat().st_size},
        "elapsed_seconds": round(time.monotonic() - start_time, 2),
    }
    if floor_heights:
        floor_y=motion['floor_collision']['floor_y_m']
        report['foot_floor_collision']={'evaluated_fbx_frames':len(floor_heights),'min_z_m':min(floor_heights),
            'passed':min(floor_heights)>=floor_y-1e-5,'note':'Actual imported FBX foot-weighted geometry at output frames.'}
        report['passed'] &= report['foot_floor_collision']['passed']
    (output_dir / "blender_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MOTIONBRIDGE_VERIFICATION " + json.dumps(report, ensure_ascii=True), flush=True)
    if not report["passed"]:
        raise RuntimeError("Blender / FBX round-trip validation failed")


if __name__ == "__main__":
    main()
