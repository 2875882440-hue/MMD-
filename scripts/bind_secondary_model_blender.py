"""Blender-only binding validation; adds namespaced diagnostic vertex groups.

No simulation, weights rewrite, body retargeting, material or modifier change
is saved. Optional --scene validates an existing animation without using it as
configuration. Run with --disable-autoexec and --python-exit-code.
"""
from pathlib import Path
import argparse, hashlib, json, math, struct, sys
import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]


def curves_digest():
    rows = []
    for action in bpy.data.actions:
        curves = []
        if hasattr(action, 'fcurves'):
            curves.extend(action.fcurves)
        for layer in action.layers:
            for strip in layer.strips:
                for bag in getattr(strip, 'channelbags', []):
                    curves.extend(bag.fcurves)
        rows.append((action.name, sorted((f.data_path, f.array_index,
            [(tuple(k.co), tuple(k.handle_left), tuple(k.handle_right), k.interpolation) for k in f.keyframe_points])
            for f in curves)))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def mesh_digest(mesh):
    sha = hashlib.sha256()
    for key in mesh.data.shape_keys.key_blocks if mesh.data.shape_keys else []:
        values = np.empty(len(key.data) * 3, dtype=np.float32)
        key.data.foreach_get('co', values)
        sha.update(key.name.encode()); sha.update(values.tobytes())
        sha.update(struct.pack('<f', key.value))
    names = {g.index: g.name for g in mesh.vertex_groups if not g.name.startswith('MBP_')}
    for v in mesh.data.vertices:
        sha.update(json.dumps(sorted((names[g.group], g.weight) for g in v.groups if g.group in names)).encode())
    return sha.hexdigest()


def world_digest(rig, names, frames):
    sha = hashlib.sha256()
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        for name in names:
            matrix = rig.matrix_world @ rig.pose.bones[name].matrix
            values = [v for row in matrix for v in row]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('非有限主骨骼矩阵')
            sha.update(struct.pack('<16f', *values))
    return sha.hexdigest()


def validate_geometry(mesh, rig, plan):
    geometry = plan['geometry']
    if len(mesh.data.vertices) != len(geometry['positions_m']):
        raise ValueError('顶点数不一致，拒绝套用蒙皮索引')
    positions = np.empty(len(mesh.data.vertices) * 3, dtype=np.float32)
    mesh.data.vertices.foreach_get('co', positions)
    expected = np.asarray(geometry['positions_m'])
    actual = np.asarray([tuple(mesh.matrix_world @ Vector(v)) for v in positions.reshape(-1, 3)])
    error = float(np.max(np.abs(actual - expected)))
    if error > 2e-6:
        raise ValueError(f'静态顶点位置与单位不匹配: {error}')
    faces = [tuple(sorted(p.vertices)) for p in mesh.data.polygons]
    if faces != [tuple(sorted(f)) for f in geometry['triangles']]:
        raise ValueError('面拓扑或顶点顺序变化，需重新映射')
    if [p.material_index for p in mesh.data.polygons] != geometry['face_material_ids']:
        raise ValueError('面材质分区不匹配')
    if [slot.material.name if slot.material else None for slot in mesh.material_slots] != geometry['material_names']:
        raise ValueError('材质槽名称或顺序不匹配；需显式适配新角色')
    mapping = {b['id']: b['blender_name'] for b in plan['bones']}
    missing = set(mapping.values()) - set(rig.data.bones.keys())
    if missing:
        raise ValueError(f'目标骨骼缺失: {missing}')
    head_error = max((rig.matrix_world @ rig.data.bones[b['blender_name']].head_local - Vector(b['rest_head_m'])).length
                     for b in plan['bones'])
    if head_error > 2e-6:
        raise ValueError(f'骨骼静态位置不匹配: {head_error}')
    group_names = {g.index: g.name for g in mesh.vertex_groups}
    weight_error = 0.0
    for vertex, source in zip(mesh.data.vertices, geometry['skin_weights']):
        expected_weights = {mapping[int(b)]: w for b, w in source.items()}
        actual_weights = {group_names[g.group]: g.weight for g in vertex.groups if g.weight > 0}
        for name in set(expected_weights) | set(actual_weights):
            # SDEF helper masks are not deform bone weights.
            if name not in mapping.values():
                continue
            weight_error = max(weight_error, abs(expected_weights.get(name, 0) - actual_weights.get(name, 0)))
    if weight_error > 2e-6:
        raise ValueError(f'原生蒙皮与Blender绑定不一致: {weight_error}')
    return {'vertices': len(mesh.data.vertices), 'triangles': len(faces), 'position_max_error_m': error,
            'bone_head_max_error_m': head_error, 'skin_max_error': weight_error, 'native_bones_bound': len(mapping)}


def validate_saved_groups(mesh, plan):
    errors = []
    for region in plan['regions']:
        pin = mesh.vertex_groups[f"MBP_{region['id']}_pin"]
        mask = mesh.vertex_groups[f"MBP_{region['id']}_region"]
        for vertex, expected in zip(region['vertex_ids'], region['pin_weights']):
            try:
                actual = pin.weight(vertex)
            except RuntimeError:
                actual = 0.0
            errors.extend((abs(actual-expected), abs(mask.weight(vertex)-1)))
    error = max(errors)
    assert error < 1e-6, error
    return error


def render_diagnostics(mesh, rig, plan, destination):
    scene = bpy.context.scene
    rig.data.pose_position = 'REST'
    for obj in scene.objects:
        if obj.type == 'MESH' and obj != mesh:
            obj.hide_render = True
    colors = {'hair': (.06, .8, .72, 1), 'cloth': (.18, .4, 1, 1),
              'accessory': (.85, .3, .95, 1), 'body': (.38, .38, .4, 1), 'boots': (.2, .2, .22, 1)}
    attribute = mesh.data.color_attributes.new(name='MBP_diagnostic', type='FLOAT_COLOR', domain='CORNER')
    for region in plan['regions']:
        base = colors[region['role']]
        pins = dict(zip(region['vertex_ids'], region['pin_weights']))
        for face_id in region['face_ids']:
            for loop_id in mesh.data.polygons[face_id].loop_indices:
                vid = mesh.data.loops[loop_id].vertex_index
                fixed = pins[vid] if region['role'] in ('hair', 'cloth') else 0
                attribute.data[loop_id].color = tuple(base[i] * (1-fixed) + (1., .45, .07, 1.)[i]*fixed for i in range(4))
    mesh.data.color_attributes.active_color = attribute
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.color_type = 'VERTEX'
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.background_type = 'WORLD'
    if scene.world is None:
        scene.world = bpy.data.worlds.new('MBP_DiagnosticWorld')
    scene.world.color = (.07, .07, .07)
    camera_data = bpy.data.cameras.new('MBP_DiagnosticCamera')
    camera = bpy.data.objects.new('MBP_DiagnosticCamera', camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = 'ORTHO'; camera_data.ortho_scale = 1.95
    scene.render.resolution_x = 640; scene.render.resolution_y = 800; scene.render.resolution_percentage = 100
    for label, y in [('front', -5), ('back', 5)]:
        camera.location = (0, y, .88)
        camera.rotation_euler = (Vector((0, 0, .86)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = str(destination/f'partition-{label}.png')
        bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'output/secondary_motion') or out.exists():
        raise ValueError('必须使用开发副本output/secondary_motion下的新目录')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if plan['enabled'] is not False or plan['simulation_performed'] is not False:
        raise ValueError('绑定阶段不得启用仿真')
    source = (args.scene or Path(plan['bindings']['blend_asset'])).resolve()
    if not source.is_relative_to(ROOT):
        raise ValueError('只允许读取当前开发副本中的角色工程')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False, use_scripts=False)
    scene = bpy.context.scene
    if scene.rigidbody_world and scene.rigidbody_world.enabled:
        raise ValueError('源工程已启用刚体仿真，不能用作只读绑定验证')
    mesh = bpy.data.objects[plan['bindings']['mesh']]
    rig = bpy.data.objects[plan['bindings']['armature']]
    geometry = validate_geometry(mesh, rig, plan)
    names = [plan['bones'][i]['blender_name'] for i in plan['protected_bone_ids']]
    frames = list(range(scene.frame_start, scene.frame_end+1)) if args.scene else [scene.frame_start]
    before = {'curves': curves_digest(), 'skin_and_morphs': mesh_digest(mesh),
              'protected_world': world_digest(rig, names, frames)}
    if any(g.name.startswith('MBP_') for g in mesh.vertex_groups):
        raise ValueError('源工程已有MBP组，拒绝覆盖')
    for region in plan['regions']:
        pin = mesh.vertex_groups.new(name=f"MBP_{region['id']}_pin")
        mask = mesh.vertex_groups.new(name=f"MBP_{region['id']}_region")
        mask.add(region['vertex_ids'], 1.0, 'REPLACE')
        for vertex, value in zip(region['vertex_ids'], region['pin_weights']):
            if value > 0:
                pin.add([vertex], value, 'REPLACE')
    mesh['motionbridge_secondary_profile'] = plan['profile_id']
    mesh['motionbridge_secondary_model_sha256'] = plan['model_sha256']
    mesh['motionbridge_secondary_state'] = 'binding_only_no_simulation'
    out.mkdir(parents=True)
    bpy.context.scene.frame_set(scene.frame_start)
    saved = out/'configured-model.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(saved))
    bpy.ops.wm.open_mainfile(filepath=str(saved), load_ui=False, use_scripts=False)
    mesh = bpy.data.objects[plan['bindings']['mesh']]; rig = bpy.data.objects[plan['bindings']['armature']]
    after = {'curves': curves_digest(), 'skin_and_morphs': mesh_digest(mesh),
             'protected_world': world_digest(rig, names, frames)}
    assert before == after, (before, after)
    assert source_hash == hashlib.sha256(source.read_bytes()).hexdigest()
    group_error = validate_saved_groups(mesh, plan)
    report = {'passed': True, 'simulation_performed': False, 'source': str(source), 'source_sha256': source_hash,
              'output': str(saved), 'geometry': geometry, 'frames_checked': len(frames),
              'protected_bones_checked': len(names), 'shape_keys': len(mesh.data.shape_keys.key_blocks),
              'new_diagnostic_groups': len(plan['regions'])*2,
              'saved_group_max_error': group_error,
              'plan_file_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(),
              'before': before, 'after_save_reopen': after, 'source_unchanged': True}
    (out/'binding-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('MBP_BINDING_PASSED', json.dumps(report, ensure_ascii=False), flush=True)
    if args.preview:
        render_diagnostics(mesh, rig, plan, out)


if __name__ == '__main__':
    main()
