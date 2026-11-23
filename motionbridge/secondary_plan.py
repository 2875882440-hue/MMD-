"""Compile model-owned secondary-motion bindings. Does not enable a solver.

Profiles are asset-specific; the compiler and diagnostics are reusable. A video
is deliberately not an input. All planned overrides live in the output plan,
never in the PMX or primary body animation.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from .secondary_motion import _inside, _strict_pmx, validate_profile
from .secondary_topology import material_partition, protected_closure, skin_weights
from .secondary_units import position, rigid_parameters, joint_parameters, mask_allows


def compile_profile(profile, root):
    # Resolve reviewed model data without mutating caller-owned profiles.
    if profile.get('surface_segmentation', {}).get('review_file') is not None:
        import copy
        from .secondary_source_map import load_model_document
        profile = copy.deepcopy(profile)
        config = profile['surface_segmentation']
        if config.get('review') is not None:
            raise ValueError('分类不能同时使用内嵌和文件配置')
        config['review'] = load_model_document(root, config['review_file'])
    path = validate_profile(profile, root)
    if profile.get('schema_version') != 2:
        raise ValueError('完整配置编译需要schema_version=2')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if profile['model'].get('sha256') != digest:
        raise ValueError('模型指纹改变，必须重新确认配置及顶点绑定')
    if profile.get('protected_surface_policy') != 'exclude_and_report':
        raise ValueError('必须显式保护身体蒙皮')
    expected_policy = {'hair': 'bone_dynamics_candidate', 'cloth': 'bone_dynamics_candidate',
                       'accessory': 'follow_source_skinning', 'body': 'kinematic', 'boots': 'kinematic'}
    for region in profile.get('regions', []):
        if (not isinstance(region.get('id'), str) or not region['id'].isascii()
            or not region['id'].replace('_', '').isalnum()
            or region.get('deformation_policy') != expected_policy.get(region.get('role'))):
            raise ValueError('分区标识或变形策略无效，身体区域必须固定')
        if not isinstance(region.get('materials'), list) or not region['materials']:
            raise ValueError('分区材质必须非空')
    for role in ('hair', 'cloth'):
        tokens = profile.get('candidate_name_tokens', {}).get(role)
        if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) or not t.strip() for t in tokens):
            raise ValueError('衣发候选名称标记必须非空')
    model = _strict_pmx(path)
    scale = profile['model']['scale']
    partitions, face_materials = material_partition(model, profile['regions'])
    protected = protected_closure(model.bones, profile['primary_bones'])
    skins = [skin_weights(v.weight, len(model.bones)) for v in model.vertices]
    referenced = set().union(*(part['vertices'] for part in partitions.values()))
    unreferenced = set(range(len(model.vertices))) - referenced
    protected_vertices = set().union(*(part['vertices'] for part in partitions.values()
                                      if part['definition']['role'] in ('body', 'boots'))) | unreferenced
    body_influence = set().union(*(set(skins[v]) for v in protected_vertices))
    dynamic = {r.bone for r in model.rigids if r.mode in (1, 2) and r.bone is not None}
    if any(type(b) is not int or not 0 <= b < len(model.bones) for b in dynamic):
        raise ValueError('动态刚体骨骼引用非法')
    votes = {i: set() for i in dynamic}
    for part in partitions.values():
        if part['definition']['role'] in ('hair', 'cloth'):
            for v in part['vertices']:
                for bone in set(skins[v]) & dynamic:
                    votes[bone].add(part['definition']['role'])
    selected, excluded = set(), []
    rules = profile['candidate_name_tokens']
    for bone in sorted(dynamic):
        name = model.bones[bone].name
        roles = votes[bone] | {role for role in ('hair', 'cloth') if any(t in name for t in rules[role])}
        reason = ('primary_dependency' if bone in protected else
                  'influences_protected_body_or_boots' if bone in body_influence else
                  'external_parent_transform_requires_adapter' if model.bones[bone].externalTransKey is not None else
                  'outside_requested_hair_cloth' if not roles else None)
        if reason:
            excluded.append({'bone_id': bone, 'bone': name, 'reason': reason})
        else:
            selected.add(bone)
    aliases = profile.get('bindings', {}).get('bone_aliases', {})
    unknown = set(aliases) - {b.name for b in model.bones}
    targets = [aliases.get(b.name, b.name) for b in model.bones]
    if unknown or len(set(targets)) != len(targets) or any(not isinstance(t, str) or not t for t in targets):
        raise ValueError('骨骼别名有未知名称、空名称或重复目标')
    blend = _inside(root, profile['bindings']['blend_asset'])
    if blend.suffix.lower() != '.blend' or not blend.is_file():
        raise ValueError('Blender角色资产不存在')
    rigids = []
    dynamic_rigids = set()
    for index, rigid in enumerate(model.rigids):
        if rigid.mode not in (0, 1, 2):
            raise ValueError('不支持的刚体模式')
        if rigid.bone is not None and not 0 <= rigid.bone < len(model.bones):
            raise ValueError('刚体骨骼引用非法')
        active = rigid.mode != 0 and rigid.bone in selected
        if active:
            dynamic_rigids.add(index)
        rigids.append({'id': index, 'name': rigid.name, 'bone_id': rigid.bone,
                       'native': {'mode': rigid.mode, 'size': rigid.size, 'position': rigid.location,
                                  'rotation': rigid.rotation},
                       'planned_mode': 'secondary_dynamic_candidate' if active else 'kinematic_follow',
                       'override_reason': None if active or rigid.mode == 0 else 'protected_or_out_of_scope',
                       'blender': rigid_parameters(rigid, scale)})
    joints = []
    adjacency = {i: set() for i in dynamic_rigids}
    anchored = set()
    for index, joint in enumerate(model.joints):
        ends = (joint.src_rigid, joint.dest_rigid)
        if any(v is not None and (type(v) is not int or not 0 <= v < len(rigids)) for v in ends):
            raise ValueError('关节刚体引用非法')
        active_ends = set(ends) & dynamic_rigids
        if len(active_ends) == 2:
            a, b = active_ends
            adjacency[a].add(b)
            adjacency[b].add(a)
        elif len(active_ends) == 1 and None not in ends:
            anchored.update(active_ends)
        joints.append({'id': index, 'name': joint.name, 'rigid_ids': ends,
                       'affects_candidates': bool(active_ends), 'blender': joint_parameters(joint, scale)})
    components = []
    remaining = set(dynamic_rigids)
    while remaining:
        members = {min(remaining)}
        todo = list(members)
        while todo:
            new = adjacency[todo.pop()] - members
            members.update(new)
            todo.extend(new)
        remaining -= members
        components.append({'rigid_ids': sorted(members), 'native_kinematic_anchor_rigids': sorted(members & anchored),
                           'requires_explicit_root_attachment': not bool(members & anchored)})
    regions = []
    ownership = Counter(v for part in partitions.values() for v in part['vertices'])
    for identifier, part in partitions.items():
        definition = part['definition']
        vertices = sorted(part['vertices'])
        movable = selected if definition['role'] in ('hair', 'cloth', 'accessory') else set()
        free = [sum(w for b, w in skins[v].items() if b in movable) for v in vertices]
        pin = [max(0.0, min(1.0, 1 - w)) for w in free]
        regions.append({'id': identifier, 'role': definition['role'], 'materials': definition['materials'],
                        'deformation_policy': definition['deformation_policy'], 'face_ids': part['face_ids'],
                        'vertex_ids': vertices, 'pin_weights': pin,
                        'fixed_vertex_count': sum(p >= 1 - 1e-6 for p in pin),
                        'free_vertex_count': sum(p <= 1e-6 for p in pin),
                        'transition_vertex_count': sum(1e-6 < p < 1 - 1e-6 for p in pin),
                        'shared_boundary_vertex_count': sum(ownership[v] > 1 for v in vertices)})
    floor = profile['floor']
    if floor != {'axis': 'Z', 'height_m': 0.0, 'collision_group': 15}:
        raise ValueError('当前地面配置只支持Blender Z=0、组15；其他值需要显式扩展适配器')
    counts = Counter(x['reason'] for x in excluded)
    warnings = [{'code': 'excluded_native_dynamic_bones', 'counts': dict(counts)},
                {'code': 'hard_accessory_skinning_requires_review',
                 'note': '硬质饰品继续沿用源蒙皮；多骨骼绑定不等于刚性附件，步骤3/4需核对。'}]
    if any(c['requires_explicit_root_attachment'] for c in components):
        warnings.append({'code': 'unanchored_components', 'note': '求解前必须补根约束，当前不允许运行。'})
    floor_allowed = sum(mask_allows(rigids[i]['blender']['collision_group'],
                                   rigids[i]['blender']['excluded_groups_mask'], 15, 0) for i in dynamic_rigids)
    if floor_allowed < len(dynamic_rigids):
        warnings.append({'code': 'native_masks_exclude_floor', 'excluded_candidate_rigids': len(dynamic_rigids)-floor_allowed,
                         'note': '原生掩码予以保留；步骤4须显式配置地面接触策略，不能直接认为已防穿地。'})
    if unreferenced:
        warnings.append({'code': 'loose_vertices_kept_kinematic', 'count': len(unreferenced)})
    plan = {'schema_version': 2, 'profile_id': profile['id'], 'phase': 'model_binding_prepared',
            'enabled': False, 'simulation_performed': False, 'solver_ready': False,
            'model_sha256': digest, 'profile_sha256': hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
            'units': {'length': 'metre', 'scale': scale, 'position': '(x,z,y)*scale',
                      'rotation': '(-rx,-rz,-ry), Euler YXZ', 'mass_policy': 'author_value_no_density_rescale',
                      'spring_policy': 'MMD importer values, not calibrated SI stiffness'},
            'bindings': {**profile['bindings'], 'blend_asset': str(blend)},
            'bones': [{'id': i, 'name': b.name, 'blender_name': targets[i], 'parent': b.parent,
                       'rest_head_m': position(b.location, scale)} for i, b in enumerate(model.bones)],
            'primary_bones': profile['primary_bones'], 'protected_bone_ids': sorted(protected),
            'selected_secondary_bone_ids': sorted(selected), 'excluded_bones': excluded,
            'regions': regions, 'rigid_bodies': rigids, 'joints': joints, 'components': components,
            'floor': floor, 'collision_policy': {'mask_bits': 'excluded groups', 'pair_requires_both_allow': True,
                'actual_contacts_tested': False, 'floor_mask_allowed_candidates': floor_allowed,
                'engine_joint_exclusion_pending': True},
            'geometry': {'positions_m': [position(v.co, scale) for v in model.vertices],
                         'triangles': model.faces, 'face_material_ids': face_materials,
                         'material_names': [m.name for m in model.materials],
                         'skin_weights': [{str(k): v for k, v in s.items()} for s in skins],
                         'unreferenced_vertex_ids': sorted(unreferenced)},
            'warnings': warnings,
            'limitations': ['没有运行衣发动力学；固定区由现有蒙皮和受保护表面推导，未手工重新绘制权重。',
                            '当前只实现PMX配置适配器；其他模型需独立配置并通过绑定校验。',
                            '原生参数转换不代表稳定的物理求解参数；碰撞、软约束、预热留待后续。']}

    if 'surface_segmentation' in profile:
        from .secondary_segmentation import build_inventory, review_gate
        config = profile['surface_segmentation']
        if not isinstance(config, dict) or config.get('schema_version') != 1 or type(config.get('required_for_surface_dynamics')) is not bool:
            raise ValueError('表面分类配置版本或启用标记非法')
        inventory = build_inventory(plan)
        gate = review_gate(inventory, config.get('review'))
        plan['surface_segmentation'] = {'required_for_surface_dynamics':config['required_for_surface_dynamics'],
            'inventory':inventory, 'review_gate':gate, 'review':config.get('review')}
        if not gate['accepted']:
            plan['warnings'].append({'code':'surface_segmentation_review_pending',
                'note':'材质连通片与重合面只完成盘点；尚不能替代语义分割和接触/附着规则。'})
    if 'secondary_contact_rules' in profile:
        from .secondary_contact_policy import compile_contact_policy
        plan['secondary_contact_rules'] = compile_contact_policy(plan, profile)
    if 'source_surface_map' in profile:
        from .secondary_source_map import load_model_document, compile_source_map
        manifest = load_model_document(root, profile['source_surface_map'])
        plan['source_surface_map'] = compile_source_map(plan, profile['surface_segmentation']['review'], manifest)
        if 'source_connections' in profile:
            from .secondary_connections import compile_connections
            config = load_model_document(root, profile['source_connections'])
            plan['source_connections'] = compile_connections(plan, profile['surface_segmentation']['review'], manifest, config)
    if 'articulated_connections' in profile:
        from .secondary_source_map import load_model_document
        from .secondary_articulated import compile_articulated
        plan['articulated_connections'] = compile_articulated(plan, load_model_document(root, profile['articulated_connections']))
    if 'surface_mounts' in profile:
        from .secondary_surface_mounts import compile_mounts
        from .secondary_source_map import load_model_document
        manifest = load_model_document(root, profile['source_surface_map'])
        plan['surface_mounts'] = compile_mounts(plan, plan['surface_segmentation']['review'], manifest, load_model_document(root, profile['surface_mounts']))
        from .secondary_coupled_surface import compile_coupled_surface
        from .secondary_source_map import SourceSurfaceMap
        plan['coupled_surface'] = compile_coupled_surface(plan, SourceSurfaceMap(plan, plan['surface_segmentation']['review'], manifest))
    return plan


def prepare(profile_path, root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    fence = root / 'output/secondary_motion'
    if not output_dir.is_relative_to(fence) or output_dir == fence:
        raise ValueError('配置产物只能写入output/secondary_motion内的独立子目录')
    if output_dir.exists():
        raise FileExistsError('输出目录已存在，请使用新的目录名，防止覆盖')
    profile = json.loads(_inside(root, profile_path).read_text(encoding='utf-8-sig'))
    plan = compile_profile(profile, root)
    output_dir.mkdir(parents=True)
    (output_dir / 'model-plan.json').write_text(json.dumps(plan, ensure_ascii=False), encoding='utf-8')
    summary = {'profile': plan['profile_id'], 'simulation_performed': False,
               'selected_secondary_bones': len(plan['selected_secondary_bone_ids']),
               'protected_bones': len(plan['protected_bone_ids']),
               'excluded_native_dynamic_bones': len(plan['excluded_bones']),
               'rigid_bodies': len(plan['rigid_bodies']), 'joints': len(plan['joints']),
               'regions': [{k: v for k, v in r.items() if k not in ('face_ids', 'vertex_ids', 'pin_weights')} for r in plan['regions']],
               'warnings': plan['warnings'], 'plan': str(output_dir / 'model-plan.json')}
    if 'surface_segmentation' in plan:
        summary['surface_segmentation_review'] = plan['surface_segmentation']['review_gate']
    if 'secondary_contact_rules' in plan:
        summary['contact_rules'] = {k:v for k,v in plan['secondary_contact_rules'].items() if k != 'definition'}
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary
