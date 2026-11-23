"""Model-owned secondary physics inspection. No simulation or body-animation writes.

The existing capture/retarget path stays independent. Native PMX data is read
strictly and reported before any physics engine is allowed to drive bones.
"""
from __future__ import annotations
from pathlib import Path
import collections
import hashlib
import json
import math
import struct


def _inside(root: Path, value: str | Path) -> Path:
    root = Path(root).resolve()
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError('角色物理配置只能引用当前开发副本内的文件。')
    return candidate


def validate_profile(profile: dict, root: Path) -> Path:
    if profile.get('schema_version') not in (1, 2):
        raise ValueError('不支持的角色物理配置版本。')
    if profile.get('enabled', False) is not False:
        raise NotImplementedError('当前仅实现物理配置检查，尚未提供动力学求解，不能启用仿真。')
    if profile.get('protect_primary_motion') is not True:
        raise ValueError('必须保护主身体动作。')
    names = profile.get('primary_bones')
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError('主动作骨骼列表必须非空且无重复。')
    model = profile.get('model', {})
    if model.get('format') != 'pmx':
        raise ValueError('当前只读检查器支持 PMX 原生物理数据。')
    scale = model.get('scale')
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('模型缩放必须为正的有限数值。')
    path = _inside(root, model.get('path', ''))
    if path.suffix.lower() != '.pmx' or not path.is_file():
        raise ValueError('模型 PMX 文件不存在。')
    return path


def _strict_pmx(path: Path):
    from .vendor import pmx_reader as pmx
    try:
        # Upstream convenience load() catches struct.error and returns a partial
        # model. Inspection must reject truncated physics instead of accepting it.
        with pmx.FileReadStream(str(path)) as stream:
            stream.strict_secondary_motion = True
            header = pmx.Header()
            header.load(stream)
            stream.setHeader(header)
            model = pmx.Model()
            model.load(stream)
            return model
    except (struct.error, EOFError, UnicodeError, pmx.InvalidFileError) as exc:
        raise ValueError('PMX 文件不完整或损坏，拒绝使用部分物理数据。') from exc


def _role(name: str, rules: dict) -> str:
    for role in ('hair', 'cloth', 'accessory'):
        if any(token in name for token in rules.get(role, [])):
            return role
    return 'unclassified'


def inspect_profile(profile: dict, root: Path) -> dict:
    path = validate_profile(profile, root)
    model = _strict_pmx(path)
    rules = profile.get('candidate_name_tokens', {})
    mode_names = {0: 'kinematic_follow', 1: 'dynamic', 2: 'dynamic_bone'}
    records = []
    issues = []
    requested = set(profile['primary_bones'])
    missing = sorted(requested - {bone.name for bone in model.bones})
    if missing:
        issues.append({'code': 'missing_primary_bones', 'bones': missing})
    protected_ancestors = set()
    for index, bone in enumerate(model.bones):
        if bone.name not in requested:
            continue
        seen = {index}
        parent = getattr(bone, 'parent', None)
        while parent not in (None, -1):
            if parent in seen or not 0 <= parent < len(model.bones):
                raise ValueError('主骨骼父级链存在环或非法引用。')
            seen.add(parent)
            ancestor = model.bones[parent]
            protected_ancestors.add(ancestor.name)
            parent = getattr(ancestor, 'parent', None)
    for index, rigid in enumerate(model.rigids):
        if rigid.bone is not None and not 0 <= rigid.bone < len(model.bones):
            raise ValueError('刚体引用了不存在的骨骼。')
        if rigid.mode not in mode_names:
            issues.append({'code': 'unsupported_rigid_mode', 'index': index, 'mode': rigid.mode})
        bone = model.bones[rigid.bone].name if rigid.bone is not None else None
        if bone in requested and rigid.mode != 0:
            issues.append({'code': 'dynamic_physics_on_primary_bone', 'rigid': rigid.name, 'bone': bone})
        elif bone in protected_ancestors and rigid.mode != 0:
            issues.append({'code': 'dynamic_physics_on_primary_ancestor', 'rigid': rigid.name, 'bone': bone})
        values = list(rigid.size) + list(rigid.location) + list(rigid.rotation) + [rigid.mass, rigid.velocity_attenuation, rigid.rotation_attenuation]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('物理数据含非有限数值。')
        records.append({'index': index, 'name': rigid.name, 'bone': bone, 'source_mode': rigid.mode,
            'mode': mode_names.get(rigid.mode, 'unsupported'), 'candidate_role': _role(rigid.name, rules),
            'shape': rigid.type, 'source_size': rigid.size, 'source_position': rigid.location,
            'source_rotation': rigid.rotation, 'source_mass': rigid.mass,
            'linear_damping': rigid.velocity_attenuation, 'angular_damping': rigid.rotation_attenuation,
            'collision_group': rigid.collision_group_number, 'collision_mask': rigid.collision_group_mask})
    joints = []
    for index, joint in enumerate(model.joints):
        for endpoint in (joint.src_rigid, joint.dest_rigid):
            if endpoint is not None and not 0 <= endpoint < len(model.rigids):
                raise ValueError('关节引用了不存在的刚体。')
        joints.append({'index': index, **vars(joint)})
    return {'schema_version': 1, 'profile_id': profile['id'], 'phase': 'profile_inspection',
        'simulation_performed': False, 'enabled': False,
        'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'model_path': str(path), 'model_scale': profile['model']['scale'],
        'source_units_preserved': True, 'coordinate_conversion_applied': False,
        'source_bones': len(model.bones), 'source_morphs': len(model.morphs),
        'native_physics': {'rigid_body_count': len(records), 'joint_count': len(joints),
            'mode_counts': dict(collections.Counter(row['mode'] for row in records)),
            'candidate_role_counts': dict(collections.Counter(row['candidate_role'] for row in records)),
            'rigid_bodies': records, 'joints': joints},
        'primary_motion_protection': {'required': True, 'protected_bones': profile['primary_bones'],
            'protected_ancestor_bones': sorted(protected_ancestors),
            'missing': missing, 'inspection_is_read_only': True},
        'blocking_issues': issues,
        'limitations': ['仅检查模型原生物理数据，尚未运行求解器或烘焙。',
            '候选衣发分类仅按显式名称标记，不等于已识别完整网格分区。',
            '原PMX单位和物理参数尚未转换为Blender求解参数。',
            '启用求解前还须检查MMD附加变换等非父级约束，并实测主骨骼世界姿态不变。',
            '不修改源模型、主身体动画或旧任务。']}


def load_and_inspect(profile_path: Path, root: Path) -> dict:
    profile_path = _inside(root, profile_path)
    profile = json.loads(profile_path.read_text(encoding='utf-8-sig'))
    return inspect_profile(profile, root)
