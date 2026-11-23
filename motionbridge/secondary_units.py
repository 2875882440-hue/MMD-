"""PMX -> Blender adapter, matching MMD Tools 4.5.14 importer semantics.

Length is converted to metres at the profile scale. Mass, damping and spring
values are author/import values, NOT a calibrated SI material model.
"""
import math


def vector(value):
    if len(value) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError('物理向量必须包含三个有限数值')
    return list(value)


def axes(value):
    x, y, z = vector(value)
    return [x, z, y]


def position(value, scale):
    return [v * scale for v in axes(value)]


def rotation(value):
    return [-v for v in axes(value)]


def mask_allows(a_group, a_mask, b_group, b_mask):
    for group in (a_group, b_group):
        if type(group) is not int or not 0 <= group < 16:
            raise ValueError('碰撞组必须在0至15之间')
    for mask in (a_mask, b_mask):
        if type(mask) is not int or not 0 <= mask <= 65535:
            raise ValueError('碰撞排除掩码必须为16位非负整数')
    return not (a_mask & (1 << b_group) or b_mask & (1 << a_group))


def rigid_parameters(rigid, scale):
    size = vector(rigid.size)
    if rigid.type not in (0, 1, 2) or any(v < 0 for v in size):
        raise ValueError('不支持的刚体形状或负尺寸')
    shape = ({'kind': 'sphere', 'radius_m': size[0] * scale} if rigid.type == 0 else
             {'kind': 'box', 'half_extents_m': position(size, scale)} if rigid.type == 1 else
             {'kind': 'capsule', 'radius_m': size[0] * scale,
              'cylinder_height_m': size[1] * scale, 'total_height_m': (size[1] + 2 * size[0]) * scale})
    mask_allows(rigid.collision_group_number, rigid.collision_group_mask, 0, 0)
    scalars = [rigid.mass, rigid.velocity_attenuation, rigid.rotation_attenuation, rigid.bounce, rigid.friction]
    if any(not math.isfinite(v) or v < 0 for v in scalars):
        raise ValueError('刚体物理参数含负值或非有限数值')
    return {'position_m': position(rigid.location, scale), 'rotation_rad': rotation(rigid.rotation),
            'rotation_order': 'YXZ', 'shape': shape, 'mass_import_value': rigid.mass,
            'linear_damping': rigid.velocity_attenuation, 'angular_damping': rigid.rotation_attenuation,
            'bounce': rigid.bounce, 'friction': rigid.friction,
            'collision_group': rigid.collision_group_number,
            'excluded_groups_mask': rigid.collision_group_mask,
            'allowed_groups': [i for i in range(16) if not rigid.collision_group_mask & (1 << i)]}


def joint_parameters(joint, scale):
    if joint.mode != 0:
        raise ValueError('当前配置编译器只支持PMX弹簧六自由度关节')
    result = {'position_m': position(joint.location, scale), 'rotation_rad': rotation(joint.rotation),
              'rotation_order': 'YXZ', 'linear_min_m': position(joint.minimum_location, scale),
              'linear_max_m': position(joint.maximum_location, scale),
              'angular_min_rad': rotation(joint.maximum_rotation),
              'angular_max_rad': rotation(joint.minimum_rotation),
              'linear_spring_import_value': axes(joint.spring_constant),
              'angular_spring_import_value': axes(joint.spring_rotation_constant)}
    for low, high in (('linear_min_m', 'linear_max_m'), ('angular_min_rad', 'angular_max_rad')):
        if any(a > b + 1e-7 for a, b in zip(result[low], result[high])):
            raise ValueError('关节下限大于上限')
    return result
