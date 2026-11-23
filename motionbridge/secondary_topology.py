"""Read-only skin partitions and conservative primary-motion dependency closure."""
import math


def skin_weights(weight, bone_count):
    kind = weight.type
    if kind == 0:
        values = [1.0]
    elif kind in (1, 3):
        w = weight.weights[0] if kind == 1 else weight.weights.weight
        values = [w, 1 - w]
    elif kind == 2:
        values = list(weight.weights)
    else:
        raise ValueError('不支持的蒙皮类型')
    if len(values) != len(weight.bones) or any(not math.isfinite(w) or w < -1e-6 or w > 1.000001 for w in values):
        raise ValueError('非法蒙皮权重')
    if abs(sum(values) - 1) > 1e-5:
        raise ValueError('蒙皮权重之和不为1')
    result = {}
    for index, w in zip(weight.bones, values):
        if w <= 0:
            continue
        if type(index) is not int or not 0 <= index < bone_count:
            raise ValueError('蒙皮引用了不存在的骨骼')
        result[index] = result.get(index, 0.0) + w
    return result


def protected_closure(bones, primary_names):
    names = [b.name for b in bones]
    if len(set(names)) != len(names):
        raise ValueError('原生骨骼名称重复，无法安全绑定')
    if set(primary_names) - set(names):
        raise ValueError('模型缺少主动作骨骼')
    edges = [set() for _ in bones]

    def ref(index):
        if type(index) is not int or not 0 <= index < len(bones):
            raise ValueError('骨骼约束含非法引用')
        return index

    # Validate all parent chains; constraint couplings below may form valid cycles.
    for i, bone in enumerate(bones):
        current = i
        seen = set()
        while current not in (None, -1):
            ref(current)
            if current in seen:
                raise ValueError('骨骼父级链存在环')
            seen.add(current)
            current = bones[current].parent
        if bone.parent not in (None, -1):
            edges[i].add(ref(bone.parent))
        extra = getattr(bone, 'additionalTransform', None)
        if getattr(bone, 'hasAdditionalRotate', False) or getattr(bone, 'hasAdditionalLocation', False):
            if extra is None or not math.isfinite(extra[1]):
                raise ValueError('非法附加变换')
            donor = ref(extra[0])
            if extra[1] != 0:
                edges[i].add(donor)
        if getattr(bone, 'isIK', False):
            coupled = {i, ref(bone.target), *(ref(link.target) for link in bone.ik_links)}
            for member in coupled:
                edges[member].update(coupled - {member})
    protected = {i for i, name in enumerate(names) if name in primary_names}
    pending = list(protected)
    while pending:
        for dep in edges[pending.pop()] - protected:
            protected.add(dep)
            pending.append(dep)
    return protected


def material_partition(model, definitions):
    if len({m.name for m in model.materials}) != len(model.materials):
        raise ValueError('材质名重复，需先消歧')
    lookup = {}
    for region in definitions:
        if region['role'] not in ('hair', 'cloth', 'accessory', 'body', 'boots'):
            raise ValueError('未知网格角色')
        for name in region['materials']:
            if name in lookup:
                raise ValueError('材质被重复分区')
            lookup[name] = region['id']
    names = {m.name for m in model.materials}
    if set(lookup) != names:
        raise ValueError(f'材质分区不完整或引用不存在的材质: {sorted(names ^ set(lookup))}')
    if len({r['id'] for r in definitions}) != len(definitions):
        raise ValueError('分区ID重复')
    output = {r['id']: {'definition': r, 'vertices': set(), 'face_ids': []} for r in definitions}
    offset = 0
    face_materials = []
    for material_index, material in enumerate(model.materials):
        if material.vertex_count < 0 or material.vertex_count % 3:
            raise ValueError('材质三角面计数非法')
        end = offset + material.vertex_count // 3
        if end > len(model.faces):
            raise ValueError('材质面数超出网格')
        region = output[lookup[material.name]]
        region['face_ids'].extend(range(offset, end))
        for face in model.faces[offset:end]:
            if len(face) != 3 or any(not 0 <= v < len(model.vertices) for v in face):
                raise ValueError('三角面引用了不存在的顶点')
            region['vertices'].update(face)
        face_materials.extend([material_index] * (end - offset))
        offset = end
    if offset != len(model.faces):
        raise ValueError('材质面数未覆盖整个网格')
    return output, face_materials
