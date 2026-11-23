"""Explicit named-profile face calibration. No automatic arbitrary-rig mapping."""
import numpy as np


def calibrate_face_proxy(rig,mesh,material_names,head_bone):
    material_ids={i for i,m in enumerate(mesh.data.materials) if m and m.name in material_names}
    groups={g.index:g.name for g in mesh.vertex_groups}
    weighted={v.index for v in mesh.data.vertices if sum(g.weight for g in v.groups if groups.get(g.group)==head_bone)>.5}
    indices=sorted({i for p in mesh.data.polygons if p.material_index in material_ids for i in p.vertices}&weighted)
    if len(indices)<30:raise ValueError('Profile face geometry is missing; cannot calibrate head avoidance')
    transform=rig.matrix_world.inverted()@mesh.matrix_world
    points=np.array([list(transform@mesh.data.vertices[i].co) for i in indices],dtype=float)
    low=points.min(0);high=points.max(0);center=(low+high)*.5;radii=(high-low)*.5
    if not np.isfinite(points).all() or np.min(radii)<.005:raise ValueError('Invalid face dimensions')
    return {'center_rest':center.tolist(),'radii_m':radii.tolist(),'mesh':mesh.name,'materials':list(material_names),
        'vertices':len(indices),'head_bone':head_bone,
        'note':'Rest face AABB inscribed ellipsoid. Does not enclose all face vertices or include hair/accessories; not an exact face mesh.'}


def animated_head_proxy(matrices,rest_matrix,profile):
    delta=np.asarray(matrices,dtype=float)@np.linalg.inv(np.asarray(rest_matrix,dtype=float))
    centers=delta[:,:3,:3]@np.asarray(profile['center_rest'])+delta[:,:3,3]
    return centers,delta[:,:3,:3],np.asarray(profile['radii_m'])
