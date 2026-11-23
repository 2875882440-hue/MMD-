"""Evaluated, weighted foot vertices for an explicitly configured Blender rig."""
import numpy as np


def foot_vertex_groups(scene, rig, tokens=('足首','つま先')):
    samples=[]
    for obj in scene.objects:
        if obj.type!='MESH' or obj.hide_render or 'contour' in obj.name:continue
        if not any(m.type=='ARMATURE' and m.object==rig for m in obj.modifiers):continue
        group_names={g.index:g.name for g in obj.vertex_groups}
        indices=[v.index for v in obj.data.vertices if sum(g.weight for g in v.groups
            if any(token in group_names.get(g.group,'') for token in tokens))>.1]
        if indices:samples.append((obj,np.array(indices,dtype=int)))
    if not samples:raise ValueError('Configured character has no weighted foot vertices')
    return samples


def evaluated_foot_minimum(samples,depsgraph):
    minimum=float('inf')
    for obj,indices in samples:
        evaluated=obj.evaluated_get(depsgraph);mesh=evaluated.to_mesh()
        try:
            if len(mesh.vertices)!=len(obj.data.vertices):raise ValueError('Foot audit requires topology-preserving modifiers')
            coords=np.empty(len(mesh.vertices)*3);mesh.vertices.foreach_get('co',coords)
            coords=coords.reshape(-1,3)[indices];matrix=np.asarray(evaluated.matrix_world)
            minimum=min(minimum,float((coords@matrix[2,:3]+matrix[2,3]).min()))
        finally:evaluated.to_mesh_clear()
    if not np.isfinite(minimum):raise ValueError('Nonfinite evaluated foot geometry')
    return minimum
