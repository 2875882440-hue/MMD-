"""Rigid head accessory/front hair surfaces repel arms, never deform ornaments."""
import numpy as np
from .body_collision import _unit,_smooth,_cap_direction
from .head_collision import avoid_arm_head


def accessory_surface(mesh,rig,head_bone='頭'):
    import hashlib,json
    from mathutils.bvhtree import BVHTree
    names={'前髮','头饰'}
    material_ids={i for i,m in enumerate(mesh.data.materials) if m and m.name in names}
    groups={g.index:g.name for g in mesh.vertex_groups}
    rigid={v.index for v in mesh.data.vertices if sum(g.weight for g in v.groups if groups.get(g.group)==head_bone)>.95}
    faces=[tuple(p.vertices) for p in mesh.data.polygons if p.material_index in material_ids and all(i in rigid for i in p.vertices)]
    if not faces:raise ValueError('Missing configured rigid hair/accessory surface')
    transform=rig.matrix_world.inverted()@mesh.matrix_world
    vertices=[tuple(transform@v.co) for v in mesh.data.vertices]
    digest=hashlib.sha256(json.dumps({'faces':faces,'vertices':{i:vertices[i] for i in sorted({i for f in faces for i in f})}},sort_keys=True).encode()).hexdigest()
    return BVHTree.FromPolygons(vertices,faces,all_triangles=False),{'materials':sorted(names),'polygons':len(faces),'head_weight_threshold':.95,'rest_geometry_sha256':digest,
        'note':'Unsigned arm-to-rigid-surface distance; not inside/outside volume collision. Rigid accessory geometry and binding stay unchanged.'}


def surface_vectors(tree,samples,radius):
    from mathutils import Vector
    force=np.zeros_like(samples);depth=np.zeros(samples.shape[:-1])
    flat=samples.reshape(-1,3);out=force.reshape(-1,3);d=depth.ravel()
    for i,p in enumerate(flat):
        location,normal,_,distance=tree.find_nearest(Vector(p),radius)
        if location is None:continue
        vector=p-np.asarray(location);length=np.linalg.norm(vector)
        direction=vector/length if length>1e-8 else np.asarray(normal)
        out[i]=direction*(radius-distance);d[i]=radius-distance
    return force,depth


def avoid_accessories(points,head_matrices,head_rest,tree,radius,fps,torso,head_proxy):
    points=np.asarray(points,float);delta=np.asarray(head_matrices,float)@np.linalg.inv(head_rest)
    rotation=delta[:,:3,:3];translation=delta[:,:3,3]
    local=np.einsum('fij,fkj->fki',rotation.swapaxes(1,2),points-translation[:,None])
    lengths=np.linalg.norm(np.diff(local,axis=1),axis=-1,keepdims=True)
    a=local[:,0];u0=_unit(local[:,1]-a);v0=_unit(local[:,2]-local[:,1]);u=u0.copy();v=v0.copy()
    alpha=np.r_[np.linspace(.25,1,7),np.ones(9)];beta=np.r_[np.zeros(7),np.linspace(0,1,9)]
    def samples(u,v):return a[:,None]+alpha[None,:,None]*lengths[:,0,None]*u[:,None]+beta[None,:,None]*lengths[:,1,None]*v[:,None]
    _,before=surface_vectors(tree,samples(u,v),radius)
    # Restrict nearest-surface queries to collision neighbourhoods.
    active=(before.max(1)>.00001).astype(float)
    for _ in range(3):active=np.maximum(active,_smooth(active,fps))
    active=active>1e-8
    if active.any():
        for _ in range(35):
            force=np.zeros((len(points),len(alpha),3));force[active]=surface_vectors(tree,samples(u,v)[active],radius)[0]
            gu=(force*alpha[None,:,None]).sum(1)/4;gv=(force*beta[None,:,None]).sum(1)/4
            u=_cap_direction(u+.6*(gu-u*np.sum(gu*u,1,keepdims=True))/lengths[:,0]-.008*(u-u0),u0,np.deg2rad(8))
            v=_cap_direction(v+.6*(gv-v*np.sum(gv*v,1,keepdims=True))/lengths[:,1]-.008*(v-v0),v0,np.deg2rad(8))
            u=_cap_direction(u0+_smooth(u-u0,fps),u0,np.deg2rad(8));v=_cap_direction(v0+_smooth(v-v0,fps),v0,np.deg2rad(8))
        solved=local.copy();solved[:,1]=a+lengths[:,0]*u;solved[:,2]=solved[:,1]+lengths[:,1]*v
        candidate=np.einsum('fij,fkj->fki',rotation,solved)+translation[:,None]
        candidate[~active]=points[~active]
    else:candidate=points.copy()
    # Existing head/torso layers take priority over ornament clearance.
    out,guard=avoid_arm_head(points,*head_proxy,radius,fps,torso,candidate=candidate)
    final_local=np.einsum('fij,fkj->fki',rotation.swapaxes(1,2),out-translation[:,None])
    final_samples=final_local[:,0,None]+alpha[None,:,None]*(final_local[:,1]-final_local[:,0])[:,None]+beta[None,:,None]*(final_local[:,2]-final_local[:,1])[:,None]
    _,after=surface_vectors(tree,final_samples,radius)
    return out,{'before_frames_over_1mm':int((before.max(1)>.001).sum()),'after_frames_over_1mm':int((after.max(1)>.001).sum()),
        'before_max_surface_overlap_m':float(before.max()),'after_max_surface_overlap_m':float(after.max()),
        'before_sum_surface_overlap_m':float(before.sum()),'after_sum_surface_overlap_m':float(after.sum()),
        'segment_limit_deg':8.,'guard':guard}
