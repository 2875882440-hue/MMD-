"""Filter a baked MMD vertex track in animated head space, preserving nearby contact surfaces.
Run in Blender background Python; input matrices correspond one-to-one with cache samples.
"""
import numpy as np
from mathutils.bvhtree import BVHTree

def stabilize(points, head, plan, head_id, sample_hz):
    faces=np.asarray(plan['geometry']['triangles'],int)
    hair_faces=np.unique([f for region in plan['regions'] if region['role']=='hair' for f in region['face_ids']])
    other=faces[~np.isin(np.arange(len(faces)),hair_faces)]
    hair=np.unique(faces[hair_faces]);tri=np.searchsorted(hair,faces[hair_faces])
    geo=plan['geometry']; selected=set(plan['selected_secondary_bone_ids'])-{head_id}
    weights=geo['skin_weights']
    influence=np.array([sum(w for b,w in weights[v].items() if int(b) in selected) for v in hair]).clip(0,1)
    influence[np.isin(hair,np.unique(other))]=0
    # Split UV vertices must receive the same attenuation.
    alias={}; groups=[]
    for v in hair:
        key=(tuple(np.round(geo['positions_m'][v],6)),tuple(sorted(weights[v].items())))
        groups.append(alias.setdefault(key,len(alias)))
    group=np.array(groups);ng=len(alias)
    inf=np.ones(ng);np.minimum.at(inf,group,influence);influence=inf[group]
    pos=np.asarray(points[:,hair],dtype=float)
    linear=head[:,:3,:3];origin=head[:,:3,3]
    local=(pos-origin[:,None,:]) @ np.linalg.inv(linear).transpose(0,2,1)
    # Symmetric kernel avoids phase delay, ends retain their input poses.
    sigma=sample_hz*.035;radius=max(1,round(3*sigma));k=np.arange(-radius,radius+1)
    kernel=np.exp(-.5*(k/sigma)**2);kernel/=kernel.sum()
    smooth=np.zeros_like(local)
    for offset,fac in zip(k,kernel):
        smooth+=fac*local[np.clip(np.arange(len(local))+offset,0,len(local)-1)]
    delta=((smooth-local) @ linear.transpose(0,2,1))*influence[None,:,None]
    # A conservative triangle envelope includes any interpolation between input and proposal.
    allowed=np.ones((len(local),ng));used=np.unique(other);other_tri=np.searchsorted(used,other).tolist()
    for i in range(len(local)):
        collider=BVHTree.FromPolygons(points[i,used].tolist(),other_tri,all_triangles=True)
        tris=pos[i,tri];center=tris.mean(axis=1)
        rad=np.maximum(np.linalg.norm(tris-center[:,None],axis=2).max(1),
                       np.linalg.norm(tris+delta[i,tri]-center[:,None],axis=2).max(1))
        blocked=[j for j,(c,r) in enumerate(zip(center,rad)) if collider.find_nearest(c,float(r+2e-6))[0] is not None]
        if blocked:allowed[i,group[tri[blocked].ravel()]]=0
    # Fade from the contact boundary over three edge rings and three display frames.
    edges=np.unique(np.sort(group[tri[:,[0,1,1,2,2,0]].reshape(-1,2)],axis=1),axis=0)
    for _ in range(3):
        a,b=edges.T;old=allowed.copy()
        for i in range(len(local)):
            np.minimum.at(allowed[i],a,old[i,b]+.25)
            np.minimum.at(allowed[i],b,old[i,a]+.25)
    allowed[[0,-1]]=0;rise=1/max(1.,sample_hz*.10)
    for i in range(1,len(local)):np.minimum(allowed[i],allowed[i-1]+rise,out=allowed[i])
    for i in range(len(local)-2,-1,-1):np.minimum(allowed[i],allowed[i+1]+rise,out=allowed[i])
    revised=(pos+delta*allowed[:,group,None]).astype(points.dtype)
    return hair,revised,local,(revised-origin[:,None,:]) @ np.linalg.inv(linear).transpose(0,2,1),influence,tri
