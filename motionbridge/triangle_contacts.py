"""Bounded corrective contact with actual triangles; no cloth dynamics.

Geometry routines are independent of Blender. A nearest-query adapter supplies
the current closest point and triangle index. Signed local surface constraints
do not establish watertight inside/outside classification.
"""
import numpy as np


def geometry_reference_error(points,reference,tolerance=1e-5):
    """All-vertex Euclidean fidelity, independent of threshold contact counts."""
    points=np.asarray(points,float);reference=np.asarray(reference,float)
    if points.shape!=reference.shape or points.ndim!=2 or points.shape[1]!=3 or not len(points):
        raise ValueError('Invalid geometry reference shape')
    if not np.isfinite(points).all() or not np.isfinite(reference).all() or not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError('Invalid geometry reference values')
    error=float(np.linalg.norm(points-reference,axis=1).max())
    return error,error<=tolerance


def proper_triangle_crossings(left,right,tolerance=1e-5):
    """Confirm non-coplanar edge-through-face crossings after BVH broad phase.

    Coplanar overlap, grazing contact and shared edges are not proper crossings.
    This is a discrete surface test, not a closed-volume or continuous solver.
    """
    left=np.asarray(left,float);right=np.asarray(right,float)
    if left.shape!=right.shape or left.ndim!=3 or left.shape[1:]!=(3,3) or not np.isfinite(left).all() or not np.isfinite(right).all() or not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('Invalid triangle crossing pairs')
    result=np.zeros(len(left),bool)
    for source,target in [(left,right),(right,left)]:
        e1=target[:,1]-target[:,0];e2=target[:,2]-target[:,0]
        normal=np.cross(e1,e2);normal_length=np.linalg.norm(normal,axis=1)
        for a,b in [(0,1),(1,2),(2,0)]:
            start=source[:,a];direction=source[:,b]-start
            h=np.cross(direction,e2);det=np.sum(e1*h,axis=1)
            scale=np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)*np.linalg.norm(direction,axis=1)
            valid=(normal_length>1e-14)&(abs(det)>np.maximum(scale*1e-12,1e-30))
            inverse=np.divide(1.,det,out=np.zeros_like(det),where=valid)
            relative=start-target[:,0];u=np.sum(relative*h,axis=1)*inverse
            q=np.cross(relative,e1);v=np.sum(direction*q,axis=1)*inverse;t=np.sum(e2*q,axis=1)*inverse
            d0=np.sum(relative*normal,axis=1)/np.maximum(normal_length,1e-30)
            d1=np.sum((source[:,b]-target[:,0])*normal,axis=1)/np.maximum(normal_length,1e-30)
            straddles=((d0>tolerance)&(d1<-tolerance))|((d1>tolerance)&(d0<-tolerance))
            result |= valid&straddles&(u>1e-7)&(v>1e-7)&(u+v<1-1e-7)&(t>1e-7)&(t<1-1e-7)
    return result


def closest_triangle(points, triangles):
    """Closest points for paired points/triangles, including degenerate faces."""
    p=np.asarray(points,float);t=np.asarray(triangles,float)
    if p.ndim!=2 or p.shape[1]!=3 or t.shape!=(len(p),3,3) or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Invalid point/triangle pairs')
    a,b,c=t[:,0],t[:,1],t[:,2];ab=b-a;ac=c-a;ap=p-a
    aa=np.sum(ab*ab,1);bb=np.sum(ab*ac,1);cc=np.sum(ac*ac,1)
    d=np.sum(ap*ab,1);e=np.sum(ap*ac,1);den=aa*cc-bb*bb
    u=(cc*d-bb*e)/np.maximum(den,1e-30);v=(aa*e-bb*d)/np.maximum(den,1e-30)
    plane=a+u[:,None]*ab+v[:,None]*ac
    inside=(den>1e-24)&(u>=0)&(v>=0)&(u+v<=1)
    candidates=[]
    for start,end in [(a,b),(b,c),(c,a)]:
        axis=end-start;fraction=np.clip(np.sum((p-start)*axis,1)/np.maximum(np.sum(axis*axis,1),1e-30),0,1)
        candidates.append(start+fraction[:,None]*axis)
    candidates=np.stack(candidates,axis=1)
    nearest=candidates[np.arange(len(p)),np.argmin(np.sum((candidates-p[:,None])**2,2),1)]
    return np.where(inside[:,None],plane,nearest)


def contact_samples(rest, edges, faces):
    """Weld duplicate corners for querying; include edge and face interiors."""
    rest=np.asarray(rest,float);edges=np.asarray(edges,int);faces=np.asarray(faces,int)
    if rest.ndim!=2 or rest.shape[1]!=3 or edges.ndim!=2 or edges.shape[1]!=2 or faces.ndim!=2 or faces.shape[1]!=3:
        raise ValueError('Invalid contact mesh')
    _,first,group=np.unique(np.round(rest,6),axis=0,return_index=True,return_inverse=True)
    we=np.unique(np.sort(group[edges],axis=1),axis=0);we=we[we[:,0]!=we[:,1]]
    wf=np.unique(np.sort(group[faces],axis=1),axis=0)
    wf=wf[(wf[:,0]!=wf[:,1])&(wf[:,1]!=wf[:,2])]
    ids=np.concatenate([np.repeat(first[:,None],3,axis=1),first[np.c_[we,we[:,1]]],first[wf]])
    weights=np.concatenate([np.tile([1.,0,0],(len(first),1)),np.tile([.5,.5,0],(len(we),1)),np.full((len(wf),3),1/3)])
    return ids,weights,group


def sample_points(points, ids, weights):
    return np.sum(np.asarray(points)[ids]*weights[:,:,None],axis=1)


def surface_contacts(points, rest_points, rest_triangles, triangles, nearest, gap=.0015, radius=.03, rest_order=False):
    """Preserve close intentional rest overlaps with the *queried* triangle.

    A rest overlap on the back of the head must not exempt a new arm contact.
    Distant rest triangles grant no overlap allowance. Ambiguous/degenerate
    triangle normals are skipped rather than inventing an escape direction.
    """
    if not np.isfinite([gap,radius]).all() or gap<0 or radius<=gap:
        raise ValueError('Invalid contact distances')
    points=np.asarray(points,float);rest_points=np.asarray(rest_points,float)
    query,face=nearest(points,radius);face=np.asarray(face,int)
    valid=face>=0;normals=np.zeros_like(points);depth=np.zeros(len(points))
    if not valid.any():return depth,normals
    ids=np.flatnonzero(valid);face=face[valid];tri=np.asarray(triangles)[face]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(normal,axis=1)
    normal/=np.maximum(length[:,None],1e-20)
    rtri=np.asarray(rest_triangles)[face]
    rn=np.cross(rtri[:,1]-rtri[:,0],rtri[:,2]-rtri[:,0]);rlen=np.linalg.norm(rn,axis=1)
    rn/=np.maximum(rlen[:,None],1e-20)
    rq=closest_triangle(rest_points[ids],rtri);rd=rest_points[ids]-rq
    signed_rest=np.sum(rd*rn,1)
    threshold=np.where(np.linalg.norm(rd,axis=1)<=radius,np.minimum(gap,signed_rest),gap)
    displacement=points[ids]-np.asarray(query)[ids]
    signed=np.sum(displacement*normal,1)
    # A nearest point on an open boundary cannot establish which volume side
    # the query occupies. Only constrain a local face-interior projection.
    supported=np.linalg.norm(displacement-signed[:,None]*normal,axis=1)<1e-6
    if rest_order:
        side=np.where(signed_rest>=0,1.,-1.)
        normal*=side[:,None];signed*=side
        threshold=np.minimum(gap,abs(signed_rest))
        # Existing attached/intersecting layers have no reliable rest ordering.
        supported &= (np.linalg.norm(rd,axis=1)>gap)&(abs(signed_rest)>1e-5)
    depth[ids]=np.where(supported&(length>1e-12)&(rlen>1e-12),np.maximum(threshold-signed,0),0)
    normals[ids]=normal
    return depth,normals


def scatter_corrections(depth,normals,ids,weights,groups,edges,step=.004):
    """Distribute interior contacts to welded vertices and diffuse locally."""
    force=normals*np.minimum(depth,step)[:,None]
    group_ids=groups[ids];count=int(groups.max())+1
    denominator=np.bincount(group_ids.ravel(),weights=(weights**2).ravel(),minlength=count)
    move=np.stack([np.bincount(group_ids.ravel(),weights=(force[:,k,None]*weights).ravel(),minlength=count)
                   for k in range(3)],axis=1)/np.maximum(denominator[:,None],1e-12)
    # Contact-free sample rows must not cancel a sparse contact through enormous
    # sample valence; their zero contribution is included consistently.
    ge=np.unique(np.sort(groups[edges],axis=1),axis=0);a,b=ge.T
    degree=np.bincount(ge.ravel(),minlength=count)
    for _ in range(6):
        neighbour=np.stack([np.bincount(a,weights=move[b,k],minlength=count)+np.bincount(b,weights=move[a,k],minlength=count)
                            for k in range(3)],axis=1)/np.maximum(degree[:,None],1)
        move=.6*move+.4*neighbour
    return move[groups]


def bound_surface(raw,baseline,candidate,extra=.02,total=.12):
    delta=candidate-baseline
    p=baseline+delta*np.minimum(1,extra/np.maximum(np.linalg.norm(delta,axis=1),1e-20))[:,None]
    delta=p-raw
    return raw+delta*np.minimum(1,total/np.maximum(np.linalg.norm(delta,axis=1),1e-20))[:,None]


def preserve_capsule_clearance(raw,baseline,candidate,caps,limits,extra=.03,total=.12,iterations=24):
    """Project onto baseline capsule support planes and two movement balls.

    A support plane of each convex capsule retains its baseline escape side.
    Unlike radial reprojection, interpolation cannot choose its opposite side.
    The baseline is feasible, including its explicitly reported old overlaps.
    Residual numerical conflicts are still checked by whole-component gates.
    """
    from .surface_clearance import capsule_distance
    normals=[];minimum=[]
    for cap,limit in zip(caps,limits):
        distance,vector=capsule_distance(baseline,*cap)
        normals.append(vector/np.maximum(distance[:,None],1e-20))
        minimum.append(np.minimum(limit,distance)-distance)
    p=bound_surface(raw,baseline,candidate,extra,total)
    for _ in range(iterations):
        for normal,threshold in zip(normals,minimum):
            amount=np.maximum(threshold-np.sum((p-baseline)*normal,axis=1),0)
            p+=normal*amount[:,None]
        p=bound_surface(raw,baseline,p,extra,total)
    return p


def regularize_contact_delta(raw,baseline,candidate,caps,limits,edges,groups,iterations=48):
    """Bound new strain while relaxing old strain, with fixed escape planes."""
    from .surface_clearance import capsule_distance
    a,b=edges.T;degree=np.maximum(np.bincount(edges.ravel(),minlength=len(raw)),1)
    rest_length=np.linalg.norm(raw[b]-raw[a],axis=1)
    old_error=abs(np.linalg.norm(baseline[b]-baseline[a],axis=1)-rest_length)
    allowance=np.maximum(old_error-5e-5,.0005)
    normals=[];minimum=[]
    for cap,limit in zip(caps,limits):
        distance,vector=capsule_distance(baseline,*cap)
        normals.append(vector/np.maximum(distance[:,None],1e-20))
        minimum.append(np.minimum(limit,distance)-distance)
    counts=np.bincount(groups);p=bound_surface(raw,baseline,candidate,extra=.03,total=.12)
    for _ in range(iterations):
        vector=p[b]-p[a];length=np.linalg.norm(vector,axis=1)
        target=np.clip(length,np.maximum(0,rest_length-allowance),rest_length+allowance)
        force=vector*((length-target)/np.maximum(length,1e-20))[:,None]
        move=np.stack([np.bincount(a,weights=force[:,k],minlength=len(p))-np.bincount(b,weights=force[:,k],minlength=len(p)) for k in range(3)],axis=1)/degree[:,None]
        p+=.8*move
        delta=p-baseline
        delta=np.stack([np.bincount(groups,weights=delta[:,k])/counts for k in range(3)],axis=1)[groups]
        p=baseline+delta
        for normal,threshold in zip(normals,minimum):
            p+=normal*np.maximum(threshold-np.sum((p-baseline)*normal,axis=1),0)[:,None]
        p=bound_surface(raw,baseline,p,extra=.03,total=.12)
    return p


def support_safe_track(baseline,candidate,caps,limits,edges,rest):
    """One smooth spatial weight per welded vertex for the entire sequence.

    Analytic capsule support bounds remove residual projection errors. Taking
    the minimum over time avoids a weight toggling on/off at a contact frame;
    diffuse only downward across the rest mesh to preserve every bound.
    """
    from .surface_clearance import capsule_distance
    _,groups=np.unique(np.round(rest,6),axis=0,return_inverse=True)
    ceiling=np.ones(len(rest))
    for f in range(len(baseline)):
        delta=candidate[f]-baseline[f]
        for cap,limit in zip(caps[f],limits):
            distance,vector=capsule_distance(baseline[f],*cap)
            normal=vector/np.maximum(distance[:,None],1e-20)
            amount=np.sum(delta*normal,axis=1)
            margin=distance-np.minimum(limit,distance)
            bound=np.where(amount < -1e-14,margin/np.maximum(-amount,1e-20),1.)
            np.minimum(ceiling,np.clip(bound,0,1),out=ceiling)
    weights=np.ones(groups.max()+1);np.minimum.at(weights,groups,ceiling)
    ge=np.unique(np.sort(groups[edges],axis=1),axis=0);a,b=ge.T
    degree=np.maximum(np.bincount(ge.ravel(),minlength=len(weights)),1)
    for _ in range(6):
        neighbour=(np.bincount(a,weights=weights[b],minlength=len(weights))+np.bincount(b,weights=weights[a],minlength=len(weights)))/degree
        weights=np.minimum(weights,.5*weights+.5*neighbour)
    alpha=weights[groups]
    return baseline+(candidate-baseline)*alpha[None,:,None],{
        'method':'whole_sequence_support_ceiling_spatial_fade_v1',
        'zero_weight_vertices':int((alpha<=1e-12).sum()),'vertices':len(rest),
        'minimum_weight':float(alpha.min()),'mean_weight':float(alpha.mean()),
        'time_varying_weights':False}


def contact_track_gate(raw,baseline,candidate,caps,limits,edges,fps,cache=None,quick=False):
    """New surface constraints may add <=0.5mm error to an already clean edge.

    Every edge may add at most 0.5mm error relative to the accepted baseline.
    This permits real contact on previously clean cloth rather than requiring
    literally zero deformation energy forever. The budget is reported.
    """
    from .surface_directed import track_metrics
    from .surface_clearance import deficits
    from .surface_topology import edge_changes
    before=cache['metrics'] if cache else track_metrics(raw,baseline,caps,limits,edges,fps)
    reasons=[];new=0;added=0.;excess=0.;energy_budget=0.
    for f in range(len(raw)):
        old=cache['depth'][f] if cache else deficits(baseline[f],caps[f],limits).max(0)
        cur=deficits(candidate[f],caps[f],limits).max(0)
        new+=int(((cur>.001)&(old<=.001)).sum())
        old_edge=cache['edges'][f] if cache else edge_changes(baseline[f],raw[f],edges)
        current_edge=edge_changes(candidate[f],raw[f],edges)
        added=max(added,float(np.maximum(current_edge-old_edge,0).max(initial=0)))
        allowance=old_edge+.0005
        excess=max(excess,float(np.maximum(current_edge-allowance,0).max(initial=0)))
        energy_budget+=float(np.sum((allowance+5e-6)**2))
        if quick and (new or added>.000505 or excess>5e-6):
            reasons=(['new_capsule_contacts'] if new else [])+(['incremental_edge_error'] if added>.000505 else [])+(['existing_edge_error'] if excess>5e-6 else [])
            return {'accepted':False,'rejection_reasons':reasons,'before':before,'quick_rejection':True}
    after=track_metrics(raw,candidate,caps,limits,edges,fps)
    if not np.isfinite(candidate).all():reasons.append('non_finite')
    if after['maximum_shift_m']>.12+5e-7:reasons.append('total_shift')
    if np.linalg.norm(candidate-baseline,axis=-1).max()>.03+5e-7:reasons.append('extra_shift')
    if new:reasons.append('new_capsule_contacts')
    if after['max_penetration_m']>before['max_penetration_m']+1e-5:reasons.append('capsule_depth')
    if added>.000505:reasons.append('incremental_edge_error')
    if excess>5e-6:reasons.append('existing_edge_error')
    if after['edge_energy']>energy_budget+1e-7:reasons.append('edge_energy_budget')
    if after['max_edge_change_m']>before['max_edge_change_m']+.000505:reasons.append('peak_edge_error')
    if after['edge_frames_over_5mm']>before['edge_frames_over_5mm']:reasons.append('large_edge_count')
    for name in ['vertex_acceleration_p99','vertex_acceleration_max']:
        if after[name]>before[name]+.1:reasons.append(name)
    return {'accepted':not reasons,'rejection_reasons':reasons,'before':before,'candidate':after,
            'max_incremental_edge_error_m':added,'max_edge_error_above_allowance_m':excess,
            'edge_energy_budget_m2':energy_budget,'new_capsule_contacts':new}


def select_contact_components(raw,baseline,candidate,caps,limits,edges,rest,fps):
    from .surface_directed import connected_surfaces,track_metrics
    from .surface_clearance import deficits
    from .surface_topology import edge_changes
    labels=connected_surfaces(rest,edges);final=baseline.copy();records=[]
    for k in np.unique(labels):
        ids=np.flatnonzero(labels==k);mapping=np.full(len(rest),-1);mapping[ids]=np.arange(len(ids))
        local=mapping[edges[labels[edges[:,0]]==k]]
        if not len(local):continue
        r=raw[:,ids];old=baseline[:,ids];delta=candidate[:,ids]-old;attempts=[]
        if np.linalg.norm(delta,axis=-1).max()<1e-6:
            records.append({'component':int(k),'strength':0.,'attempts':[],
                            'gate':{'accepted':True,'rejection_reasons':[],'no_material_change':True}})
            print('MESH_COMPONENT_GATE',k,'unchanged',flush=True)
            continue
        cache={'metrics':track_metrics(r,old,caps,limits[:,ids],local,fps),
               'depth':np.array([deficits(old[f],caps[f],limits[:,ids]).max(0) for f in range(len(raw))]),
               'edges':np.array([edge_changes(old[f],r[f],local) for f in range(len(raw))])}
        for strength in [1.,.5,.25,.125,.0625,.03125,.015625]:
            trial=old+strength*delta
            gate=contact_track_gate(r,old,trial,caps,limits[:,ids],local,fps,cache=cache,quick=True)
            attempts.append({'strength':strength,'rejection_reasons':gate['rejection_reasons']})
            if gate['accepted']:
                final[:,ids]=trial;break
        else:strength=0.;gate['accepted']=False
        records.append({'component':int(k),'strength':strength,'attempts':attempts,'gate':gate})
        print('MESH_COMPONENT_GATE',k,strength,gate['rejection_reasons'],flush=True)
    combined=contact_track_gate(raw,baseline,final,caps,limits,edges,fps)
    if not combined['accepted']:final=baseline.copy()
    return final,{'components':records,'combined':combined,
                  'note':'Whole-component strength backoff. Each edge allows <=0.5mm extra error plus 5um numeric tolerance. Old capsule/deep-contact, >5mm count and actual-motion guards retained. Aggregate energy may increase only within this explicit per-edge budget.'}
