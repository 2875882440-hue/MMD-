"""Conservative topology regularization of already collision-corrected surfaces.

Only the extra correction is smoothed. Previous per-vertex capsule clearance,
the 3cm movement cap, and mesh strain diagnostics take priority over smoothing.
"""
import numpy as np
from .surface_clearance import project_surface,deficits


def edge_changes(points,reference,edges):
    i,j=edges.T
    return abs(np.linalg.norm(points[i]-points[j],axis=1)-np.linalg.norm(reference[i]-reference[j],axis=1))


def _proposal(raw,baseline,caps,limits,edges,iterations=12):
    i,j=edges.T;n=len(raw);degree=np.maximum(np.bincount(edges.ravel(),minlength=n),1)
    length=np.linalg.norm(raw[i]-raw[j],axis=1);p=baseline.copy()
    for _ in range(iterations):
        vector=p[j]-p[i];current=np.linalg.norm(vector,axis=1)
        # Preserve the current skinned pose, not an unrelated rest-pose shape.
        excess=current-np.clip(current,np.maximum(0,length*.90-.0005),length*1.10+.0005)
        correction=vector*(excess/np.maximum(current,1e-10))[:,None]
        move=np.stack([np.bincount(i,weights=correction[:,k],minlength=n)-np.bincount(j,weights=correction[:,k],minlength=n) for k in range(3)],axis=1)/degree[:,None]
        p=project_surface(raw,caps,limits,initial=p+.8*move,iterations=2)
    return p


def guard_surface(raw,baseline,candidate,caps,limits,edges):
    """Reject new vertex penetration, excess displacement, or worse edge strain."""
    baseline_depth=deficits(baseline,caps,limits).max(0)
    allowed=baseline_depth+1e-7
    allowed[baseline_depth<=.001]=np.minimum(allowed[baseline_depth<=.001],.001)
    old_edges=edge_changes(baseline,raw,edges)
    delta=candidate-baseline
    delta*=np.minimum(1,.01/np.maximum(np.linalg.norm(delta,axis=1),1e-12))[:,None]
    active=np.flatnonzero(np.linalg.norm(delta,axis=1)>1e-10)
    def collision_guard(scale):
        p=baseline.copy()
        if not len(active):return p
        weight=np.full(len(active),scale)
        for _ in range(20):
            q=baseline[active]+delta[active]*weight[:,None]
            bad=(deficits(q,caps,limits[:,active]).max(0)>allowed[active])|(np.linalg.norm(q-raw[active],axis=1)>.0300005)
            if not bad.any():break
            weight[bad]*=.5
        q=baseline[active]+delta[active]*weight[:,None]
        bad=(deficits(q,caps,limits[:,active]).max(0)>allowed[active])|(np.linalg.norm(q-raw[active],axis=1)>.0300005)
        q[bad]=baseline[active][bad];p[active]=q
        return p
    scale=1.
    for _ in range(12):
        p=collision_guard(scale);new_edges=edge_changes(p,raw,edges)
        if (new_edges.max(initial=0)<=old_edges.max(initial=0)+1e-7
            and np.sum(new_edges**2)<=np.sum(old_edges**2)+1e-10
            and (new_edges>.005).sum()<=(old_edges>.005).sum()):return p,scale
        scale*=.5
    return baseline.copy(),0.


def regularize_surface_tracks(raw,baseline,capsules,limits,edges,fps):
    raw=np.asarray(raw,float);baseline=np.asarray(baseline,float);caps=np.asarray(capsules,float)
    limits=np.asarray(limits,float);edges=np.asarray(edges)
    if (raw.ndim!=3 or raw.shape[-1]!=3 or baseline.shape!=raw.shape or len(raw)<1
        or caps.shape!=(len(raw),len(limits),2,3) or limits.shape[1:]!=(raw.shape[1],)
        or edges.ndim!=2 or edges.shape[1]!=2 or edges.dtype.kind not in 'iu'
        or not len(edges) or edges.min()<0 or edges.max()>=raw.shape[1]
        or not all(np.isfinite(x).all() for x in [raw,baseline,caps,limits,fps]) or fps<=0 or (limits<0).any()):
        raise ValueError('Invalid surface topology tracks')
    proposals=baseline.copy();active=0
    for f in range(len(raw)):
        if edge_changes(baseline[f],raw[f],edges).max()>.005:
            proposals[f]=_proposal(raw[f],baseline[f],caps[f],limits,edges);active+=1
    delta=proposals-baseline
    delta*=np.minimum(1,.01/np.maximum(np.linalg.norm(delta,axis=-1,keepdims=True),1e-12))
    radius=max(1,round(fps/15));weights=np.r_[np.arange(1,radius+2),np.arange(radius,0,-1)].astype(float);weights/=weights.sum()
    padded=np.pad(delta,[(radius,radius),(0,0),(0,0)],mode='edge')
    delta=sum(w*padded[k:k+len(raw)] for k,w in enumerate(weights))
    final=baseline.copy();scales=[];before_hits=after_hits=0;before_max=after_max=0.;before_energy=after_energy=0.
    for f in range(len(raw)):
        final[f],scale=guard_surface(raw[f],baseline[f],baseline[f]+delta[f],caps[f],limits,edges);scales.append(scale)
        for label,p in [('before',baseline[f]),('after',final[f])]:
            change=edge_changes(p,raw[f],edges);hits=int((change>.005).sum());peak=float(change.max());energy=float(np.sum(change**2))
            if label=='before':before_hits+=hits;before_max=max(before_max,peak);before_energy+=energy
            else:after_hits+=hits;after_max=max(after_max,peak);after_energy+=energy
    return final,{'enabled':True,'method':'topology_relaxation_collision_guard_v1','proposal_frames':active,
        'edge_frames_changed_over_5mm_before':before_hits,'edge_frames_changed_over_5mm_after':after_hits,
        'maximum_edge_change_before_m':before_max,'maximum_edge_change_after_m':after_max,
        'edge_change_squared_sum_before':before_energy,'edge_change_squared_sum_after':after_energy,
        'maximum_extra_vertex_shift_m':float(np.linalg.norm(final-baseline,axis=-1).max()),
        'backoff_frames':int((np.array(scales)<1).sum()),'rejected_frames':int((np.array(scales)==0).sum()),
        'note':'Preserves previous per-vertex capsule penetration within 0.1 micrometer; no new >1mm samples. Keeps total 3cm and extra 1cm caps. Reduces edge distortion, not cloth physics or self-collision.'}
