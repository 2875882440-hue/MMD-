"""Bounded mesh-vertex clearance against animated body capsules (NumPy only)."""
import numpy as np


def capsule_distance(points,start,end):
    points=np.asarray(points,float);start=np.asarray(start,float);end=np.asarray(end,float)
    axis=end-start;length=np.sum(axis*axis,axis=-1,keepdims=True)
    amount=np.clip(np.sum((points-start)*axis,axis=-1,keepdims=True)/np.maximum(length,1e-12),0,1)
    vector=points-(start+amount*axis)
    return np.linalg.norm(vector,axis=-1),vector


def project_surface(points,capsules,limits,max_shift=.03,initial=None,iterations=5):
    """Retain rest-pose overlaps via per-vertex limits; resolve new penetration.

    Does not simulate cloth or certify triangle intersections. The movement cap
    takes precedence over fully eliminating overlaps; unresolved values report.
    """
    original=np.asarray(points,float)
    caps=np.asarray(capsules,float);limits=np.asarray(limits,float)
    if original.ndim!=2 or original.shape[1]!=3 or caps.ndim!=3 or caps.shape[1:]!=(2,3) or limits.shape!=(len(caps),len(original)):
        raise ValueError('Invalid surface/capsule shapes')
    if not all(np.isfinite(x).all() for x in [original,caps,limits,max_shift]) or max_shift<0 or (limits<0).any():raise ValueError('Invalid surface constraints')
    result=original.copy() if initial is None else np.asarray(initial,float).copy()
    if result.shape!=original.shape or not np.isfinite(result).all():raise ValueError('Invalid surface initial state')
    for _ in range(iterations):
        for (start,end),minimum in zip(caps,limits):
            distance,vector=capsule_distance(result,start,end)
            # Deterministic perpendicular fallback for an exactly axial point.
            axis=end-start;reference=np.eye(3)[np.argmin(abs(axis))]
            fallback=np.cross(axis,reference)
            if np.linalg.norm(fallback)<1e-9:fallback=np.array([1.,0.,0.])
            fallback/=np.linalg.norm(fallback)
            direction=np.where((distance>1e-9)[:,None],vector/np.maximum(distance[:,None],1e-9),fallback)
            result+=direction*np.maximum(minimum-distance,0)[:,None]
            offset=result-original
            result=original+offset*np.minimum(1,max_shift/np.maximum(np.linalg.norm(offset,axis=-1),1e-12))[:,None]
    return result


def deficits(points,capsules,limits):
    return np.maximum(np.stack([minimum-capsule_distance(points,*cap)[0] for cap,minimum in zip(capsules,limits)]),0)
