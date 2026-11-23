import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from motionbridge.surface_topology import regularize_surface_tracks,guard_surface,edge_changes
from motionbridge.surface_clearance import deficits


def strip():
    raw=np.stack([np.full(12,.15),np.zeros(12),np.linspace(.1,.5,12)],axis=1)
    baseline=raw.copy();baseline[:,1]=np.tile([-.018,.018],6)
    caps=np.array([[[0,0,0],[0,0,1]]]);limits=np.full((1,12),.05)
    edges=np.stack([np.arange(11),np.arange(1,12)],axis=1)
    return raw,baseline,caps,limits,edges


def test_regularization_reduces_zigzag_without_new_overlap_or_excess_displacement():
    raw,old,caps,limits,edges=strip()
    raw=np.tile(raw,(9,1,1));old=np.tile(old,(9,1,1));caps=np.tile(caps,(9,1,1,1))
    out,report=regularize_surface_tracks(raw,old,caps,limits,edges,30)
    assert report['edge_change_squared_sum_after']<report['edge_change_squared_sum_before']*.5
    assert report['maximum_extra_vertex_shift_m']<=.01+1e-8
    assert np.linalg.norm(out-raw,axis=-1).max()<=.0300005
    assert np.allclose(out,out[:1])
    assert not deficits(out[0],caps[0],limits).any()


def test_guard_keeps_original_capsule_clearance_when_candidate_moves_into_body():
    raw,old,caps,limits,edges=strip();raw[:,0]=.055;old[:,0]=.055
    candidate=old.copy();candidate[:,0]=.025
    out,_=guard_surface(raw,old,candidate,caps,limits,edges)
    assert np.all(deficits(out,caps,limits).max(0)<=deficits(old,caps,limits).max(0)+1.1e-7)
    assert np.linalg.norm(out-old,axis=-1).max()<=.01+1e-8
    assert edge_changes(out,raw,edges).max()<=edge_changes(old,raw,edges).max()+1.1e-7


def test_unmodified_surface_is_exact_and_rotated_translated_solution_is_equivalent():
    raw,old,caps,limits,edges=strip()
    out,report=regularize_surface_tracks(raw[None],raw[None],caps[None],limits,edges,30)
    assert np.array_equal(out[0],raw) and report['maximum_extra_vertex_shift_m']==0
    expected,_=regularize_surface_tracks(raw[None],old[None],caps[None],limits,edges,30)
    r=Rotation.from_euler('xyz',[20,30,40],degrees=True).as_matrix();t=np.array([2.,-3.,4.])
    result,_=regularize_surface_tracks(raw[None]@r.T+t,old[None]@r.T+t,caps[None]@r.T+t,limits,edges,30)
    assert np.allclose(result,expected@r.T+t,atol=1e-8)


def test_invalid_topology_rejected():
    raw,old,caps,limits,edges=strip();edges[0,0]=100
    with pytest.raises(ValueError):regularize_surface_tracks(raw[None],old[None],caps[None],limits,edges,30)
