import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from motionbridge.surface_clearance import capsule_distance,project_surface,deficits


def test_clearance_reduces_penetration_and_enforces_shift_cap():
    points=np.array([[.02,.02,.5],[.03,.01,.6],[.2,0,.5]])
    caps=np.array([[[0,0,0],[0,0,1]]]);limits=np.full((1,3),.07)
    out=project_surface(points,caps,limits)
    assert deficits(out,caps,limits).sum()<deficits(points,caps,limits).sum()
    assert np.linalg.norm(out-points,axis=1).max()<=.03+1e-10
    assert np.array_equal(out[2],points[2])


def test_original_attachment_overlap_can_be_preserved_and_transform_is_equivariant():
    points=np.array([[.02,.01,.5],[.03,.02,.6]]);caps=np.array([[[0,0,0],[0,0,1]]])
    baseline=np.array([capsule_distance(points,*caps[0])[0]])
    assert np.allclose(project_surface(points,caps,baseline),points)
    limits=np.full((1,2),.07);out=project_surface(points,caps,limits)
    r=Rotation.from_euler('xyz',[20,30,40],degrees=True).as_matrix();t=np.array([3,4,5])
    assert np.allclose(project_surface(points@r.T+t,caps@r.T+t,limits),out@r.T+t)


def test_surface_invalid_data_fails():
    with pytest.raises(ValueError):project_surface(np.array([[np.nan,0,0]]),np.zeros((1,2,3)),np.ones((1,1)))
