import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from motionbridge.body_collision import avoid_arm_torso,constrain_body_collision,torso_space
from motionbridge.exporters import solve_rotations
from test_world_contacts import fixture_motion


def crossing(n=90):
    p=np.tile([[.22,.18,0],[.10,-.03,.025],[-.12,.03,.04]],(n,1,1)).astype(float)
    c=np.zeros((n,3));b=np.tile(np.eye(3),(n,1,1));r=np.array([.15,.28,.10])
    return p,c,b,r,.025


def test_avoidance_reduces_collision_preserves_lengths_and_bounds_edits():
    args=crossing();p=args[0];out,report=avoid_arm_torso(*args,30)
    assert report['after_total_proxy_deficit_m']<report['before_total_proxy_deficit_m']*.5
    assert np.allclose(out[:,0],p[:,0])
    assert np.allclose(np.linalg.norm(np.diff(out,axis=1),axis=-1),np.linalg.norm(np.diff(p,axis=1),axis=-1),atol=1e-10)
    a=np.diff(out,axis=1);b=np.diff(p,axis=1)
    angle=np.arccos(np.clip(np.sum(a*b,axis=-1)/np.linalg.norm(a,axis=-1)/np.linalg.norm(b,axis=-1),-1,1))
    assert np.rad2deg(angle).max()<=25+1e-7
    assert np.max(abs(np.diff(out,axis=0)))<1e-10


def test_clear_motion_unchanged_and_solver_equivariant_under_world_turn_and_translation():
    args=list(crossing());p=args[0].copy();p[:,:,2]+=.5
    out,report=avoid_arm_torso(p,*args[1:],30)
    assert np.array_equal(p,out) and report['corrected_frames']==0
    expected,_=avoid_arm_torso(*args,30)
    rot=Rotation.from_euler('xyz',[35,73,18],degrees=True).as_matrix();shift=np.array([2.,3.,-4.])
    moved=args[0]@rot.T+shift;center=args[1]@rot.T+shift;basis=rot@args[2]
    actual,_=avoid_arm_torso(moved,center,basis,*args[3:],30)
    assert np.allclose(actual,expected@rot.T+shift,atol=1e-9)


def test_source_preserves_root_legs_floor_metadata_and_input():
    m,_=fixture_motion(travel=True);m['floor_collision']={'enabled':True,'example':'preserve'}
    old=copy.deepcopy(m);out=constrain_body_collision(m)
    _,w,p=solve_rotations(m);_,nw,np_=solve_rotations(out)
    assert np.array_equal(w[:,:16],nw[:,:16]) and np.array_equal(p[:,:18],np_[:,:18])
    assert out['floor_collision']==m['floor_collision'] and m==old
    assert constrain_body_collision(out)==out


def test_invalid_inputs_rejected():
    args=list(crossing());args[0][0,0,0]=np.nan
    with pytest.raises(ValueError):avoid_arm_torso(*args,30)
    m,_=fixture_motion();m['root_mode']='in_place'
    with pytest.raises(ValueError):constrain_body_collision(m)
    with pytest.raises(ValueError):torso_space(np.zeros((2,3)),np.zeros((2,3)),np.zeros((2,3)),np.zeros(3),np.zeros(3),np.zeros(3))


def test_blender_float32_geometry_produces_serializable_report():
    import json
    hips=np.array([0,0,0],dtype=np.float32);left=np.array([.2,.5,0],dtype=np.float32);right=left.copy();right[0]*=-1
    center,basis,radii,radius=torso_space(hips[None],left[None],right[None],hips,left,right)
    json.dumps({'radii':radii.tolist(),'arm_radius':radius})
    p=crossing(1)[0].astype(np.float32)
    _,report=avoid_arm_torso(p,center,basis,radii,radius,30)
    json.dumps(report)
