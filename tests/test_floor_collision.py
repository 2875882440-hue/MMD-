import copy
import numpy as np
import pytest
from test_world_contacts import fixture_motion
from motionbridge.exporters import solve_rotations
from motionbridge.floor_collision import constrain_world_floor,foot_proxy_heights,lift_envelope


def motion_with_clearance(track):
    m,_=fixture_motion(travel=True);_,w,p=solve_rotations(m)
    low=foot_proxy_heights(p,w,m['rest_positions']).min(axis=1)
    p[:,:,1]+=(np.asarray(track)-low)[:,None]
    for f,points in zip(m['frames'],p):f['positions']=points.tolist()
    return m


def test_unilateral_contact_fixes_penetration_without_airborne_clamp_or_arm_changes():
    height=np.full(150,.02);height[20:45]=-.04;height[80:100]=.3
    m=motion_with_clearance(height);original=copy.deepcopy(m)
    result=constrain_world_floor(m);_,w,p=solve_rotations(result);_,old,raw=solve_rotations(m)
    assert foot_proxy_heights(p,w,m['rest_positions']).min()>=.001-1e-6
    assert np.array_equal(w[:,[0,3,6,9,12,13,14,15,16,17,18,19,20,21]],old[:,[0,3,6,9,12,13,14,15,16,17,18,19,20,21]])
    assert np.array_equal(p[:,0][:,[0,2]],raw[:,0][:,[0,2]])
    assert np.allclose(p[85:95],raw[85:95])
    assert m==original and constrain_world_floor(result)==result


def test_clear_airborne_motion_is_unchanged():
    m=motion_with_clearance(np.full(150,.4));out=constrain_world_floor(m)
    assert np.allclose([f['positions'] for f in m['frames']],[f['positions'] for f in out['frames']])
    assert out['floor_collision']['corrected_foot_samples']==0


def test_lift_envelope_enforces_obstacle_and_velocity_without_downward_motion():
    req=np.zeros(90);req[45]=.06;out=lift_envelope(req,30,.3)
    assert np.all(out>=req) and np.all(out>=0)
    assert np.max(abs(np.diff(out)))<=.01+1e-12
    assert np.count_nonzero(out[:30])==0 and out[45]==.06
    assert np.array_equal(lift_envelope(out,30,.3),out)


def test_invalid_collision_input_fails_explicitly():
    with pytest.raises(ValueError):lift_envelope([0,float('nan')],30)
    m=motion_with_clearance(np.full(150,.1));m['root_mode']='in_place'
    with pytest.raises(ValueError):constrain_world_floor(m)
