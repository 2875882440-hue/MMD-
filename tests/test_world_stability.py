import copy
import numpy as np
from scipy.spatial.transform import Rotation
from test_world_contacts import fixture_motion
from motionbridge.stability import fk_positions
from motionbridge.exporters import solve_rotations
from motionbridge.world_stability import stabilize_world_upper_body


def animated_motion(angles):
    m,_=fixture_motion(jump=True,travel=True)
    w=np.array(m['world_rotations']);n=len(w)
    yaw=Rotation.from_euler('y',np.linspace(0,360,n)[:,None],degrees=True).as_matrix()
    for j in range(22):w[:,j]=yaw
    elbow=Rotation.from_euler('x',angles[:,None],degrees=True).as_matrix()
    w[:,18]=yaw@elbow;w[:,20]=w[:,18]
    p=fk_positions(w,m['parents'],np.array(m['rest_positions']),np.array([f['positions'][0] for f in m['frames']]))
    m['world_rotations']=w.tolist()
    for f,pt in zip(m['frames'],p):f['positions']=pt.tolist()
    return m


def test_local_jitter_reduced_without_changing_turn_travel_jump_or_legs():
    angles=np.tile([-25.,25.],75);m=animated_motion(angles);before=copy.deepcopy(m)
    result=stabilize_world_upper_body(m)
    _,w,p=solve_rotations(result);_,raw,old=solve_rotations(m)
    assert np.array_equal(w[:,0],raw[:,0])
    assert np.array_equal(w[:,[1,2,4,5,7,8,10,11]],raw[:,[1,2,4,5,7,8,10,11]])
    assert np.array_equal(p[:,0],old[:,0])
    assert result['world_stability']['max_local_speed_after_deg_s']<result['world_stability']['max_local_speed_before_deg_s']*.3
    assert m==before
    assert stabilize_world_upper_body(result)==result


def test_fast_coherent_rotation_keeps_rate_and_no_phase_lag_interior():
    m=animated_motion(np.arange(150)*18.)  # 540 degrees/sec
    result=stabilize_world_upper_body(m)
    _,w,_=solve_rotations(result);_,raw,_=solve_rotations(m)
    delta=w[8:-8,18].swapaxes(-1,-2)@raw[8:-8,18]
    assert np.max(Rotation.from_matrix(delta).magnitude())<1e-6
