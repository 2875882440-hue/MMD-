import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_world_contacts import fixture_motion
from motionbridge.exporters import solve_rotations
from motionbridge.stability import fk_positions
from motionbridge.world_reprojection import reconcile_world_arms


def observations(confidence=.98):
    m,_=fixture_motion(travel=True);_,w,p=solve_rotations(m);n=len(p)
    desired=w.copy();rot=Rotation.from_euler('z',12,degrees=True).as_matrix()
    desired[:,16]=rot@w[:,16];desired[:,18]=rot@w[:,18]
    goal=fk_positions(desired,m['parents'],np.array(m['rest_positions']),p[:,0])
    wc=np.tile(np.eye(3),(n,1,1));cr=np.tile([0.,0.,3.],(n,1));K=np.tile([[700.,0,320],[0,700,240],[0,0,1]],(n,1,1))
    cam=goal-goal[:,:1]+cr[:,None];h=np.einsum('fij,fkj->fki',K,cam);px=h[:,:,:2]/h[:,:,2:]
    kp=np.zeros((n,17,3))
    for source,j in [(16,5),(17,6),(18,7),(19,8),(20,9),(21,10)]:kp[:,j,:2]=px[:,source];kp[:,j,2]=confidence
    return m,wc,cr,K,kp


def test_observed_arm_fits_better_with_fixed_bones_and_preserves_other_branches():
    args=observations();original=copy.deepcopy(args[0]);out=reconcile_world_arms(*args)
    r=out['image_arm_refinement']['sides']['left']
    assert r['mean_shoulder_relative_pixel_error_after']<r['mean_shoulder_relative_pixel_error_before']*.9
    assert r['maximum_rotation_adjustment_deg']<=12+1e-8
    _,old,p=solve_rotations(original);_,new,q=solve_rotations(out)
    assert np.array_equal(old[:,:16],new[:,:16]) and np.array_equal(p[:,0],q[:,0])
    assert np.max(np.linalg.norm(q[:,20]-p[:,20],axis=1))<=.06+1e-8
    assert args[0]==original
    assert reconcile_world_arms(out,*args[1:])==out


def test_unreliable_observations_do_not_invent_arm_motion():
    args=observations(.2);out=reconcile_world_arms(*args)
    assert out['world_rotations']==args[0]['world_rotations']
    assert out['image_arm_refinement']['sides']['left']['trusted_frames']==0


def test_invalid_camera_rejected():
    args=list(observations());args[3][:]=0
    with pytest.raises(ValueError):reconcile_world_arms(*args)


def test_confidence_dropout_fades_correction_without_moving_untrusted_frame():
    args=list(observations());args[-1][75,5,2]=.1
    out=reconcile_world_arms(*args)
    old=np.asarray(args[0]['world_rotations']);new=np.asarray(out['world_rotations'])
    delta=Rotation.from_matrix(new[:,16]@old[:,16].swapaxes(1,2)).magnitude()
    assert delta[75]==0
    assert delta[60]>delta[70]>delta[74]>=0
    assert np.rad2deg(abs(np.diff(delta))).max()<1.5
