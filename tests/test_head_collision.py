import copy,json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from motionbridge.head_collision import avoid_arm_head,constrain_head_collision
from motionbridge.body_collision import arm_clearance
from motionbridge.blender_head import animated_head_proxy
from motionbridge.exporters import solve_rotations
from test_world_contacts import fixture_motion


def sample():
    n=60
    p=np.tile([[.25,-.1,0],[.12,.05,.02],[-.02,.03,.03]],(n,1,1))
    center=np.zeros((n,3));basis=np.tile(np.eye(3),(n,1,1));radii=np.array([.10,.13,.10])
    torso=(center+np.array([0,-.5,0]),basis,np.array([.14,.22,.08]),.025)
    return p,center,basis,radii,.025,30,torso


def test_head_constraint_reduces_overlap_without_changing_bone_lengths():
    args=sample();out,r=avoid_arm_head(*args)
    assert r['after']['total_proxy_deficit_m']<r['before']['total_proxy_deficit_m']*.5
    assert r['max_segment_adjustment_deg']<=20+1e-6 and r['torso_safeguard_passed']
    assert np.array_equal(out[:,0],args[0][:,0])
    assert np.allclose(np.linalg.norm(np.diff(out,axis=1),axis=-1),np.linalg.norm(np.diff(args[0],axis=1),axis=-1))
    assert np.max(abs(np.diff(out,axis=0)))<1e-10
    json.dumps(r)


def test_torso_guard_backs_off_if_head_escape_would_hit_another_obstacle():
    args=list(sample());args[-1]=(args[1]+[0,0,.15],args[2],np.array([.15,.15,.1]),.025)
    out,r=avoid_arm_head(*args)
    assert r['torso_safeguard_passed'] and r['backoff_frames']>0
    before=arm_clearance(args[0],*args[-1]);after=arm_clearance(out,*args[-1])
    assert np.all(after.min(1)>=np.minimum(before.min(1),0)-1.1e-6)


def test_external_accessory_candidate_cannot_reintroduce_head_overlap():
    args=sample();clear,_=avoid_arm_head(*args)
    out,report=avoid_arm_head(clear,*args[1:],candidate=args[0])
    assert report['backoff_frames']>0 and report['torso_safeguard_passed']
    before=arm_clearance(clear,*args[1:5]);after=arm_clearance(out,*args[1:5])
    assert np.all(np.maximum(-after,0).sum(1)<=np.maximum(-before,0).sum(1)+1.1e-6)
    assert np.allclose(np.linalg.norm(np.diff(out,axis=1),axis=-1),np.linalg.norm(np.diff(clear,axis=1),axis=-1))


def test_clear_animation_is_exact_and_rotated_motion_has_equivalent_solution():
    args=list(sample());clear=args[0]+[0,0,1]
    out,r=avoid_arm_head(clear,*args[1:]);assert np.array_equal(out,clear) and r['corrected_frames']==0
    expected,_=avoid_arm_head(*args);rot=Rotation.from_euler('xyz',[50,-30,17],degrees=True).as_matrix();shift=np.array([3.,1.,-5.])
    tc,tb,tr,ta=args[-1]
    out,_=avoid_arm_head(args[0]@rot.T+shift,args[1]@rot.T+shift,rot@args[2],*args[3:6],(tc@rot.T+shift,rot@tb,tr,ta))
    assert np.allclose(out,expected@rot.T+shift,atol=1e-9)


def test_source_preserves_head_root_floor_body_metadata_and_is_idempotent():
    m,_=fixture_motion(travel=True);m['floor_collision']={'enabled':True};m['body_collision']={'enabled':True}
    old=copy.deepcopy(m);out=constrain_head_collision(m);_,w,p=solve_rotations(m);_,nw,np_=solve_rotations(out)
    assert np.array_equal(w[:,:16],nw[:,:16]) and np.array_equal(p[:,:18],np_[:,:18])
    assert m==old and out['body_collision']==m['body_collision'] and out['floor_collision']==m['floor_collision']
    assert constrain_head_collision(out)==out
    m['root_mode']='in_place'
    with pytest.raises(ValueError):constrain_head_collision(m)


def test_calibrated_proxy_follows_head_rest_transform_and_keeps_dimensions():
    rest=np.eye(4);rest[:3,3]=[0,0,1]
    transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('y',85,degrees=True).as_matrix();transform[:3,3]=[2,0,1]
    profile={'center_rest':[0,.1,1.1],'radii_m':[.08,.06,.10]}
    c,b,r=animated_head_proxy([rest,transform@rest],rest,profile)
    assert np.allclose(c[0],profile['center_rest'])
    assert np.allclose(c[1],transform[:3,:3]@profile['center_rest']+transform[:3,3])
    assert np.allclose(b[1],transform[:3,:3]) and np.array_equal(r,profile['radii_m'])
