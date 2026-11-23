import copy
import numpy as np
import pytest
from motionbridge.contacts import contact_tracks, lock_support_feet
from motionbridge.hands import associate_wrists
from motionbridge.exporters import create_demo_motion, solve_rotations
from motionbridge.stability import stabilize_motion
from scipy.spatial.transform import Rotation


def foot_observations(count=60):
    image=np.zeros((count,33,3))
    image[:,:,0]=.5
    image[:,:,1]=.5
    image[:,[27,28,31,32],1]=.9
    return image,np.ones((count,33))


def test_contacts_release_raised_and_uncertain_feet_and_airborne_interval():
    image,c=foot_observations()
    image[20:40,[27,31],1]=.65
    c[40:50,[28,32]]=.1
    image[50:55,[27,28,31,32],1]=.6
    w,segments=contact_tracks(image,c,30,720,1280)
    assert w[5:15].min()>.9
    assert w[22:38,0].max()==0
    assert w[41:49,1].max()==0
    assert w[51:54].max()==0
    assert len(segments)>=2


def test_contact_lock_reduces_support_slip_and_preserves_lengths_and_input():
    motion=create_demo_motion(fps=30,seconds=2)
    rest=np.array(motion['rest_positions'])
    for f,frame in enumerate(motion['frames']):
        frame['positions']=(rest+[np.sin(f*.1)*.025,0,0]).tolist()
    original=copy.deepcopy(motion)
    image,c=foot_observations()
    result=lock_support_feet(motion,image,c,720,1280)
    _,world,fk=solve_rotations(result)
    assert motion==original
    report=result['foot_contacts']
    assert report['mean_support_horizontal_step_after_m'] < report['mean_support_horizontal_step_before_m']*.5
    for j,p in enumerate(motion['parents']):
        if p>=0:
            np.testing.assert_allclose(np.linalg.norm(fk[:,j]-fk[:,p],axis=-1),np.linalg.norm(rest[j]-rest[p]),atol=2e-6)


def test_no_supported_contacts_do_not_move_pose():
    motion=stabilize_motion(create_demo_motion(fps=30,seconds=2))
    image,c=foot_observations()
    result=lock_support_feet(motion,image,c*.1,720,1280)
    np.testing.assert_allclose([f['positions'] for f in result['frames']],[f['positions'] for f in motion['frames']],atol=2e-7)


def test_wrist_assignment_tracks_crossing_without_screen_side_or_confidence_promotion():
    n=30
    image=np.zeros((n,33,3))
    image[:,[11,12],:2]=[[.4,.3],[.6,.3]]
    image[:,[23,24],:2]=[[.45,.65],[.55,.65]]
    x=np.linspace(.25,.75,n)
    image[:,15,:2]=np.stack([x,np.full(n,.4)],axis=1)
    image[:,16,:2]=np.stack([1-x,np.full(n,.44)],axis=1)
    image[:,13,:2]=image[:,15,:2]+[0,.1]
    image[:,14,:2]=image[:,16,:2]+[0,.1]
    c=np.ones((n,33))
    orig=image.copy()
    candidates=[[{'wrist':[float(1-v),.43],'score':.95},{'wrist':[float(v),.39],'score':.95}] for v in x]
    result,support,report=associate_wrists(image,c,candidates,30,1280,1280)
    assert report['associated_wrist_samples']>35
    assert result[-1,15,0]>.7 and result[-1,16,0]<.3
    np.testing.assert_array_equal(image,orig)
    np.testing.assert_array_equal(c,1)


def test_ambiguous_or_far_hands_abstain():
    image=np.ones((2,33,3))*.5
    c=np.ones((2,33))
    candidates=[[{'wrist':[.5,.5],'score':1.}], [{'wrist':[.99,.01],'score':1.}]]
    result,support,report=associate_wrists(image,c,candidates,30,720,1280)
    assert support.max()==0
    np.testing.assert_array_equal(result,image)


def test_adaptive_stability_preserves_coherent_fast_arm_but_rejects_isolated_flip():
    m=create_demo_motion(fps=30,seconds=1)
    rest=np.array(m['rest_positions'])
    for f,frame in enumerate(m['frames']):
        p=rest.copy()
        rot=Rotation.from_euler('z',f*45,degrees=True).as_matrix()
        p[[7,8]]=rest[6]+(rest[[7,8]]-rest[6])@rot.T
        frame['positions']=p.tolist()
        frame['confidence']=[1.]*len(rest)
    legacy=stabilize_motion(m)
    m['settings']={'enhance_motion':True}
    improved=stabilize_motion(m)
    target=np.array([f['positions'][8] for f in m['frames']])
    old=np.array([f['positions'][8] for f in legacy['frames']])
    new=np.array([f['positions'][8] for f in improved['frames']])
    assert np.linalg.norm(new-target,axis=-1).mean()<np.linalg.norm(old-target,axis=-1).mean()*.65
    assert improved['rotation_stabilization']['trusted_coherent_samples']>0


def test_head_relative_target_goal_adapts_to_target_proportions():
    from motionbridge.target_ik import hand_goal, chain_ik
    from motionbridge.exporters import JOINT_NAMES, default_rest_positions
    p=default_rest_positions()
    p[8]=p[4]+[.02,.05,0]
    target={'shoulder':np.array([.15,.5,0]),'head':np.array([0,.9,0]),'chest':np.array([0,.4,0])}
    goal=hand_goal(p,JOINT_NAMES,target,'Left',[.2,.18])
    assert goal[1]>.9  # follows target head, not the adult source shoulder ratio
    middle,end=chain_ik(target['shoulder'],[.25,.6,0],[.3,.7,0],goal,[.2,.18])
    assert np.linalg.norm(middle-target['shoulder'])==pytest.approx(.2)
    assert np.linalg.norm(end-middle)==pytest.approx(.18)


def test_adaptive_does_not_treat_depth_only_rotation_as_evidence_for_fast_motion():
    m=create_demo_motion(fps=30,seconds=1)
    rest=np.array(m['rest_positions'])
    for f,frame in enumerate(m['frames']):
        p=rest.copy()
        rot=Rotation.from_euler('y',f*40,degrees=True).as_matrix()
        p[[7,8]]=rest[6]+(rest[[7,8]]-rest[6])@rot.T
        frame['positions']=p.tolist()
        frame['confidence']=[1.]*len(rest)
    m['settings']={'enhance_motion':True}
    result=stabilize_motion(m)
    assert result['rotation_stabilization']['trusted_coherent_samples']==0
