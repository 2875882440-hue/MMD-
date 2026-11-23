import numpy as np
from motionbridge.body_identity import stabilize_arm_observations


def observations(n=40):
    kp=np.zeros((n,17,3));kp[:,:,2]=.9
    for j in range(17):kp[:,j,:2]=[100+j*30,100+j*20]
    kp[:,9,:2]=[150,400];kp[:,10,:2]=[650,400]
    return kp,np.tile([400,400,1000.],(n,1))


def test_only_swapped_wrists_are_reassigned_not_torso_or_legs():
    kp,boxes=observations();truth=kp.copy()
    kp[15:18,[9,10]]=kp[15:18,[10,9]]
    corrected,report=stabilize_arm_observations(kp,boxes)
    np.testing.assert_allclose(corrected,truth)
    assert report['wrist_swaps']==[15,16,17]
    assert not report['elbow_swaps']


def test_one_hand_duplicate_is_interpolated_with_reduced_confidence():
    kp,boxes=observations();kp[15:17,9,:2]=kp[15:17,10,:2]
    corrected,report=stabilize_arm_observations(kp,boxes)
    assert np.allclose(corrected[15:17,9,:2],[150,400])
    assert np.all(corrected[15:17,9,2]<.3)
    assert report['interpolated']==[[15,9],[16,9]]


def test_fast_continuous_motion_and_genuine_crossing_are_preserved():
    kp,boxes=observations()
    kp[:,9,0]=np.linspace(0,1000,len(kp));kp[:,10,0]=np.linspace(1000,0,len(kp))
    kp[:,9,1]=400;kp[:,10,1]=420
    corrected,report=stabilize_arm_observations(kp,boxes)
    np.testing.assert_allclose(corrected,kp)
    assert not report['interpolated']
