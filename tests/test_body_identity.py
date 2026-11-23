import numpy as np
from motionbridge.body_identity import stabilize_coco_identity, COCO_FLIP

def test_recovers_label_glitches_through_a_complete_turn():
    n=91
    theta=np.linspace(0,2*np.pi,n)
    correct=np.zeros((n,17,3))
    correct[...,2]=.95
    for left,right in [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]:
        width=.15 if left in (5,11) else .12
        correct[:,left,0]=width*np.cos(theta)
        correct[:,right,0]=-width*np.cos(theta)
        correct[:,left,1]=left*.03+.025*np.sin(theta)
        correct[:,right,1]=left*.03-.025*np.sin(theta)
    raw=correct.copy()
    raw[28:39]=raw[28:39][:,COCO_FLIP]
    raw[44]=raw[44,COCO_FLIP]
    boxes=np.tile([0,0,1.],(n,1))
    fixed,report=stabilize_coco_identity(raw,boxes)
    np.testing.assert_allclose(fixed,correct)
    assert report['swapped_frames']==list(range(28,39))+[44]

def test_stationary_identity_is_unchanged():
    points=np.zeros((10,17,3))
    points[:,:,0]=np.arange(17)*2
    points[:,:,2]=.8
    fixed,report=stabilize_coco_identity(points,np.tile([10,10,50.],(10,1)))
    np.testing.assert_array_equal(fixed,points)
    assert report['swapped_frames']==[]
