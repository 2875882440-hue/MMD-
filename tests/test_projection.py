import copy
import numpy as np
import pytest
from motionbridge.hands import image_observation_weights
from motionbridge.pipeline import _fill_and_smooth
from motionbridge.projection import fit_projected_chain, refine_projected_arms
from motionbridge.exporters import create_demo_motion, solve_rotations
from motionbridge.stability import stabilize_motion


def test_independent_hand_observation_survives_body_confidence_gap_without_promoting_body():
    c = np.ones((12,33))
    c[4:8,15] = .1
    p = np.zeros((12,33,3))
    p[4:8,15,1] = -.2
    original = c.copy()
    support = np.zeros((12,2)); support[4:8,0] = .85
    weights = image_observation_weights(c, support)
    old, _ = _fill_and_smooth(p, c, 0, 30)
    new, _ = _fill_and_smooth(p, weights, 0, 30)
    assert old[5,15,1] == 0 and new[5,15,1] == -.2
    np.testing.assert_array_equal(c, original)
    np.testing.assert_array_equal(weights[:,16], c[:,16])
    with pytest.raises(ValueError):
        image_observation_weights(c, np.full((12,2), np.nan))


def test_projection_recovers_visible_raise_despite_wrong_depth_and_keeps_lengths():
    start = np.zeros(3)
    prior = np.array([[.1,0,0], [.1,.1,.2828427], [.1,.2,.5656854]])
    lengths = [.1,.3,.3]
    targets = np.array([[.1,0],[.1,.28],[.1,.56]])
    new, _ = fit_projected_chain(start, prior, lengths, targets, np.ones(3))
    assert np.linalg.norm(new[-1,:2]-targets[-1]) < .01
    np.testing.assert_allclose(np.linalg.norm(np.diff(np.vstack((start,new)),axis=0),axis=1), lengths, atol=1e-8)


def test_untrusted_projection_cannot_drag_arm_to_image_outlier():
    start = np.zeros(3)
    prior = np.array([[.1,0,0],[.4,0,0],[.7,0,0]])
    new, _ = fit_projected_chain(start, prior, [.1,.3,.3], np.full((3,2), 10.), np.zeros(3))
    np.testing.assert_array_equal(new, prior)


def test_detail_smoothing_keeps_credible_fast_curve_but_not_isolated_outlier():
    p = np.zeros((30,33,3)); t=np.linspace(0,1,30)
    p[:,15,0] = .1*np.sin(t*9)
    p[15,16,0] = 1
    c=np.ones((30,33))
    old,_=_fill_and_smooth(p,c,.8,30)
    new,_=_fill_and_smooth(p,c,.8,30,preserve_details=True)
    assert np.mean(abs(new[:,15]-p[:,15])) < np.mean(abs(old[:,15]-p[:,15]))
    assert new[15,16,0] < .4


def test_projection_baked_fk_and_original_confidence_remain_consistent():
    m=stabilize_motion(create_demo_motion(fps=12,seconds=1))
    original=copy.deepcopy(m)
    q=np.array([f['positions'] for f in m['frames']])
    im=np.zeros((len(q),33,3)); c=np.ones((len(q),33))
    for mp,j in {11:6,12:10,13:7,14:11,15:8,16:12,23:13,24:17}.items():
        im[:,mp,:2]=q[:,j,:2]*[.4,-.4]+.5
    m2=refine_projected_arms(m,im,c,1000,1000)
    _,_,fk=solve_rotations(m2)
    np.testing.assert_allclose(fk,[f['positions'] for f in m2['frames']],atol=2e-6)
    assert m == original
    assert [f.get('confidence') for f in m2['frames']] == [f.get('confidence') for f in m['frames']]


def test_projection_does_not_reintroduce_unobserved_axial_spin():
    from scipy.spatial.transform import Rotation
    from motionbridge.exporters import default_rest_positions
    from motionbridge.stability import fk_positions
    m=create_demo_motion(fps=12,seconds=1)
    rest=default_rest_positions(); n=len(m['frames'])
    world=np.tile(np.eye(3),(n,len(rest),1,1))
    world[:,6]=Rotation.from_euler('x',(np.arange(n)*60)[:,None],degrees=True).as_matrix()
    poses=fk_positions(world,m['parents'],rest,np.zeros((n,3)))
    m['world_rotations']=world.tolist()
    for frame,p in zip(m['frames'],poses): frame['positions']=p.tolist()
    m['rotation_stabilization']={'enabled':True,'limb_speed_limit_deg_s':720}
    im=np.zeros((n,33,3))
    for mp,j in {11:6,12:10,13:7,14:11,15:8,16:12,23:13,24:17}.items():
        im[:,mp,:2]=poses[:,j,:2]*[.4,-.4]+.5
    out=refine_projected_arms(m,im,np.ones((n,33)),1000,1000)
    arm=np.array(out['world_rotations'])[:,6]
    assert np.max(np.linalg.norm(arm-arm[:1],axis=(1,2))) < .01


def test_independent_support_reaches_motion_and_target_weights_without_changing_confidence():
    from motionbridge.exporters import landmarks_to_motion, default_rest_positions
    rest=default_rest_positions(); n=4
    world=np.zeros((n,33,3))
    for mp,j in {11:6,12:10,13:7,14:11,15:8,16:12,23:13,24:17,25:14,26:18,27:15,28:19,31:16,32:20,7:4,8:4}.items():
        world[:,mp]=rest[j]*[1,-1,-1]
    im=world.copy(); im[:,:,:2]=world[:,:,:2]/2.5+.5
    c=np.ones((n,33)); c[:,15]=.1
    support=np.zeros((n,2)); support[:,0]=.85
    motion=landmarks_to_motion(world,im,c,30,{'width':1000,'height':1000},
        {'refine_motion':True,'enhance_motion':True,'stabilize_rotations':True},hand_support=support)
    assert all(f['confidence'][8]==.1 and f['observation_weights'][8]==.85 for f in motion['frames'])
    assert motion['schema_version']=='1.3' and motion['projection_refinement']['enabled']
    solve_rotations(motion)
