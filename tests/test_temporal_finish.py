import numpy as np
from scipy.spatial.transform import Rotation
from motionbridge.temporal import smooth_rotations, finish_motion
from motionbridge.stability import stabilize_motion
from motionbridge.exporters import create_demo_motion, solve_rotations


def test_post_ik_filter_rejects_persistent_alternating_jitter_without_lagging_fast_sweep():
    fps=30; t=np.arange(90)/fps
    clean=60*np.sin(2*np.pi*.7*t)
    noisy=clean+12*(-1.)**np.arange(len(t))
    world=Rotation.from_euler('z',noisy[:,None],degrees=True).as_matrix()[:,None]
    out=smooth_rotations(world,fps)[:,0]
    angle=Rotation.from_matrix(out).as_euler('xyz',degrees=True)[:,2]
    core=slice(8,-8)
    assert np.sqrt(np.mean((angle[core]-clean[core])**2))<3
    assert np.ptp(angle[core])>.94*np.ptp(clean[core])
    assert abs(np.argmax(angle[core])-np.argmax(clean[core]))<=1
    np.testing.assert_allclose(np.linalg.det(out),1,atol=1e-10)


def test_final_motion_keeps_fixed_fk_and_does_not_retain_stale_hand_rotation():
    m=stabilize_motion(create_demo_motion(fps=30,seconds=1))
    out=finish_motion(m)
    _,world,poses=solve_rotations(out)
    names=out['joint_names']; parents=out['parents']; rest=np.array(out['rest_positions'])
    for j,p in enumerate(parents):
        if p>=0:
            np.testing.assert_allclose(np.linalg.norm(poses[:,j]-poses[:,p],axis=-1),np.linalg.norm(rest[j]-rest[p]),atol=1e-6)
    np.testing.assert_allclose(world[:,names.index('LeftHand')],world[:,names.index('LeftForeArm')])
    assert finish_motion(out)==out
    assert 'temporal_finish' not in m
