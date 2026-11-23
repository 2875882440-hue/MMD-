"""Centered rotation averaging after IK; NumPy-only for Blender reuse.

Rate limiting alone does not remove alternating sub-limit jitter. Average short
windows on SO(3), then project back to proper rotations, preserving fixed-bone FK.
"""
import copy
import numpy as np


def smooth_values(values, fps, seconds=.065):
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return values.copy()
    sigma = max(.5, fps * seconds)
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-.5 * (offsets / sigma)**2)
    weights /= weights.sum()
    padded = np.pad(values, [(radius, radius)] + [(0, 0)] * (values.ndim-1), mode='edge')
    return sum(w * padded[k:k+len(values)] for k, w in enumerate(weights))


def smooth_rotations(world, fps, seconds=.065):
    averaged = smooth_values(world, fps, seconds)
    u, _, vh = np.linalg.svd(averaged)
    correction = np.ones(averaged.shape[:-2] + (3,))
    correction[..., -1] = np.linalg.det(u @ vh)
    return (u * correction[..., None, :]) @ vh


def finish_motion(motion):
    from .exporters import solve_rotations
    from .stability import fk_positions
    if motion.get('temporal_finish', {}).get('enabled'):
        return copy.deepcopy(motion)
    _, world, poses = solve_rotations(motion)
    final = smooth_rotations(world, motion['fps'])
    # No observed hand roll exists. A stale pre-IK hand orientation would twist
    # against its newly corrected forearm, so terminal hands inherit it.
    names = motion['joint_names']
    for side in ['Left', 'Right']:
        final[:, names.index(side+'Hand')] = final[:, names.index(side+'ForeArm')]
    roots = smooth_values(poses[:, 0], motion['fps'])
    points = fk_positions(final, motion['parents'], np.asarray(motion['rest_positions']), roots)
    result = copy.deepcopy(motion)
    result['world_rotations'] = final.round(9).tolist()
    for frame, p in zip(result['frames'], points):
        frame['positions'] = p.round(7).tolist()
    result['temporal_finish'] = {'enabled':True, 'method':'centered_so3_post_ik_v1', 'sigma_seconds':.065,
        'mean_position_change_m':float(np.linalg.norm(points-poses,axis=-1).mean()),
        'note':'Final short-window smoothing after projection/contact IK. Can soften brief gestures and move contact endpoints; not accuracy.'}
    return result
