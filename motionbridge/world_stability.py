"""Local upper-body denoising after GVHMR, preserving world travel and legs.

Centered SO(3) filtering is applied in parent coordinates, never to the root.
It preserves constant-rate rotation away from boundaries; brief gestures can
still soften. This is a stability correction, not recovered capture truth.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from .temporal import smooth_rotations


def stabilize_world_upper_body(motion):
    from .exporters import solve_rotations
    from .stability import fk_positions
    result = copy.deepcopy(motion)
    if motion.get('world_stability', {}).get('enabled'):
        return result
    if motion.get('root_mode') != 'estimated_world_translation':
        raise ValueError('World stability requires estimated world motion')
    _, world, poses = solve_rotations(motion)
    parents = motion['parents']; names = motion['joint_names']
    local = world.copy()
    for j, parent in enumerate(parents):
        if parent >= 0:
            local[:, j] = world[:, parent].swapaxes(-1, -2) @ world[:, j]
    selected = [names.index(n) for n in ['LeftShoulder', 'RightShoulder',
                'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm']]
    raw = local[:, selected].copy()
    averaged = smooth_rotations(raw, motion['fps'], .05)
    blend = np.ones(raw.shape[:2])
    protected = np.zeros(raw.shape[:2], dtype=bool)
    if len(raw) > 2:
        velocity = Rotation.from_matrix((raw[:-1].swapaxes(-1,-2) @ raw[1:]).reshape(-1,3,3)).as_rotvec().reshape(len(raw)-1,len(selected),3)
        speed = np.linalg.norm(velocity,axis=-1)
        coherence = np.sum(velocity[:-1]*velocity[1:],axis=-1) / np.maximum(speed[:-1]*speed[1:],1e-12)
        ratio = np.minimum(speed[:-1],speed[1:]) / np.maximum(np.maximum(speed[:-1],speed[1:]),1e-12)
        protected[1:-1] = (coherence>.65) & (ratio>.35) & (np.maximum(speed[:-1],speed[1:])*motion['fps']>np.deg2rad(450))
        # Coherent fast swings get less filtering, not a higher speed cap.
        # Smooth the selection weights so the protection itself does not pop.
        blend -= .85*gaussian_filter1d(protected.astype(float),max(.5,motion['fps']*.025),axis=0,mode='nearest')
    correction = Rotation.from_matrix((raw.swapaxes(-1,-2) @ averaged).reshape(-1,3,3)).as_rotvec().reshape(*raw.shape[:2],3)
    local[:, selected] = raw @ Rotation.from_rotvec((correction*blend[...,None]).reshape(-1,3)).as_matrix().reshape(raw.shape)
    final = world.copy()
    # Only reconstruct affected branches: root, torso and legs remain bitwise
    # identical, and the original local wrist orientation follows its forearm.
    affected = set(selected)
    for j, parent in enumerate(parents):
        if parent in affected:
            affected.add(j)
        if j in affected:
            final[:, j] = final[:, parent] @ local[:, j]
    points = fk_positions(final, parents, np.asarray(motion['rest_positions']), poses[:, 0])
    for frame, p in zip(result['frames'], points):
        frame['positions'] = p.tolist()
    result['world_rotations'] = final.tolist()
    def speeds(tracks):
        if len(tracks) < 2: return np.array([0.])
        delta = tracks[:-1].swapaxes(-1, -2) @ tracks[1:]
        return np.rad2deg(Rotation.from_matrix(delta.reshape(-1, 3, 3)).magnitude()) * motion['fps']
    result['world_stability'] = {'enabled': True, 'method': 'parent_local_centered_so3_v1',
        'sigma_seconds': .05, 'joint_names': [names[j] for j in selected],
        'coherent_fast_joint_samples':int(protected.sum()),
        'max_local_speed_before_deg_s': float(speeds(raw).max()),
        'max_local_speed_after_deg_s': float(speeds(local[:, selected]).max()),
        'root_and_leg_rotations_unchanged': True,
        'mean_joint_change_m': float(np.linalg.norm(points-poses, axis=-1).mean()),
        'note': 'No global rate limit. Short gestures may soften; this is not motion accuracy.'}
    result.setdefault('warnings', []).append('局部去抖只处理肩臂相对旋转，保留根部转身与位移；极短动作可能变软，不能修复持续错误姿态。')
    solve_rotations(result)
    return result
