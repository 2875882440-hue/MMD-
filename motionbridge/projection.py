"""Offline image-constrained arm fitting with fixed lengths and depth priors.

Image agreement is a diagnostic, not ground truth. A low-confidence body wrist
may use separately validated hand evidence; stored body confidence is unchanged.
"""
import copy
import numpy as np


def _directions(angles):
    a = np.asarray(angles).reshape(3, 2)
    return np.column_stack((np.cos(a[:, 1]) * np.cos(a[:, 0]), np.sin(a[:, 1]),
                            np.cos(a[:, 1]) * np.sin(a[:, 0])))


def _angles(vectors):
    v = vectors / np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-9)
    return np.column_stack((np.arctan2(v[:, 2], v[:, 0]), np.arcsin(np.clip(v[:, 1], -1, 1)))).ravel()


def fit_projected_chain(start, prior_points, lengths, target_xy, weights, previous=None):
    """Optimize clavicle, upper arm and forearm as three fixed-length links."""
    from scipy.optimize import least_squares
    start, prior = np.asarray(start), np.asarray(prior_points)
    lengths, weights = np.asarray(lengths), np.clip(weights, 0, 1)
    vectors = np.diff(np.vstack((start, prior)), axis=0)
    x0 = _angles(vectors)
    directions = _directions(x0)
    def residual(x):
        d = _directions(x)
        p = start + np.cumsum(d * lengths[:, None], axis=0)
        # Clavicle stays near its measured orientation; elbow/wrist image
        # locations outweigh ambiguous monocular depth when observations agree.
        image_error = ((p[:, :2] - target_xy) * (weights * [1.5, 3., 5.])[:, None]).ravel()
        depth = (p[:, 2] - prior[:, 2]) * .12
        pose = ((d - directions) * np.array([.10, .012, .012])[:, None]).ravel()
        temporal = np.zeros(9) if previous is None else ((d - previous) * (.12 + .15 * (1 - weights))[:, None]).ravel()
        return np.concatenate((image_error, depth, pose, temporal))
    if weights[1:].max() < .35:
        return prior.copy(), directions
    # Warm start from the previous fitted branch; monocular depth admits several
    # near-equivalent minima and independent initializations can alternate.
    seed = x0 if previous is None else _angles(previous)
    fitted = least_squares(residual, seed, max_nfev=24, ftol=2e-4, xtol=2e-4, gtol=2e-4)
    d = _directions(fitted.x)
    return start + np.cumsum(d * lengths[:, None], axis=0), d


def refine_projected_arms(motion, image, confidence, width, height):
    from scipy.ndimage import median_filter, gaussian_filter1d
    from scipy.spatial.transform import Rotation
    from .exporters import solve_rotations, _swing
    from .stability import fk_positions
    result = copy.deepcopy(motion)
    _, world, poses = solve_rotations(motion)
    world = world.copy()
    rest = np.asarray(motion['rest_positions'])
    parents = motion['parents']
    image, confidence = np.asarray(image), np.asarray(confidence)
    if image.shape != (len(poses), 33, 3) or confidence.shape != (len(poses), 33):
        raise ValueError('Invalid projected observations')
    xy = image[..., :2] * [width / height, -1.]
    xy -= xy[:, [23,24]].mean(axis=1)[:, None]
    # Calibrate only visible shoulders/hips, never a potentially wrong hand.
    mp, js = [11,12,23,24], [6,10,13,17]
    torso = poses[:, js, :2] - poses[:, :1, :2]
    w = np.clip((confidence[:, mp] - .4) / .4, 0, 1)
    denom = np.sum(xy[:, mp] ** 2 * w[..., None], axis=(1,2))
    scale = np.sum(xy[:, mp] * torso * w[..., None], axis=(1,2)) / np.maximum(denom, 1e-9)
    valid = (np.sum(w > .5, axis=1) >= 3) & (scale > .1) & (scale < 20) & (denom > 1e-5)
    if valid.sum() < min(3, len(poses)):
        result['projection_refinement'] = {'enabled':False, 'reason':'insufficient_camera_anchors'}
        return result
    frames = np.arange(len(poses))
    scale = np.interp(frames, frames[valid], scale[valid])
    scale = gaussian_filter1d(median_filter(scale, size=max(1, round(motion['fps'] * .15)) | 1), max(.5, motion['fps'] * .08))
    goals = xy * scale[:, None, None] + poses[:, :1, :2]
    prior = poses.copy()
    rate_limited = 0
    limits = motion.get('rotation_stabilization', {})
    cap = np.deg2rad(limits.get('limb_speed_limit_deg_s', 2160)) / motion['fps']
    for chain, observations in [([5,6,7,8], [11,13,15]), ([9,10,11,12], [12,14,16])]:
        lengths = np.linalg.norm(np.diff(rest[chain], axis=0), axis=1)
        previous = None
        for f in range(len(poses)):
            weights = np.clip((confidence[f, observations] - .3) / .5, 0, 1)
            if not valid[f]:
                weights *= .25
            target, previous = fit_projected_chain(poses[f,chain[0]], poses[f,chain[1:]], lengths,
                                                   goals[f,observations], weights, previous)
            directions = np.diff(np.vstack((poses[f,chain[0]], target)), axis=0)
            for k, joint in enumerate(chain[:-1]):
                # Carry the corrected orientation through time. Reattaching
                # each frame to its old axial roll can inject a 180-degree
                # twist although adjacent endpoint directions barely move.
                basis = world[f-1,joint] if f else world[f,joint]
                old = basis @ (rest[chain[k+1]] - rest[joint])
                rotation = _swing(old, directions[k]) @ basis
                if f and limits.get('enabled'):
                    delta = Rotation.from_matrix(rotation @ world[f-1,joint].T).as_rotvec()
                    size = np.linalg.norm(delta)
                    if size > cap:
                        rotation = Rotation.from_rotvec(delta * cap / size).as_matrix() @ world[f-1,joint]
                        rate_limited += 1
                world[f,joint] = rotation
    final = fk_positions(world, parents, rest, poses[:, 0])
    result['world_rotations'] = world.round(9).tolist()
    result['positions_are_solved'] = True
    result['schema_version'] = '1.3'
    for frame, points in zip(result['frames'], final):
        frame['positions'] = points.round(7).tolist()
    js, mp = [7,8,11,12], [13,15,14,16]
    good = (confidence[:, mp] >= .6) & valid[:, None]
    before = np.linalg.norm(prior[:,js,:2] - goals[:,mp], axis=-1) / scale[:,None]
    after = np.linalg.norm(final[:,js,:2] - goals[:,mp], axis=-1) / scale[:,None]
    result['projection_refinement'] = {'enabled':True, 'method':'fixed_length_image_depth_fit_v1',
        'valid_camera_frames':int(valid.sum()), 'rate_limited_samples':rate_limited,
        'mean_image_height_error_before':float(before[good].mean()) if good.any() else None,
        'mean_image_height_error_after':float(after[good].mean()) if good.any() else None,
        'note':'Agreement with estimated 2D observations, not labeled accuracy. Depth and occluded hands remain uncertain.'}
    return result
