"""Camera-relative pose corrections shared by every uploaded video.

Image alignment is a weak-perspective prior, not recovered world translation.
Keep observation confidence separate from corrected coordinates.
"""
from __future__ import annotations

import numpy as np


def align_image_observations(world, image, confidence, width, height, fps):
    """Fuse reliable image-plane positions with the monocular depth estimate.

    All inputs still use MediaPipe's camera axes (Y down). Fit scale from the
    torso only so a wrongly reconstructed arm cannot change camera calibration.
    Image Z is deliberately ignored: it is not a metric depth observation.
    """
    from scipy.ndimage import gaussian_filter1d, median_filter

    world = np.asarray(world, dtype=float).copy()
    image, confidence = np.asarray(image, float), np.asarray(confidence, float)
    if image.shape != world.shape or confidence.shape != world.shape[:2]:
        raise ValueError("Image/world observation shapes differ")
    if not np.isfinite(image[..., :2]).all() or not np.isfinite(confidence).all():
        raise ValueError("Image observations must be finite after gap filling")
    if not (np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0):
        return world, {"applied": False, "reason": "missing_image_dimensions"}
    points = image[..., :2] * [float(width) / height, 1.]
    image_hip = (points[:, 23] + points[:, 24]) * .5
    world_hip = (world[:, 23] + world[:, 24]) * .5
    points -= image_hip[:, None]
    relative = world - world_hip[:, None]
    torso = [11, 12, 23, 24]
    weights = np.clip((confidence[:, torso] - .4) / .4, 0, 1)
    anchor_confidence = np.minimum(confidence[:, 23], confidence[:, 24])
    denom = np.sum(weights * np.sum(points[:, torso] ** 2, axis=-1), axis=-1)
    numerator = np.sum(weights * np.sum(points[:, torso] * relative[:, torso, :2], axis=-1), axis=-1)
    scale = numerator / np.maximum(denom, 1e-9)
    valid = (anchor_confidence >= .6) & (np.sum(weights > .5, axis=1) >= 3) & (denom > 1e-5) & (scale > .1) & (scale < 20)
    if np.count_nonzero(valid) < min(3, len(world)):
        return world, {"applied": False, "reason": "insufficient_torso_observations"}
    # Robust calibration follows zoom slowly, without frame-to-frame scale jumps.
    indices = np.arange(len(world))
    scale = np.interp(indices, indices[valid], scale[valid])
    typical = float(np.median(scale))
    scale = np.clip(scale, typical * .4, typical * 2.5)
    window = max(1, round(fps * .15)) | 1
    scale = median_filter(scale, size=window, mode="nearest")
    scale = gaussian_filter1d(scale, max(.5, fps * .08), mode="nearest")
    target = points * scale[:, None, None]
    # Low-confidence occluded joints retain their interpolated 3D estimate.
    blend = .9 * np.clip((confidence - .35) / .45, 0, 1)
    blend *= np.clip((anchor_confidence - .35) / .35, 0, 1)[:, None]
    # Reject wildly displaced observations; ordinary shoulder/hand differences
    # remain correctable. This limit scales with the estimated torso size.
    torso_size = np.linalg.norm((relative[:, 11] + relative[:, 12]) * .5, axis=-1)
    discrepancy = np.linalg.norm(target - relative[..., :2], axis=-1)
    blend *= np.minimum(1., np.maximum(torso_size, .15)[:, None] * 1.5 / np.maximum(discrepancy, 1e-8))
    before = relative[..., :2].copy()
    relative[..., :2] += blend[..., None] * (target - relative[..., :2])
    world = relative + world_hip[:, None]
    reliable = confidence >= .6
    return world, {
        "applied": True, "method": "confidence_weighted_weak_perspective_v1",
        "calibrated_frames": int(valid.sum()), "meters_per_image_height_median": round(float(np.median(scale)), 6),
        "image_alignment_residual_before_m": round(float(np.mean(np.linalg.norm(before - target, axis=-1)[reliable])), 6),
        "image_alignment_residual_after_m": round(float(np.mean(np.linalg.norm(relative[..., :2] - target, axis=-1)[reliable])), 6),
        "note": "Residual measures agreement with 2D estimates, not ground-truth mocap accuracy. Depth and scene translation remain ambiguous.",
    }


def fit_two_bone(start, target, pole, length_a, length_b, fallback):
    """Exact fixed-length two-bone IK with closest reachable endpoint.

    Pole is an observed elbow/knee position. Fallback provides a stable bend
    direction when the observed limb is straight; no skeleton-specific axes.
    """
    start, target, pole = np.asarray(start), np.asarray(target), np.asarray(pole)
    vector = target - start
    distance = float(np.linalg.norm(vector))
    if distance < 1e-9:
        vector = np.asarray(fallback) - start
        if np.linalg.norm(vector) < 1e-9:
            vector = np.array([1., 0., 0.])
    axis = vector / np.linalg.norm(vector)
    reach = np.clip(distance, abs(length_a - length_b) + 1e-7, length_a + length_b - 1e-7)
    along = (length_a ** 2 - length_b ** 2 + reach ** 2) / (2 * reach)
    radius = np.sqrt(max(0., length_a ** 2 - along ** 2))
    bend = pole - start
    bend -= axis * np.dot(bend, axis)
    if np.linalg.norm(bend) < 1e-6:
        bend = np.asarray(fallback) - start
        bend -= axis * np.dot(bend, axis)
    if np.linalg.norm(bend) < 1e-6:
        reference = np.eye(3)[np.argmin(np.abs(axis))]
        bend = reference - axis * np.dot(reference, axis)
    bend /= np.linalg.norm(bend)
    return start + axis * along + bend * radius, start + axis * reach


def refine_limb_endpoints(poses, initial_fk, rest, names, confidence):
    """Preserve the torso and fit hands/ankles without stretching bones."""
    result = initial_fk.copy()
    chains = [("LeftArm", "LeftForeArm", "LeftHand"), ("RightArm", "RightForeArm", "RightHand"),
              ("LeftUpLeg", "LeftLeg", "LeftFoot"), ("RightUpLeg", "RightLeg", "RightFoot")]
    for chain in chains:
        if not all(name in names for name in chain):
            continue
        a, b, c = [names.index(name) for name in chain]
        length_a, length_b = np.linalg.norm(rest[b] - rest[a]), np.linalg.norm(rest[c] - rest[b])
        previous_bend = None
        for frame in range(len(poses)):
            start = result[frame, a]
            strength = float(np.clip((min(confidence[frame, a], confidence[frame, c]) - .25) / .5, 0, 1))
            if strength <= 0:
                previous_bend = None
                continue
            target = initial_fk[frame, c] * (1 - strength) + poses[frame, c] * strength
            pole_weight = float(np.clip((confidence[frame, b] - .25) / .5, 0, 1))
            fallback = start + previous_bend if previous_bend is not None else initial_fk[frame, b]
            pole = pole_weight * poses[frame, b] + (1 - pole_weight) * fallback
            middle, end = fit_two_bone(start, target, pole, length_a, length_b, fallback)
            result[frame, b], result[frame, c] = middle, end
            previous_bend = middle - start
            if chain[2].endswith("Foot"):
                toe_name = chain[2].replace("Foot", "ToeBase")
                if toe_name in names:
                    toe = names.index(toe_name)
                    result[frame, toe] = end + initial_fk[frame, toe] - initial_fk[frame, c]
    return result
