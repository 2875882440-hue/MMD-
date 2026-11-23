"""Temporal rotation cleanup for newly processed motion; never edits old jobs."""
from __future__ import annotations
import copy
import numpy as np


def fk_positions(world, parents, rest, roots):
    positions = np.empty(world.shape[:2] + (3,), dtype=float)
    positions[:, 0] = roots
    for joint, parent in enumerate(parents):
        if parent >= 0:
            positions[:, joint] = positions[:, parent] + world[:, parent] @ (rest[joint] - rest[parent])
    return positions


def stabilize_motion(motion):
    """Return a new document with explicit world rotations and matching fixed-bone FK.

    Roll on a singly observed segment is unobservable: retain it by parallel
    transport. A centered medoid rejects isolated orientation spikes, then a
    frame-rate-independent rate limit bounds large remaining jumps. Neither
    operation estimates ground-truth accuracy; rapid motion can be attenuated.
    SciPy is needed only at processing time, never to read baked tracks in Blender.
    """
    from scipy.spatial.transform import Rotation
    from .exporters import validate_motion, solve_rotations, _swing
    names, parents, rest, poses, fps = validate_motion(motion)
    if motion.get('rotation_stabilization', {}).get('enabled') and 'world_rotations' in motion:
        solve_rotations(motion)  # validate existing tracks; do not filter twice
        return copy.deepcopy(motion)
    _, initial, solved = solve_rotations(motion)
    world = initial.copy()
    children = [[k for k, p in enumerate(parents) if p == j] for j in range(len(names))]
    for joint, links in enumerate(children):
        if len(links) == 1:
            child = links[0]
            offset = rest[child] - rest[joint]
            previous = initial[0, joint]
            for f in range(len(poses)):
                target = solved[f, child] - solved[f, joint]
                current = _swing(previous @ offset, target) @ previous
                current = Rotation.from_matrix(current).as_matrix()
                world[f, joint] = current
                previous = current
        elif not links and parents[joint] >= 0:
            world[:, joint] = world[:, parents[joint]]
    changes = []
    max_speed = 0.
    adaptive = bool(motion.get('settings', {}).get('enhance_motion', False))
    confidence = np.asarray([f.get('confidence',[1.]*len(names)) for f in motion['frames']])
    trusted_fast_samples = 0
    torso = {'Hips', 'Spine', 'Chest', 'Neck', 'Head'}
    for joint, name in enumerate(names):
        raw = Rotation.from_matrix(world[:, joint]).as_quat()
        trusted = np.zeros(len(raw),dtype=bool)
        if adaptive and len(raw)>2:
            delta = Rotation.from_matrix(np.swapaxes(world[:-1,joint],-1,-2)@world[1:,joint]).as_rotvec()
            magnitude = np.linalg.norm(delta,axis=-1)
            coherence = np.sum(delta[:-1]*delta[1:],axis=-1)/np.maximum(magnitude[:-1]*magnitude[1:],1e-9)
            ratio = np.minimum(magnitude[:-1],magnitude[1:])/np.maximum(np.maximum(magnitude[:-1],magnitude[1:]),1e-9)
            evidence = confidence[:,joint].copy()
            for child in children[joint]:
                evidence = np.minimum(evidence,confidence[:,child])
            projected_fast = np.zeros(len(raw)-2,dtype=bool)
            # A depth-only flip can look temporally coherent after smoothing.
            # Require corroborating image-plane angular motion before raising
            # the speed cap; confidence alone is not proof of a real fast move.
            for child in children[joint]:
                vectors = solved[:,child,:2]-solved[:,joint,:2]
                lengths = np.linalg.norm(vectors,axis=-1)
                angles = np.arctan2(vectors[:-1,0]*vectors[1:,1]-vectors[:-1,1]*vectors[1:,0],np.sum(vectors[:-1]*vectors[1:],axis=-1))
                cutoff = np.deg2rad(180 if name in torso else 360)/fps
                projected_fast |= (np.minimum(np.abs(angles[:-1]),np.abs(angles[1:]))>cutoff)&(angles[:-1]*angles[1:]>0)&(lengths[1:-1]>.03)
            trusted[1:-1] = (coherence>.6)&(ratio>.3)&(evidence[1:-1]>.8)&(evidence[:-2]>.7)&(evidence[2:]>.7)&projected_fast
            trusted_fast_samples += int(trusted.sum())
        filtered = np.empty_like(raw)
        for f in range(len(raw)):
            window = raw[max(0, f - 2):min(len(raw), f + 3)]
            scores = (1 - np.abs(window @ window.T)).sum(axis=1)
            # Tie-break toward the current frame, avoiding a systematic phase lag.
            if adaptive:
                scores += np.abs(np.arange(len(window))-(f-max(0,f-2)))*1e-10
            filtered[f] = raw[f] if trusted[f] else window[int(np.argmin(scores))]
        base_limit = 360 if name in torso else 720
        previous = filtered[0]
        for f, target in enumerate(filtered):
            if previous @ target < 0:
                target = -target
            angle = 2 * np.arccos(np.clip(previous @ target, -1., 1.))
            limit = np.deg2rad(base_limit*(2 if name in torso else 3) if trusted[f] else base_limit) / fps
            amount = min(1., limit / max(angle, 1e-12))
            half = angle / 2
            if half < 1e-8:
                result = target.copy()
            else:
                result = (np.sin((1 - amount) * half) * previous + np.sin(amount * half) * target) / np.sin(half)
            result /= np.linalg.norm(result)
            if f:
                max_speed = max(max_speed, float(np.rad2deg(2 * np.arccos(np.clip(abs(result @ previous), 0., 1.))) * fps))
            changes.append(float(np.rad2deg(2 * np.arccos(np.clip(abs(result @ raw[f]), 0., 1.)))))
            filtered[f] = result
            previous = result
        world[:, joint] = Rotation.from_quat(filtered).as_matrix()
    positions = fk_positions(world, parents, rest, solved[:, 0])
    result = copy.deepcopy(motion)
    result['schema_version'] = '1.1'
    result['world_rotations'] = world.round(9).tolist()
    result['positions_are_solved'] = True
    for frame, points in zip(result['frames'], positions):
        frame['positions'] = points.round(7).tolist()
    report = {'enabled': True, 'method': 'confidence_coherent_rotation_v2' if adaptive else 'parallel_transport_medoid_rate_limit_v1',
        'adaptive':adaptive,'trusted_coherent_samples':trusted_fast_samples,
        'quaternion_medoid_window_frames': 5, 'torso_head_speed_limit_deg_s': 720 if adaptive else 360,
        'limb_speed_limit_deg_s': 2160 if adaptive else 720, 'maximum_world_speed_deg_s': max_speed,
        'maximum_rotation_adjustment_deg': max(changes), 'median_rotation_adjustment_deg': float(np.median(changes)),
        'mean_joint_displacement_m': float(np.linalg.norm(positions - solved, axis=-1).mean()),
        'note': 'Stability diagnostics, not accuracy. Changes estimated poses and can attenuate rapid motion.'}
    result['rotation_stabilization'] = report
    result.setdefault('warnings', []).append('旋转稳定会过滤短时朝向异常并限制转速，可能减弱快速动作；不等于恢复真实手掌朝向或动作准确率。')
    return result
