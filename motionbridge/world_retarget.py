"""World-root transfer for an inspected target rig, with explicit scale/origin."""
import numpy as np


def target_ground_offsets(sole_heights, weights, fps, leg_length):
    """Bounded vertical fit to target soles; unsupported intervals keep motion.

    Returns only a bounded baseline offset with short support fades. Long
    unsupported intervals get zero correction, never a forced ground clamp.
    """
    from .temporal import smooth_values
    soles=np.asarray(sole_heights,float);weights=np.asarray(weights,float)
    if soles.ndim!=2 or soles.shape[1]!=2 or weights.shape!=soles.shape or not np.isfinite(soles).all() or not np.isfinite(weights).all():
        raise ValueError('Expected finite synchronized target sole heights and weights')
    observations=np.full(len(soles),np.nan)
    for f in range(len(soles)):
        valid=weights[f]>=.5
        if valid.any():observations[f]=.003-float(np.min(soles[f,valid]))
    good=np.flatnonzero(np.isfinite(observations))
    if len(good)<3:return np.zeros(len(soles))
    values=np.interp(np.arange(len(soles)),good,np.clip(observations[good],-.12*leg_length,.12*leg_length))
    values=smooth_values(values,fps,.06)
    # Do not carry a stance correction across a long flight/unknown interval:
    # doing so can introduce floating when the source height already recovered.
    values*=smooth_values(np.max(weights,axis=1),fps,.04)
    # Limit correction velocity without altering original root movement.
    step=.30*leg_length/fps
    for f in range(1,len(values)):values[f]=np.clip(values[f],values[f-1]-step,values[f-1]+step)
    return values


def world_root_offsets(poses, source_leg_length, target_leg_length, initial_lift):
    """Input Blender Z-up pelvis positions. Preserve all deltas, including jumps.

    The scene origin is arbitrary: align the initial pose to the existing stage,
    then apply one fixed leg-length scale. Never reground each subsequent frame.
    """
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('Expected finite pelvis positions [frames,3]')
    values = np.array([source_leg_length, target_leg_length, initial_lift])
    if not np.isfinite(values).all() or min(source_leg_length, target_leg_length) <= 1e-6:
        raise ValueError('Invalid source/target body scale')
    scale = target_leg_length / source_leg_length
    offsets = (poses - poses[:1]) * scale
    offsets[:, 2] += initial_lift
    return offsets, {'scale': float(scale), 'scale_method': 'mean two-leg segment length ratio',
                     'origin': 'first pelvis aligned to original stage; initial sole offset only',
                     'input_delta_range_m': np.ptp(poses, axis=0).tolist(),
                     'target_delta_range_m': np.ptp(offsets, axis=0).tolist(),
                     'per_frame_ground_reset': False, 'metric_accuracy_measured': False}
