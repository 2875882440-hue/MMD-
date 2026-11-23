"""Validated GVHMR world-motion adapter, isolated from MediaPipe pose solving.

Used by the explicitly selected experimental engine; never the default solver.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .exporters import COORDINATE_SYSTEM, solve_rotations

SMPL22_NAMES = [
    "Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg",
    "Spine2", "LeftFoot", "RightFoot", "Chest", "LeftToeBase", "RightToeBase",
    "Neck", "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm",
    "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
]
SMPL22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

def gvhmr_to_motion(path: Path, source=None):
    """Preserve world Y-up meter coordinates and all 22 joint rotations.

    A fixed body shape is required. Reject varying rest skeletons instead of
    silently stretching bones or discarding motion. No confidence is invented.
    """
    with np.load(path, allow_pickle=False) as archive:
        required = ["world_body_pose", "world_global_orient", "world_transl",
                    "joints_world", "rest_joints", "parents", "fps"]
        arrays = {key: archive[key].copy() for key in required}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("GVHMR output contains non-finite data")
    pose = arrays["world_body_pose"]
    length = len(pose)
    if not length or pose.shape != (length, 63):
        raise ValueError("Expected SMPL-X body pose [frames,63]")
    orient, transl = arrays["world_global_orient"], arrays["world_transl"]
    joints, rest = arrays["joints_world"], arrays["rest_joints"]
    if orient.shape != (length, 3) or transl.shape != (length, 3) or joints.shape != (length, 22, 3):
        raise ValueError("Invalid GVHMR world-motion shape")
    if rest.shape != (length, 22, 3) or np.max(np.abs(rest-rest[:1])) > 1e-5:
        raise ValueError("GVHMR rest skeleton must be constant across frames")
    if not np.array_equal(arrays["parents"], SMPL22_PARENTS):
        raise ValueError("Unexpected SMPL-X body hierarchy")
    fps = float(arrays["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid motion frame rate")
    if not np.allclose(joints[:, 0], transl + rest[:, 0], atol=2e-5, rtol=0):
        raise ValueError("World pelvis and model translation disagree")
    aa = np.concatenate((orient[:, None], pose.reshape(length, 21, 3)), axis=1)
    local = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(length, 22, 3, 3)
    world = local.copy()
    for joint, parent in enumerate(SMPL22_PARENTS):
        if parent >= 0:
            world[:, joint] = world[:, parent] @ local[:, joint]
    motion = {
        "schema_version": "1.1", "fps": fps, "num_frames": length,
        "is_demo": False, "coordinate_system": COORDINATE_SYSTEM,
        "joint_names": SMPL22_NAMES.copy(), "parents": SMPL22_PARENTS.copy(),
        "rest_positions": rest[0].tolist(), "world_rotations": world.tolist(),
        "frames": [{"time": i/fps, "positions": frame.tolist(), "confidence": [0.0]*22}
                   for i, frame in enumerate(joints)],
        "source": dict(source or {}), "settings": {"engine": "gvhmr_experimental"},
        "solver": "gvhmr_smplx_world", "positions_are_solved": True,
        "root_motion": True, "root_mode": "estimated_world_translation",
        "confidence_available": False,
        "warnings": ["GVHMR 世界位移是单目模型估计，尚无真实尺度或轨迹标定。",
                     "本接口不提供三维关节准确率；confidence=0 表示未知，不表示实测质量。",
                     "GVHMR/人体模型许可分别适用；此引擎仍处于本地研究验证阶段。"],
        "global_motion": {"engine": "GVHMR", "experimental": True,
                          "root_translation_range_m": np.ptp(transl, axis=0).tolist(),
                          "rotation_source": "predicted SMPL-X local rotations",
                          "front_facing_correction": False, "root_reset": False},
    }
    # Shared exporter must agree with upstream joints before publishing a motion.
    solve_rotations(motion)
    return motion
