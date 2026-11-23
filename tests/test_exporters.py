import copy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motionbridge.exporters import (
    JOINT_NAMES, PARENTS, _swing, create_demo_motion, default_rest_positions,
    export_bvh, landmarks_to_motion, solve_rotations, validate_motion,
)


def rest_motion():
    motion = create_demo_motion(fps=24, seconds=1 / 24)
    motion["frames"][0]["positions"] = default_rest_positions().tolist()
    return motion


def test_rest_pose_has_identity_rotations_and_exact_fk():
    motion = rest_motion()
    local, world, fk = solve_rotations(motion)
    np.testing.assert_allclose(local, np.tile(np.eye(3), (1, len(JOINT_NAMES), 1, 1)), atol=1e-7)
    np.testing.assert_allclose(fk[0], motion["rest_positions"], atol=1e-7)


def test_rigid_body_rotation_does_not_double_rotate_children():
    motion = rest_motion()
    rotation = Rotation.from_euler("xyz", [20, 55, -12], degrees=True).as_matrix()
    posed = np.asarray(motion["rest_positions"]) @ rotation.T
    motion["frames"][0]["positions"] = posed.tolist()
    local, world, fk = solve_rotations(motion)
    np.testing.assert_allclose(local[0, 0], rotation, atol=1e-7)
    np.testing.assert_allclose(local[0, 1:], np.tile(np.eye(3), (len(JOINT_NAMES) - 1, 1, 1)), atol=1e-7)
    np.testing.assert_allclose(fk[0], posed, atol=1e-7)


def test_synthetic_arm_motion_preserves_bone_lengths_and_reconstructs():
    motion = create_demo_motion(fps=12, seconds=2)
    local, world, fk = solve_rotations(motion)
    positions = np.array([frame["positions"] for frame in motion["frames"]])
    np.testing.assert_allclose(fk, positions, atol=3e-7)
    np.testing.assert_allclose(np.linalg.det(local), 1, atol=1e-7)
    assert np.max(np.abs(local[:, 7] - np.eye(3))) > .5


def test_variable_length_noisy_measurements_export_with_fixed_bone_lengths(tmp_path):
    """Real video estimates change length; the exported rig must not stretch."""
    motion = create_demo_motion(fps=12, seconds=1)
    rest = np.array(motion["rest_positions"])
    rng = np.random.default_rng(52)
    for frame in motion["frames"]:
        measured = np.array(frame["positions"])
        measured[1:] += rng.normal(0, .015, measured[1:].shape)
        measured[8] += [.08, -.03, .01]  # Deliberately inconsistent wrist distance.
        frame["positions"] = measured.tolist()
    local, world, fk = solve_rotations(motion)
    assert np.isfinite(local).all()
    np.testing.assert_allclose(np.linalg.det(local), 1, atol=1e-7)
    np.testing.assert_allclose(fk[:, 0], 0, atol=1e-9)
    for joint, parent in enumerate(PARENTS):
        if parent >= 0:
            lengths = np.linalg.norm(fk[:, joint] - fk[:, parent], axis=-1)
            np.testing.assert_allclose(lengths, np.linalg.norm(rest[joint] - rest[parent]), atol=1e-7)
    export_bvh(motion, tmp_path / "estimated-body.bvh")
    assert "Frames: 12" in (tmp_path / "estimated-body.bvh").read_text()


@pytest.mark.parametrize("vector", [[1, 0, 0], [0, 1, 0], [.2, .6, .3]])
def test_opposite_vector_swing_is_a_rotation(vector):
    a = np.array(vector, dtype=float)
    result = _swing(a, -a)
    np.testing.assert_allclose(result @ a, -a, atol=1e-8)
    assert np.linalg.det(result) == pytest.approx(1)


def test_bvh_channels_are_local_rotations_in_hierarchy_order(tmp_path):
    motion = create_demo_motion(fps=12, seconds=1)
    path = tmp_path / "motion.bvh"
    export_bvh(motion, path)
    text = path.read_text()
    assert "ROOT Hips" in text
    assert "Frames: 12" in text
    data = np.array([[float(value) for value in line.split()] for line in text.split("Frame Time: ")[1].splitlines()[1:]])
    assert data.shape == (12, 3 + 3 * len(JOINT_NAMES))
    assert np.isfinite(data).all()
    order = []
    def walk(joint):
        order.append(joint)
        for i, p in enumerate(PARENTS):
            if p == joint:
                walk(i)
    walk(0)
    local, _, _ = solve_rotations(motion)
    recovered = Rotation.from_euler("ZXY", data[:, 3:].reshape(-1, 3), degrees=True).as_matrix().reshape(12, len(JOINT_NAMES), 3, 3)
    np.testing.assert_allclose(recovered, local[:, order], atol=1e-7)


def test_landmarks_conversion_preserves_confidence_and_fixed_root():
    rng = np.random.default_rng(4)
    world = rng.normal(0, .2, (3, 33, 3))
    confidence = np.ones((3, 33))
    confidence[1, 13] = .1
    motion = landmarks_to_motion(world, world, confidence, 24, {}, {})
    assert motion["root_mode"] == "in_place"
    assert motion["frames"][1]["confidence"][7] == .1
    np.testing.assert_allclose(np.array([f["positions"][0] for f in motion["frames"]]), 0)
    expected = (world[0, 13] - (world[0, 23] + world[0, 24]) * .5) * [1, -1, -1]
    np.testing.assert_allclose(motion["frames"][0]["positions"][7], expected, atol=1e-7)
    validate_motion(motion)


def test_invalid_hierarchy_and_nan_fail_before_export(tmp_path):
    motion = rest_motion()
    motion["parents"][2] = 20
    with pytest.raises(ValueError, match="topologically"):
        export_bvh(motion, tmp_path / "bad.bvh")
    motion = rest_motion()
    motion["frames"][0]["positions"][3][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        export_bvh(motion, tmp_path / "bad.bvh")
