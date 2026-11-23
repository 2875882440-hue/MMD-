import copy

import numpy as np
import pytest

from motionbridge.exporters import (
    JOINT_NAMES, PARENTS, _length, create_demo_motion, default_rest_positions,
    landmarks_to_motion, solve_rotations,
)
from motionbridge.refinement import align_image_observations, fit_two_bone


def observations(width=720, height=1280, frames=8):
    rest = default_rest_positions()
    world = np.zeros((frames, 33, 3))
    mapping = {11: 6, 13: 7, 15: 8, 12: 10, 14: 11, 16: 12,
               23: 13, 25: 14, 27: 15, 31: 16, 24: 17, 26: 18, 28: 19, 32: 20, 7: 4, 8: 4}
    for mp, joint in mapping.items():
        world[:, mp] = rest[joint] * [1, -1, -1]
    image = world.copy()
    image[..., 0] = .5 + world[..., 0] / 2.5 * height / width
    image[..., 1] = .5 + world[..., 1] / 2.5
    return world, image, np.ones((frames, 33))


@pytest.mark.parametrize("width,height", [(720, 1280), (1920, 1080), (640, 640)])
def test_image_alignment_corrects_arm_height_with_aspect_ratio(width, height):
    truth, image, confidence = observations(width, height)
    damaged = truth.copy()
    damaged[:, [13, 15, 14, 16], 1] += .25
    original = damaged.copy()
    fixed, report = align_image_observations(damaged, image, confidence, width, height, 30)
    before = np.linalg.norm(damaged[:, 15, :2] - truth[:, 15, :2], axis=-1).mean()
    after = np.linalg.norm(fixed[:, 15, :2] - truth[:, 15, :2], axis=-1).mean()
    assert after < before * .15
    np.testing.assert_allclose(fixed[..., 2], damaged[..., 2])
    np.testing.assert_equal(damaged, original)
    assert report["applied"]


def test_low_confidence_image_outlier_cannot_drag_wrist():
    world, image, confidence = observations()
    image[:, 15, :2] += 5
    confidence[:, 15] = .1
    fixed, _ = align_image_observations(world, image, confidence, 720, 1280, 30)
    np.testing.assert_allclose(fixed[:, 15], world[:, 15])
    assert np.all(confidence[:, 15] == .1)


def test_unreliable_or_missing_camera_calibration_falls_back():
    world, image, confidence = observations()
    for width, scores in [(0, confidence), (720, confidence * .1)]:
        fixed, report = align_image_observations(world, image, scores, width, 1280, 30)
        assert report["applied"] is False
        np.testing.assert_equal(fixed, world)


@pytest.mark.parametrize("target", [[.3, .2, .1], [5., 0, 0], [0, 0, 0], [.01, 0, 0]])
def test_ik_keeps_lengths_for_reachable_unreachable_and_folded_targets(target):
    middle, end = fit_two_bone(np.zeros(3), np.array(target), np.array([.2, .2, 0]), .3, .25, np.array([.3, 0, 0]))
    assert np.isfinite(middle).all() and np.isfinite(end).all()
    assert np.linalg.norm(middle) == pytest.approx(.3, abs=1e-7)
    assert np.linalg.norm(end - middle) == pytest.approx(.25, abs=1e-7)
    if .05 < np.linalg.norm(target) < .55:
        np.testing.assert_allclose(end, target, atol=1e-7)


def test_endpoint_solver_recovers_reachable_hand_and_keeps_legacy_unchanged():
    legacy = create_demo_motion(fps=12, seconds=1)
    for frame in legacy["frames"]:
        frame["positions"] = default_rest_positions().tolist()
        frame["positions"][8] = (np.asarray(frame["positions"][8]) + [-.18, .15, .12]).tolist()
    target = np.asarray([f["positions"][8] for f in legacy["frames"]])
    before = copy.deepcopy(legacy)
    legacy_fk = solve_rotations(legacy)[2]
    improved = copy.deepcopy(legacy)
    improved["solver"] = "endpoint_ik_v2"
    local, _, fk = solve_rotations(improved)
    assert np.mean(np.linalg.norm(legacy_fk[:, 8] - target, axis=-1)) > .02
    np.testing.assert_allclose(fk[:, 8], target, atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(local), 1, atol=1e-7)
    assert legacy == before
    np.testing.assert_array_equal(solve_rotations(legacy)[2], legacy_fk)


def test_low_confidence_endpoint_is_not_forced_to_bad_target():
    motion = create_demo_motion(fps=12, seconds=1)
    for frame in motion["frames"]:
        frame["positions"][8] = [3, 3, 3]
        frame["confidence"][8] = .1
    original_fk = solve_rotations(motion)[2]
    motion["solver"] = "endpoint_ik_v2"
    refined_fk = solve_rotations(motion)[2]
    np.testing.assert_allclose(refined_fk[:, 8], original_fk[:, 8], atol=1e-7)


def test_low_confidence_frames_do_not_set_bone_proportions():
    a, b = np.zeros((10, 3)), np.zeros((10, 3))
    b[:8, 0], b[8:, 0] = .5, .25
    confidence = np.array([.1] * 8 + [1.] * 2)
    assert _length(a, b, .3, .1, .6, confidence) == pytest.approx(.25)


def test_refined_preview_equals_export_fk_and_confidence_is_preserved():
    world, image, confidence = observations()
    world[:, 15, 1] += .2
    confidence[3, 13] = .1
    motion = landmarks_to_motion(world, image, confidence, 30, {"width": 720, "height": 1280}, {})
    assert motion["positions_are_solved"] is True
    assert motion["solver"] == "endpoint_ik_v2"
    assert motion["frames"][3]["confidence"][JOINT_NAMES.index("LeftForeArm")] == .1
    preview = np.asarray([f["positions"] for f in motion["frames"]])
    exported = solve_rotations(motion)[2]
    np.testing.assert_allclose(preview, exported, atol=1e-6)
    rest = np.asarray(motion["rest_positions"])
    for child, parent in enumerate(PARENTS):
        if parent >= 0:
            np.testing.assert_allclose(np.linalg.norm(preview[:, child] - preview[:, parent], axis=-1),
                                       np.linalg.norm(rest[child] - rest[parent]), atol=2e-7)
    np.testing.assert_allclose(preview[:, 0], 0)


def test_opt_out_keeps_original_positions():
    world, image, confidence = observations()
    motion = landmarks_to_motion(world, image, confidence, 30, {"width": 720, "height": 1280}, {"refine_motion": False})
    assert motion["solver"] == "direction_v1"
    np.testing.assert_allclose(motion["frames"][0]["positions"][8], world[0, 15] * [1, -1, -1], atol=1e-7)
