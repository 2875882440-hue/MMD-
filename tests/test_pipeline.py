import numpy as np
import pytest

import motionbridge.pipeline as pipeline
from motionbridge.pipeline import (
    BODY18_LIMBS,
    _body18,
    _draw_body18,
    _fill_and_smooth,
    _longest_false_run,
    _target_dimensions,
    process_video,
)


def test_resize_preserves_landscape_portrait_and_small_inputs():
    assert _target_dimensions(1920, 1080) == (1280, 720)
    assert _target_dimensions(1080, 1920) == (720, 1280)
    assert _target_dimensions(640, 480) == (640, 480)


def test_missing_data_is_interpolated_without_inventing_confidence():
    points = np.zeros((5, 1, 3), dtype=float)
    points[:, 0, 0] = [0, np.nan, np.nan, np.nan, 4]
    confidence = np.array([[1], [0], [0], [0], [1]], dtype=float)
    before = confidence.copy()
    result, uncertain = _fill_and_smooth(points, confidence, 0, 24)
    assert result[:, 0, 0].tolist() == [0, 1, 2, 3, 4]
    assert uncertain == []
    np.testing.assert_equal(confidence, before)


def test_wholly_unobserved_joint_fails_instead_of_outputting_nan():
    with pytest.raises(RuntimeError, match="关键点"):
        _fill_and_smooth(np.full((3, 1, 3), np.nan), np.zeros((3, 1)), 0.5, 24)


def test_uncertain_fallback_is_marked_and_smoothing_reduces_jitter():
    points = np.zeros((9, 1, 3), dtype=float)
    points[4, 0, 0] = 1
    confidence = np.ones((9, 1)) * 0.1
    result, uncertain = _fill_and_smooth(points, confidence, 0.8, 24)
    assert uncertain == [0]
    assert 0 < result[4, 0, 0] < 0.5
    np.testing.assert_equal(confidence, np.ones((9, 1)) * 0.1)


def test_openpose_mapping_has_correct_sides_and_neck_confidence():
    points = np.zeros((33, 3), dtype=float)
    points[:, 0] = np.arange(33)
    confidence = np.ones(33)
    confidence[12] = 0.2
    body, score = _body18(points, confidence)
    assert body[2, 0] == 12  # right shoulder
    assert body[5, 0] == 11  # left shoulder
    assert body[1, 0] == 11.5
    assert score[1] == 0.2
    assert body[14, 0] == 5 and body[15, 0] == 2
    assert len(BODY18_LIMBS) == 17


def test_low_confidence_pose_is_not_drawn_as_certain_measurement():
    black = np.zeros((240, 320, 3), dtype=np.uint8)
    points = np.full((18, 2), 0.5)
    assert not _draw_body18(black, points, np.zeros(18)).any()
    assert _draw_body18(black, points, np.ones(18)).any()
    assert not black.any()  # source frame is not mutated


def test_longest_gap_includes_start_and_end():
    assert _longest_false_run(np.array([False, False, True, False, False, False])) == 3


def test_cancelled_task_never_opens_input(tmp_path):
    with pytest.raises(InterruptedError):
        process_video(tmp_path / "anything.mp4", tmp_path, {}, lambda *_: None, lambda: True)


def test_non_video_is_rejected_before_loading_ml(tmp_path):
    source = tmp_path / "not_video.txt"
    source.write_text("invalid", encoding="utf-8")
    with pytest.raises(RuntimeError, match="格式"):
        process_video(source, tmp_path / "out", {}, lambda *_: None, lambda: False)


def test_invalid_settings_are_rejected(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"invalid")
    with pytest.raises(RuntimeError, match="采样帧率"):
        process_video(source, tmp_path / "out", {"sample_fps": 120}, lambda *_: None, lambda: False)


def test_overlong_video_is_rejected_before_running_decoder(tmp_path, monkeypatch):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"placeholder")
    (tmp_path / "pose_landmarker_full.task").write_bytes(b"placeholder")
    monkeypatch.setattr(pipeline, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "_video_info", lambda _: {
        "width": 1920, "height": 1080, "fps": 30, "frames": 1830, "duration": 61.0,
    })
    monkeypatch.setattr(pipeline, "_run_ffmpeg", lambda *_: pytest.fail("Overlong input must not be processed"))
    with pytest.raises(RuntimeError, match="60 秒"):
        process_video(source, tmp_path / "out", {}, lambda *_: None, lambda: False)


def test_corrupt_video_is_rejected_without_loading_ml(tmp_path, monkeypatch):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"not a video")
    (tmp_path / "pose_landmarker_full.task").write_bytes(b"placeholder")
    monkeypatch.setattr(pipeline, "MODEL_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="读取视频"):
        process_video(source, tmp_path / "out", {}, lambda *_: None, lambda: False)
