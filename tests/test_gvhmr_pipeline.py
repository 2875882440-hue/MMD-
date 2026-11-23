import sys
import time
from pathlib import Path
import numpy as np
import pytest
from motionbridge import gvhmr_pipeline as pipeline


def test_coco_conversion_preserves_anatomical_sides_and_pixel_units():
    kp = np.arange(51, dtype=float).reshape(17, 3)
    result = pipeline.coco_to_body18(kp, 640, 480) * [640, 480, 1]
    assert np.allclose(result[4], kp[10])  # Right wrist
    assert np.allclose(result[7], kp[9])   # Left wrist
    assert np.allclose(result[1, :2], (kp[5, :2] + kp[6, :2]) / 2)
    assert result[1, 2] == min(kp[5, 2], kp[6, 2])


def test_unknown_runtime_is_not_reported_as_gpu_success(tmp_path):
    status = pipeline.engine_status(tmp_path)
    assert not status['installed'] and status['missing']
    assert status['runtime_checked'] is False


def test_settings_never_apply_in_place_solver_to_world_motion():
    source = {'sample_fps': 12, 'smoothing': .9, 'enhance_motion': True, 'root_mode': 'in_place'}
    result = pipeline.effective_settings(source)
    assert result['sample_fps'] == 30 and result['root_mode'] == 'estimated_world_translation'
    assert result['keypoint_identity'] == 'temporal_local' and not result['enhance_motion']
    assert source['sample_fps'] == 12  # no mutation of original jobs


def test_world_corrections_can_be_disabled_independently():
    result=pipeline.effective_settings({'world_arm_correction':False,'world_foot_contacts':True})
    assert result['keypoint_identity']=='temporal' and result['world_foot_contacts']
    result=pipeline.effective_settings({'world_arm_correction':True,'world_foot_contacts':False})
    assert result['keypoint_identity']=='temporal_local' and not result['world_foot_contacts']


def test_cancel_stops_owned_child_before_it_can_write(tmp_path):
    marker = tmp_path / 'should_not_exist'
    command = [sys.executable, '-c', 'import time,pathlib;time.sleep(2);pathlib.Path(' + repr(str(marker)) + ').touch()']
    start = time.monotonic()
    with pytest.raises(InterruptedError):
        pipeline.run_worker(command, tmp_path / 'cancel.log', lambda: time.monotonic() - start > .3)
    time.sleep(2.1)
    assert not marker.exists()


def test_worker_failure_is_explicit_and_does_not_fall_back(tmp_path):
    with pytest.raises(RuntimeError, match='不会回退'):
        pipeline.run_worker([sys.executable, '-c', 'raise RuntimeError("test failure")'],
                            tmp_path / 'failure.log', lambda: False)
