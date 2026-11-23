import numpy as np
import pytest
from motionbridge.world_retarget import world_root_offsets


def test_target_root_preserves_scaled_travel_and_jump_without_ground_reset():
    pelvis = np.array([[2.,3.,1.], [2.4,2.8,1.3], [2.8,3.1,1.1]])
    result, report = world_root_offsets(pelvis, 1., .5, .08)
    assert np.allclose(np.diff(result, axis=0), np.diff(pelvis, axis=0)*.5)
    assert np.allclose(result[0], [0,0,.08])
    assert result[1,2] == pytest.approx(.23)
    assert report['per_frame_ground_reset'] is False
    assert report['metric_accuracy_measured'] is False


def test_target_root_rejects_bad_body_scale_and_nonfinite_positions():
    with pytest.raises(ValueError):
        world_root_offsets([[0,0,1]], 0, .5, 0)
    with pytest.raises(ValueError):
        world_root_offsets([[0,float('nan'),1]], 1, .5, 0)
