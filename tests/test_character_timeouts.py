import json

import pytest

from motionbridge import retarget


def test_long_clip_budgets_cover_measured_per_frame_cost():
    short = retarget.character_timeouts(313, True)
    long = retarget.character_timeouts(1801, True)
    # 901-frame real bake took ~1208 seconds. A 60-second/30fps clip
    # must not retain the old fixed 1800-second ceiling.
    assert long[0] > 3000 and long[1] > 600
    assert all(a <= b for a, b in zip(short, long))
    assert retarget.character_timeouts(1801, False)[0] == 600


@pytest.mark.parametrize('count', [0, -1, True, 1.5])
def test_invalid_frame_count_rejected(count):
    with pytest.raises(ValueError):
        retarget.character_timeouts(count, True)


def test_export_forwards_long_budgets_and_cancellation(tmp_path, monkeypatch):
    motion = tmp_path / 'motion.json'
    motion.write_text(json.dumps({'frames': [{}] * 1801}), encoding='utf-8')
    (tmp_path / 'saved_project_audit.json').write_text('{"passed": true}', encoding='utf-8')
    for name in ['Nahida_MotionBridge_Dance.blend', 'character_preview.png']:
        (tmp_path / name).write_bytes(b'x' * 1001)
    calls = []
    cancelled = lambda: False

    def capture(blender, script, arguments, log_path, cancel, timeout=600):
        calls.append((script.name, cancel, timeout))

    monkeypatch.setattr(retarget, 'run_script', capture)
    report = retarget.export_character(tmp_path, 'blender',
        {'scene_path': 'source.blend', 'weights_scene_path': 'weights.blend'},
        motion, tmp_path, cancelled, lambda *_: None)
    assert report['passed']
    assert [c[0] for c in calls] == ['retarget_nahida.py', 'audit_retarget_review.py']
    assert all(c[1] is cancelled for c in calls)
    assert calls[0][2] > 3000 and calls[1][2] > 600
