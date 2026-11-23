import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from motionbridge.exporters import create_demo_motion, solve_rotations, export_bvh
from motionbridge.stability import stabilize_motion


@pytest.mark.parametrize('fps', [12, 24, 30])
def test_single_frame_flip_is_bounded_without_stretching_or_mutating_input(fps):
    motion = create_demo_motion(fps=fps, seconds=1)
    rest = np.array(motion['rest_positions'])
    for f in motion['frames']:
        f['positions'] = rest.tolist()
    motion['frames'][fps//2]['positions'] = (rest @ Rotation.from_euler('y', 175, degrees=True).as_matrix().T).tolist()
    original = copy.deepcopy(motion)
    result = stabilize_motion(motion)
    local, world, fk = solve_rotations(result)
    assert motion == original
    np.testing.assert_allclose(fk, [f['positions'] for f in result['frames']], atol=2e-7)
    for child, parent in enumerate(result['parents']):
        if parent >= 0:
            np.testing.assert_allclose(np.linalg.norm(fk[:, child]-fk[:, parent], axis=-1), np.linalg.norm(rest[child]-rest[parent]), atol=2e-7)
    speed = Rotation.from_matrix((world[:-1].swapaxes(-1,-2) @ world[1:]).reshape(-1,3,3)).magnitude() * fps
    assert np.rad2deg(speed).max() < 720.001
    assert [f['confidence'] for f in result['frames']] == [f['confidence'] for f in motion['frames']]
    assert result['rotation_stabilization']['enabled']


def test_saved_tracks_preserve_roll_and_are_idempotent(tmp_path):
    result = stabilize_motion(create_demo_motion(fps=24, seconds=1))
    assert stabilize_motion(result) == result
    world = np.asarray(result['world_rotations'])
    np.testing.assert_array_equal(solve_rotations(result)[1], world)
    export_bvh(result, tmp_path / 'motion.bvh')
    assert 'Frames: 24' in (tmp_path / 'motion.bvh').read_text()


@pytest.mark.parametrize('mutation', ['shape', 'reflection', 'nonfinite', 'position'])
def test_invalid_or_stale_tracks_fail_instead_of_silently_diverging(mutation):
    result = stabilize_motion(create_demo_motion(fps=12, seconds=1))
    if mutation == 'shape':
        result['world_rotations'].pop()
    elif mutation == 'reflection':
        result['world_rotations'][0][0] = np.diag([-1.,1.,1.]).tolist()
    elif mutation == 'nonfinite':
        result['world_rotations'][0][0][0][0] = float('nan')
    else:
        result['frames'][0]['positions'][8][0] += .1
    with pytest.raises(ValueError):
        solve_rotations(result)
