import numpy as np
import pytest
from motionbridge.global_motion import gvhmr_to_motion, SMPL22_NAMES, SMPL22_PARENTS
from motionbridge.exporters import default_rest_positions, JOINT_NAMES, solve_rotations, export_bvh

def sample_arrays():
    standard = dict(zip(JOINT_NAMES, default_rest_positions()))
    standard["Spine2"] = (standard["Spine"] + standard["Chest"]) * .5
    rest = np.array([standard[n] for n in SMPL22_NAMES])
    rest += [0.0, -.08, 0.0]  # Nonzero native pelvis offset catches double-offset bugs.
    positions = np.stack((rest, rest * [-1., 1., -1.]))
    translation = np.array([[1., 1., 2.], [2., 1., 3.]])
    positions += translation[:, None]
    return dict(world_body_pose=np.zeros((2, 63)),
                world_global_orient=np.array([[0., 0., 0.], [0., np.pi, 0.]]),
                world_transl=translation, joints_world=positions,
                rest_joints=np.repeat(rest[None], 2, axis=0), parents=np.array(SMPL22_PARENTS), fps=np.array(30.))

def write_motion(tmp_path, arrays):
    path = tmp_path / "world.npz"
    np.savez(path, **arrays)
    return gvhmr_to_motion(path)

def test_translation_and_back_turn_survive_json_and_bvh(tmp_path):
    motion = write_motion(tmp_path, sample_arrays())
    local, world, joints = solve_rotations(motion)
    assert motion["root_motion"] and not motion["confidence_available"]
    np.testing.assert_allclose(joints[:, 0], [[1., .92, 2.], [2., .92, 3.]])
    np.testing.assert_allclose(world[1, 0] @ [0., 0., 1.], [0., 0., -1.], atol=1e-12)
    export_bvh(motion, tmp_path / "world.bvh")
    rows = (tmp_path / "world.bvh").read_text().split("Frame Time: ")[1].splitlines()[1:]
    np.testing.assert_allclose([list(map(float, row.split()[:3])) for row in rows], [[1,1,2],[2,1,3]])

@pytest.mark.parametrize("fault", ["root_reset", "wrong_joints", "changing_shape", "nan", "hierarchy"])
def test_invalid_world_data_is_rejected(tmp_path, fault):
    arrays = sample_arrays()
    if fault == "root_reset":
        arrays["world_transl"][:] = 0
    elif fault == "wrong_joints":
        arrays["joints_world"][1, 20, 2] += .2
    elif fault == "changing_shape":
        arrays["rest_joints"][1, 16, 0] += .1
    elif fault == "nan":
        arrays["world_body_pose"][0, 0] = np.nan
    else:
        arrays["parents"][21] = 18
    with pytest.raises(ValueError):
        write_motion(tmp_path, arrays)
