"""Small CPU fixtures exercise data semantics without mocking simulator output."""
import json

import h5py
import numpy as np
import pytest

from burger.data import canonical as C
from burger.data.exporters import writers
from burger.data.inspection import check_structure


def make_canonical(path, *, bases=((-0.615, 0, 0),), wrist=False):
    info = C.DatasetInfo(
        backend="maniskill", language="Draw a square", control_freq=20, robot="panda_stick",
        cameras=["agentview_rgb"] + (["eye_in_hand_rgb"] if wrist else []),
        action_space={"type": "maniskill_pd_ee_pose_abs", "dim": 6},
    )
    tips = np.array([[0.4, 0.01, 0.02], [0.41, 0.02, 0.02], [0.42, 0.03, 0.03], [0.43, 0.04, 0.08]])
    targets = tips + [0.005, 0.001, 0.0]
    images = np.zeros((4, 32, 48, 3), dtype=np.uint8)
    images[:, :16, :, 0] = 220
    images[:, 16:, :, 2] = 200
    with C.CanonicalWriter(path, info) as out:
        for index, base in enumerate(bases):
            obs = dict(agentview_rgb=images, eef_pos=tips + base, eef_quat_wxyz=np.tile([0., 1., 0., 0.], (4, 1)),
                       pen_tip_painter=tips, pen_tip_world=tips + base)
            if wrist:
                obs["eye_in_hand_rgb"] = images.copy()
            out.add_episode(C.Episode(
                obs=obs, actions=np.column_stack((targets, np.tile([np.pi, 0, 0], (4, 1)))),
                arrays={"targets_world": targets + base}, meta={"svg_index": index, "success": False},
            ))
    return tips, targets, images


def test_frame_mapping_is_invariant_to_backend_world_translation(tmp_path):
    path = tmp_path / "input.h5"
    tips, targets, _ = make_canonical(path, bases=((-0.615, 0, 0), (-0.7645, 0, 0.912)))
    assert C.validate_file(path, require_painter=True)["total_steps"] == 8
    with h5py.File(path) as f:
        state0, action0 = writers.painter_state_action(f["episodes/ep_0"], 4)
        state1, action1 = writers.painter_state_action(f["episodes/ep_1"], 4)
    np.testing.assert_allclose(state0, state1, atol=1e-7)
    np.testing.assert_allclose(action0, action1, atol=1e-7)
    np.testing.assert_allclose(state0[:, :3], tips + [-0.615, 0, 0])
    np.testing.assert_allclose(action0[:, :3], targets)
    np.testing.assert_allclose(action0[:, 3:], np.tile([np.pi, 0, 0, 1], (4, 1)))


@pytest.mark.parametrize("corruption, message", [
    ("nan_action", "NaN"),
    ("bad_quaternion", "unit quaternions"),
    ("wrong_total", "totals"),
    ("short_targets", "targets_world"),
    ("moving_base", "constant translation"),
    ("missing_camera", "required observations"),
    ("incomplete", "not completed"),
    ("wrong_version", "version"),
])
def test_corrupt_canonical_is_rejected_before_export(tmp_path, corruption, message):
    path = tmp_path / "input.h5"
    make_canonical(path)
    with h5py.File(path, "r+") as f:
        g = f["episodes/ep_0"]
        if corruption == "nan_action":
            g["actions"][0, 0] = np.nan
        elif corruption == "bad_quaternion":
            g["obs/eef_quat_wxyz"][0] = [0, 0, 0, 0]
        elif corruption == "wrong_total":
            f.attrs["total_steps"] = 20
        elif corruption == "short_targets":
            del g["arrays/targets_world"]
            g.create_dataset("arrays/targets_world", data=np.zeros((3, 3)))
        elif corruption == "moving_base":
            g["obs/pen_tip_world"][1, 0] += 0.1
        elif corruption == "missing_camera":
            del g["obs/agentview_rgb"]
        elif corruption == "incomplete":
            f.attrs["painter_write_complete"] = False
        elif corruption == "wrong_version":
            f.attrs["painter_canonical_version"] = "unknown"
    with pytest.raises(ValueError, match=message):
        writers.write_lerobot(path, tmp_path, "output")
    assert not (tmp_path / "output").exists()


def test_failed_recording_does_not_look_complete(tmp_path):
    path = tmp_path / "broken.h5"
    info = C.DatasetInfo("isaac", "Draw", 20, "panda", ["agentview_rgb"], {"dim": 6})
    with pytest.raises(RuntimeError):
        with C.CanonicalWriter(path, info):
            raise RuntimeError("simulator failed")
    with pytest.raises(ValueError, match="not completed"):
        C.validate_file(path)


def test_legacy_hdf5_preserves_pixels_and_labels_approximations(tmp_path):
    path = tmp_path / "input.h5"
    _, _, images = make_canonical(path)
    output = tmp_path / "legacy.hdf5"
    writers.write_libero_hdf5(path, output)
    assert check_structure(str(output), None)
    with h5py.File(output) as f:
        data = f["data"]
        demo = data["demo_0"]
        np.testing.assert_array_equal(demo["obs/agentview_rgb"][:], images[:, ::-1])
        assert demo["actions"].shape == (4, 7)
        assert not data.attrs["painter_simulator_replayable"]
        assert "not measured task success" in data.attrs["painter_reward_semantics"]
        assert "eye_in_hand_rgb" in json.loads(data.attrs["painter_missing"])
        assert not demo.attrs["success"]


def test_structure_validator_rejects_missing_fields_and_validates_native_actions(tmp_path):
    source = tmp_path / "input.h5"
    make_canonical(source)
    output = tmp_path / "native.hdf5"
    writers.write_libero_hdf5(source, output, action_mode="native")
    assert check_structure(str(output), None)
    with h5py.File(output, "r+") as f:
        del f["data/demo_0/actions"]
    assert not check_structure(str(output), None)


def test_missing_targets_does_not_destroy_existing_export(tmp_path):
    path = tmp_path / "input.h5"
    make_canonical(path)
    with h5py.File(path, "r+") as f:
        del f["episodes/ep_0/arrays/targets_world"]
    output = tmp_path / "existing.hdf5"
    output.write_bytes(b"previous complete export")
    with pytest.raises(ValueError, match="targets_world"):
        writers.write_libero_hdf5(path, output, overwrite=True)
    assert output.read_bytes() == b"previous complete export"


def test_heterogeneous_camera_presence_is_rejected(tmp_path):
    path = tmp_path / "input.h5"
    make_canonical(path, bases=((-0.615, 0, 0), (-0.615, 0, 0)), wrist=True)
    with h5py.File(path, "r+") as f:
        del f["episodes/ep_1/obs/eye_in_hand_rgb"]
    with pytest.raises(ValueError, match="camera presence"):
        C.validate_file(path)
