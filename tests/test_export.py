import argparse
import builtins
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from burger.data.exporters import export as E
from burger.data.exporters import writers
from test_data_contract import make_canonical


def writer_args():
    parser = argparse.ArgumentParser()
    E.add_writer_args(parser)
    return parser.parse_args([])


def test_delegate_resolves_paths_before_changing_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_canonical(Path("input.h5"))
    fake_python = tmp_path / "other-env" / "python"
    fake_python.parent.mkdir()
    fake_python.touch()
    monkeypatch.setenv("PAINTER_LEROBOT_PYTHON", str(fake_python))
    original_import = builtins.__import__

    def import_without_lerobot(name, *args, **kwargs):
        if name == "lerobot":
            raise ImportError("not installed in source environment")
        return original_import(name, *args, **kwargs)

    def run(cmd, **kwargs):
        assert Path(cmd[cmd.index("--canonical") + 1]) == tmp_path / "input.h5"
        assert Path(cmd[cmd.index("--out_dir") + 1]) == tmp_path / "output"
        assert kwargs["cwd"] == str(E._REPO_ROOT)
        assert kwargs["env"]["PAINTER_EXPORT_DELEGATED"] == "1"
        result_dir = tmp_path / "output" / cmd[cmd.index("--lerobot_repo_name") + 1]
        result_dir.mkdir()
        (result_dir / "painter_export.json").write_text('{"format":"lerobot"}')

    monkeypatch.setattr(builtins, "__import__", import_without_lerobot)
    monkeypatch.setattr(subprocess, "run", run)
    assert E.export(Path("input.h5"), ["lerobot"], Path("output"), writer_args())["lerobot"]["format"] == "lerobot"


def test_unknown_format_fails_before_writing_any_output(tmp_path):
    make_canonical(tmp_path / "input.h5")
    with pytest.raises(ValueError, match="formats"):
        E.export(tmp_path / "input.h5", ["libero", "typo"], tmp_path / "output", writer_args())
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("object_name", ["../outside", "/absolute", "nested/name", r"nested\\name", ".", "", "has space"])
def test_object_names_cannot_escape_output_directory(tmp_path, object_name):
    make_canonical(tmp_path / "input.h5")
    with pytest.raises(ValueError, match="object_name"):
        E.export(tmp_path / "input.h5", ["libero"], tmp_path / "output", writer_args(), object_name=object_name)
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "outside_demo.hdf5").exists()


def test_bad_lerobot_name_fails_before_other_exports_are_written(tmp_path):
    make_canonical(tmp_path / "input.h5")
    args = writer_args()
    args.lerobot_repo_name = "../outside"
    with pytest.raises(ValueError, match="lerobot_repo_name"):
        E.export(tmp_path / "input.h5", ["libero", "lerobot"], tmp_path / "output", args)
    assert not (tmp_path / "output").exists()


def test_failed_lerobot_export_preserves_previous_dataset(tmp_path, monkeypatch):
    make_canonical(tmp_path / "input.h5")
    previous = tmp_path / "dataset"
    previous.mkdir()
    (previous / "marker").write_text("complete")

    def fail(*args, **kwargs):
        raise RuntimeError("video encoder failed")

    monkeypatch.setattr(writers, "_write_lerobot", fail)
    with pytest.raises(RuntimeError, match="encoder"):
        writers.write_lerobot(tmp_path / "input.h5", tmp_path, "dataset", overwrite=True)
    assert (previous / "marker").read_text() == "complete"
    assert not list(tmp_path.glob(".lerobot-export-*"))


@pytest.mark.export
def test_lerobot_roundtrip(tmp_path, monkeypatch):
    """Actual parquet/video export and official LeRobot reader; no simulator or GPU."""
    pytest.importorskip("lerobot", reason="Run scripts/setup_export.sh to enable the actual format roundtrip")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    source = tmp_path / "canonical.h5"
    tips, targets, _ = make_canonical(source, bases=((-0.615, 0, 0), (-0.7645, 0, 0.912)), wrist=True)
    result = writers.write_lerobot(source, tmp_path, "roundtrip", end_signal_length=2)
    output = tmp_path / "roundtrip"
    saved = json.loads((output / "painter_export.json").read_text())
    assert saved["data_contract"]["convention"] == "painter-absolute-target-v1"
    assert saved["data_contract"]["action"]["dimension"] == 8
    assert result["total_steps"] == 8
    dataset = LeRobotDataset("roundtrip", root=output, video_backend="pyav")
    assert len(dataset) == 8
    assert dataset.meta.info["codebase_version"] == "v2.1"
    for index in range(8):
        local_index = index % 4
        row = dataset[index]
        state = row["observation.state"].numpy()
        action = row["action"].numpy()
        np.testing.assert_allclose(state[:3], tips[local_index] + [-0.615, 0, 0], atol=1e-6)
        np.testing.assert_allclose(action[:3], targets[local_index], atol=1e-6)
        np.testing.assert_allclose(action[3:7], [np.pi, 0, 0, 1], atol=1e-6)
        assert action[7] == (local_index >= 2)
        assert state[7] == (local_index >= 3)
        for image_key in ("observation.image", "observation.wrist_image"):
            image = row[image_key].numpy()
            assert image.shape == (3, 32, 48)
            # Verify decoded orientation and RGB order with robust lossy-codec margins.
            assert image[0, :12].mean() > 0.7
            assert image[2, 20:].mean() > 0.6
            assert image[0, 20:].mean() < 0.15
