"""Generation preflight and failed-run output preservation without a simulator."""
from pathlib import Path
import subprocess
import sys

import pytest

from burger.pipeline import generate


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("command", [[], ["generate"], ["export"], ["check"], ["import-maniskill"]])
def test_public_help_without_simulator_environment(command, tmp_path):
    """The installed public commands work outside the source directory."""
    result = subprocess.run(
        [sys.executable, "-m", "burger", *command, "--help"],
        cwd=tmp_path, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "Traceback" not in result.stderr


def arguments(tmp_path, *extra):
    return ["--backend", "isaac", "--svg_dir", str(ROOT / "examples/svg"),
            "--output_dir", str(tmp_path), "--source_desc", "draw the supplied geometric sketch", *extra]


@pytest.mark.parametrize("extra", [
    ["--format", "typo"], ["--parallel", "0"], ["--object_name", "../escaped"],
    ["--source_desc", ""], ["--appearance_random", "mild"], ["--max_draw_dist", "nan"],
    ["--backend_option", "a=1", "--backend_option", "a=2"],
    ["--format", "canonical,canonical"], ["--end_signal_length", "0"],
    ["--image_hw", "0", "128"], ["--delta_gain", "nan"], ["--lerobot_repo_name", "../outside"],
])
def test_rejects_invalid_arguments_before_backend(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(generate, "get_backend", lambda _: pytest.fail("simulator started"))
    with pytest.raises(SystemExit):
        generate.main(arguments(tmp_path, *extra))


def test_reexec_preserves_caller_relative_paths_and_explicit_argv(tmp_path, monkeypatch):
    (tmp_path / "input").mkdir()
    py = tmp_path / "python"
    py.touch()
    monkeypatch.chdir(tmp_path)
    argv = ["--backend", "isaac", "--svg_dir", "input", "--output_dir", "out",
            "--source_desc", "draw", "--summary_json", "summary.json"]
    args = generate.build_arg_parser().parse_args(argv)
    generate.validate_args(args)

    class MissingBackend:
        available = staticmethod(lambda: False)
        venv_python = staticmethod(lambda: py)

    calls = []
    monkeypatch.setattr(generate.subprocess, "call", lambda cmd, **kw: calls.append((cmd, kw)) or 0)
    generate.maybe_reexec(args, MissingBackend, argv)
    cmd, _ = calls[0]
    assert cmd[-6:] == ["--svg_dir", str(tmp_path / "input"), "--output_dir", str(tmp_path / "out"),
                        "--summary_json", str(tmp_path / "summary.json")]


def test_failed_generation_preserves_previous_canonical(tmp_path, monkeypatch):
    output = tmp_path / "drawing/isaac"
    output.mkdir(parents=True)
    canonical = output / "drawing.canonical.h5"
    canonical.write_bytes(b"previous successful output")

    class FailingBackend:
        available = staticmethod(lambda: True)

        def generate(self, cfg, path, work_dir, log):
            path.write_bytes(b"incomplete")
            raise RuntimeError("worker failed")

    monkeypatch.setattr(generate, "get_backend", lambda _: FailingBackend)
    with pytest.raises(RuntimeError, match="worker failed"):
        generate.main(arguments(tmp_path, "--overwrite", "--format", "canonical"))
    assert canonical.read_bytes() == b"previous successful output"
    assert not list(output.glob("*.pending"))
    assert not (output / "generation_summary.json").exists()
