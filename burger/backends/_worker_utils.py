"""Validation and cleanup shared by simulator worker launchers."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import threading
import math

import h5py


def validate_gpu_selection(device, *, backend):
    """Preserve scheduler masks; accept masked rendering only on a verified single GPU.

    CUDA ordinals can be remapped while Vulkan/EGL ordinals remain physical.
    We do not infer a cross-API mapping for masked multi-GPU hosts.
    """
    if device is None:
        device = 0
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise ValueError("GPU device must be a nonnegative integer")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        return device
    message = (
        f"{backend}: cannot safely map inherited CUDA_VISIBLE_DEVICES={mask!r} "
        "to the rendering GPU. The mask is preserved. Masked generation requires "
        "a dedicated single-GPU host, --device 0, and a mask of 0 or that GPU's "
        "full UUID; masked multi-GPU hosts are unsupported."
    )
    if device != 0 or not mask or "," in mask or (mask != "0" and not mask.startswith("GPU-")):
        raise RuntimeError(message)
    try:
        inventory = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        rows = [tuple(value.strip() for value in row.split(",")) for row in inventory.splitlines() if row.strip()]
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(message + " Could not verify the host GPU inventory with nvidia-smi.") from error
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][0] != "0" or not rows[0][1].startswith("GPU-"):
        raise RuntimeError(message)
    if mask not in ("0", rows[0][1]):
        raise RuntimeError(message)
    return 0


def validate_options(overrides, defaults, extra_names=()):
    """Catch misspelled controls and JSON type errors before launching a simulator."""
    unknown = set(overrides) - set(defaults) - set(extra_names)
    if unknown:
        raise ValueError(f"Unknown backend options: {', '.join(sorted(unknown))}")
    for key, value in overrides.items():
        default = defaults.get(key)
        if (isinstance(default, bool) or key in ("scene_textures", "scene_lighting")) and not isinstance(value, bool):
            raise ValueError(f"{key} must be a JSON boolean (true or false)")
        if key == "appearance_seed" and (not isinstance(value, int) or value < 0):
            raise ValueError("appearance_seed must be a nonnegative JSON integer")
        if isinstance(default, (float, int)) and not isinstance(default, bool):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite JSON number")
            if isinstance(default, int) and int(value) != value:
                raise ValueError(f"{key} must be an integer")
        if isinstance(default, str) and not isinstance(value, str) and key != "robot_init_qpos":
            raise ValueError(f"{key} must be a string")
    result = dict(defaults, **overrides)
    for key in ("max_dots", "timeout_s", "sim_dt", "decimation", "tcp_init_max_trials"):
        if key in result and result[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("hold_steps", "raise_steps", "end_hold_steps", "end_lift_steps", "start_from_prev_lift", "start_approach_steps", "start_settle_steps"):
        if key in result and result[key] < 0:
            raise ValueError(f"{key} cannot be negative")
    if result.get("start_from_prev_lift", 0) > 0 and result.get("arm_controller") != "pd_ee":
        raise ValueError("start_from_prev_lift requires arm_controller=pd_ee")
    for key in ("tcp_init_random", "raise_ramp", "end_lift_ramp"):
        if key in result and result[key] not in (0, 1):
            raise ValueError(f"{key} must be 0 or 1")
    for axis in ("x", "y"):
        low, high = f"tcp_init_{axis}_min", f"tcp_init_{axis}_max"
        if low in result and high in result and result[low] > result[high]:
            raise ValueError(f"{low} cannot exceed {high}")
    if "canvas_size" in result:
        size = result["canvas_size"]
        if not isinstance(size, (list, tuple)) or len(size) != 2 or any(
            not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0 for value in size
        ):
            raise ValueError("canvas_size must be two positive finite numbers")
    for key, choices in (
        ("arm_controller", ("osc", "pd_ee")),
        ("ik", ("maniskill_lm", "dls")),
        ("raise_xy", ("base", "last")),
        ("end_lift_xy", ("base", "last")),
        ("appearance_random", ("none", "mild", "wide")),
        ("antialiasing", ("Off", "FXAA")),
        ("render_mode", ("RealTimePathTracing",)),
        ("sampled_lighting", (True,)),
        ("samples_per_pixel", (1, 2, 4, 8)),
    ):
        if key in result and result[key] not in choices:
            raise ValueError(f"{key} must be one of {choices}")
    return result


def validate_parts(parts, svg_files) -> dict:
    """Require exactly one complete, nonempty episode for each requested input."""
    expected = {i: str(Path(path).resolve()) for i, path in enumerate(svg_files)}
    seen = {}
    total_steps = 0
    for part in parts:
        with h5py.File(part, "r") as stream:
            if "episodes" not in stream:
                raise RuntimeError(f"Worker output has no episodes: {part}")
            for name, episode in stream["episodes"].items():
                index = int(episode.attrs.get("svg_index", -1))
                source = str(Path(str(episode.attrs.get("svg_file", ""))).resolve())
                if index in seen or index not in expected or expected[index] != source:
                    raise RuntimeError(f"Unexpected or duplicate worker episode {name}: index={index}, source={source}")
                steps = len(episode["actions"])
                if steps < 1 or int(episode.attrs.get("num_steps", -1)) != steps:
                    raise RuntimeError(f"Empty or incomplete worker episode {name} in {part}")
                if bool(episode.attrs.get("truncated", False)):
                    raise RuntimeError(f"Truncated worker episode {name} cannot be published")
                seen[index] = source
                total_steps += steps
    if seen != expected:
        raise RuntimeError(f"Worker output is missing episodes: {sorted(set(expected) - set(seen))}")
    return {"num_episodes": len(seen), "total_steps": total_steps}


@contextmanager
def termination_as_interrupt():
    """Let launcher cleanup run on SIGTERM as it does on Ctrl-C."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt("Simulator launcher received SIGTERM")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def stop_process_group(proc):
    """Terminate Kit and its child processes, then reap the worker."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
