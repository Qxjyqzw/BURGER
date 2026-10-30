"""Format writers: canonical episodes -> LIBERO ``*_demo.hdf5`` or LeRobot v2.1 dataset."""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

from burger.data import canonical as C

LIBERO_ACTION_MODES = ("auto", "native", "libero_delta")


def validate_dataset_name(name: str, argument: str = "repo_name") -> str:
    """Keep object/repository names inside their requested output directory."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"{argument} must be one folder name starting with a letter or number and containing only letters, numbers, _, . or -")
    return name


def write_libero_hdf5(canonical_path: str | Path, out_path: str | Path, image_hw=None, action_mode: str = "auto",
                      delta_gain: float = 4.0, overwrite: bool = False, image_compression: str | None = None) -> dict:
    """Atomically write the legacy LIBERO-style training layout; see docs/DATA_FORMAT.md."""
    C.validate_file(canonical_path)
    out_path = Path(out_path).resolve()
    if out_path == Path(canonical_path).resolve():
        raise ValueError("export output must differ from the canonical source")
    if out_path.exists() and not overwrite:
        raise FileExistsError(out_path)
    if not np.isfinite(delta_gain) or delta_gain <= 0:
        raise ValueError("delta_gain must be positive and finite")
    if image_hw is not None and (len(image_hw) != 2 or min(image_hw) < 1):
        raise ValueError("image_hw must contain positive height and width")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".libero-export-", dir=out_path.parent) as staging:
        temporary = Path(staging) / out_path.name
        result = _write_libero_hdf5(canonical_path, temporary, image_hw, action_mode, delta_gain,
                                    image_compression=image_compression)
        os.replace(temporary, out_path)
    result["path"] = str(out_path)
    return result


# ------------------------------------------------------------------------------------------
# LIBERO HDF5
# ------------------------------------------------------------------------------------------
def _write_libero_hdf5(canonical_path: str | Path, out_path: str | Path, image_hw=None, action_mode: str = "auto",
                      delta_gain: float = 4.0, overwrite: bool = False, image_compression: str | None = None) -> dict:
    """Write the official LIBERO layout (``data/demo_i/...`` + ``env_args``/``problem_info`` attrs).

    * images are stored in LIBERO's OpenGL row order (flipped w.r.t. the canonical top-down frames);
    * ``action_mode="native"`` keeps the backend actions (LIBERO backend: 7-D OSC deltas, ManiSkill:
      6-D absolute ``pd_ee_pose`` targets); ``"libero_delta"`` converts absolute position targets
      into approximate LIBERO-style deltas with ``derive_libero_delta_actions``; these do not
      make simulator dynamics interchangeable. ``"auto"`` = native for LIBERO-style actions, delta otherwise;
    * missing joint/gripper/wrist data (ManiSkill ``panda_stick`` has no gripper) is zero-filled and
      flagged in the ``painter_missing`` attribute.
    """
    if action_mode not in LIBERO_ACTION_MODES:
        raise ValueError(f"action_mode must be one of {LIBERO_ACTION_MODES}")
    out_path = Path(out_path)
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(out_path)
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    info = C.read_info(canonical_path)
    bc = info.backend_config
    if action_mode == "auto":
        action_mode = "native" if info.action_space.get("type") == "libero_osc_pose_delta" else "libero_delta"
    total = 0
    missing = set()
    with h5py.File(canonical_path, "r") as src, h5py.File(out_path, "w") as out:
        data = out.create_group("data")
        names = C.episode_names(src)
        for i, name in enumerate(names):
            ep = C.load_episode(src["episodes"][name])
            T = ep.num_steps
            demo = data.create_group(f"demo_{i}")
            obs = demo.create_group("obs")
            agent = ep.obs["agentview_rgb"]
            if image_hw is not None:
                agent = C.resize_images(agent, image_hw)
            eih = ep.obs.get("eye_in_hand_rgb")
            if eih is None:
                eih = np.zeros_like(agent); missing.add("eye_in_hand_rgb")
            elif image_hw is not None:
                eih = C.resize_images(eih, image_hw)
            # official LIBERO files store uncompressed uint8 frames; compression is optional
            ckw = dict(compression=image_compression) if image_compression else {}
            obs.create_dataset("agentview_rgb", data=agent[:, ::-1], **ckw)
            obs.create_dataset("eye_in_hand_rgb", data=eih[:, ::-1], **ckw)
            ee_pos = np.asarray(ep.obs["eef_pos"], dtype=np.float64)
            ee_quat_xyzw = C.quat_wxyz_to_xyzw(np.asarray(ep.obs["eef_quat_wxyz"], dtype=np.float64))
            ee_ori = C.quat_xyzw_to_axisangle(ee_quat_xyzw)
            joint = np.asarray(ep.obs["joint_pos"], dtype=np.float64) if "joint_pos" in ep.obs else np.zeros((T, 7))
            if "joint_pos" not in ep.obs:
                missing.add("joint_states")
            grip = np.asarray(ep.obs["gripper_qpos"], dtype=np.float64) if "gripper_qpos" in ep.obs else np.zeros((T, 2))
            if "gripper_qpos" not in ep.obs:
                missing.add("gripper_states")
            obs.create_dataset("ee_pos", data=ee_pos); obs.create_dataset("ee_ori", data=ee_ori)
            obs.create_dataset("ee_states", data=np.concatenate([ee_pos, ee_ori], 1))
            obs.create_dataset("gripper_states", data=grip); obs.create_dataset("joint_states", data=joint)
            for k in ("pen_tip_painter", "pen_tip_world", "ink_count"):
                if k in ep.obs:
                    obs.create_dataset(k, data=ep.obs[k])
            if "painter_compare_rgb" in ep.obs:   # extra comparison render (e.g. 320x240), same OpenGL row order as the other images
                obs.create_dataset("painter_compare_rgb", data=ep.obs["painter_compare_rgb"][:, ::-1], **ckw)
            if action_mode == "libero_delta" and info.action_space.get("type") != "libero_osc_pose_delta":
                targets = ep.arrays.get("targets_world")
                if targets is None:
                    raise ValueError("libero_delta actions need arrays/targets_world in the canonical episode")
                actions = C.derive_libero_delta_actions(targets[:T], ee_pos, gain_pos=delta_gain)
            else:
                actions = np.asarray(ep.actions, dtype=np.float64)
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("robot_states", data=np.concatenate([grip, ee_pos, ee_quat_xyzw], 1))
            rewards = np.zeros(T, np.uint8); dones = np.zeros(T, np.uint8); rewards[-1] = 1; dones[-1] = 1
            demo.create_dataset("rewards", data=rewards); demo.create_dataset("dones", data=dones)
            state = ep.arrays.get("backend_state")
            if state is None:
                state = np.zeros((T, 0)); missing.add("states")
            state = np.asarray(state, dtype=np.float64)
            demo.create_dataset("states", data=state)
            for k in ("waypoints_world", "waypoints_painter", "ink_dots_xyz"):
                if k in ep.arrays:
                    demo.create_dataset({"waypoints_world": "ee_waypoints_world", "waypoints_painter": "ee_waypoints_painter"}.get(k, k), data=ep.arrays[k])
            demo.attrs["num_samples"] = T
            demo.attrs["init_state"] = state[0] if len(state) and state.shape[1] else np.zeros(0)
            demo.attrs["model_file"] = ep.meta.get("model_file", "")
            for k, v in ep.meta.items():
                if k in ("model_file", "num_steps"):
                    continue
                demo.attrs[k] = v
            total += T
        # dataset-level attrs (same keys as LIBERO's create_dataset.py + painter extras)
        env_args = bc.get("env_args") or {"type": 1, "env_name": bc.get("env_name", info.backend), "problem_name": bc.get("problem_name", info.backend),
                                           "bddl_file": bc.get("bddl_file", ""), "env_kwargs": bc.get("env_kwargs", {})}
        problem_info = bc.get("problem_info") or {"problem_name": env_args["problem_name"], "domain_name": "robosuite" if info.backend == "libero" else "maniskill",
                                                  "language_instruction": info.language}
        data.attrs["env_name"] = env_args["env_name"]
        data.attrs["problem_info"] = json.dumps(problem_info)
        data.attrs["macros_image_convention"] = "opengl"
        data.attrs["bddl_file_name"] = env_args.get("bddl_file", "")
        data.attrs["bddl_file_content"] = bc.get("bddl_file_content", "")
        data.attrs["env_args"] = json.dumps(env_args)
        data.attrs["num_demos"] = len(names)
        data.attrs["total"] = total
        data.attrs["tag"] = "libero-v1"
        data.attrs["date"] = _dt.datetime.now().strftime("%m-%d-%Y")
        data.attrs["time"] = _dt.datetime.now().strftime("%H:%M:%S")
        data.attrs["repository_version"] = bc.get("robosuite_version", "n/a")
        data.attrs["painter_generation"] = json.dumps(dict(info.generation, backend=info.backend, robot=info.robot, action_space=info.action_space,
                                                           action_mode=action_mode, delta_gain=delta_gain, image_hw=list(image_hw) if image_hw else None,
                                                           canonical=str(canonical_path)))
        data.attrs["painter_missing"] = json.dumps(sorted(missing))
        data.attrs["painter_export_convention"] = "legacy-libero-training-v1"
        data.attrs["painter_control_freq"] = info.control_freq
        data.attrs["painter_simulator_replayable"] = False
        data.attrs["painter_replay_note"] = "Simulator replay is not validated or supported by this export"
        data.attrs["painter_reward_semantics"] = "synthetic terminal marker, not measured task success"
        data.attrs["painter_action_mode"] = action_mode
        data.attrs["painter_action_semantics"] = (
            "approximate clipped OSC-style position delta; zero rotation; fixed closed gripper"
            if action_mode == "libero_delta" and info.action_space.get("type") != "libero_osc_pose_delta"
            else "native backend actions; see painter_generation.action_space"
        )
    return dict(format="libero", path=str(out_path), num_demos=len(names), total_steps=total, missing=sorted(missing), write_time_s=time.time() - t0)


# ------------------------------------------------------------------------------------------
# LeRobot v2.1
# ------------------------------------------------------------------------------------------
MANISKILL_ROBOT_BASE = np.array([-0.615, 0.0, 0.0])   # DrawSVGParallel-v1 robot base in world (state frame)
PAINTER_TARGET_EULER = np.array([np.pi, 0.0, 0.0])     # pen pointing down, as in the ManiSkill pd_ee_pose targets
PAINTER_EXPORT_CONVENTION = "painter-absolute-target-v1"
STATE_NAMES = ["pen_tip_x", "pen_tip_y", "pen_tip_z", "eef_qw", "eef_qx", "eef_qy", "eef_qz", "previous_end"]
ACTION_NAMES = ["target_x", "target_y", "target_z", "target_rx", "target_ry", "target_rz", "pen_gripper", "end"]


def painter_data_contract(end_signal_length: int) -> dict:
    return dict(
        convention=PAINTER_EXPORT_CONVENTION, container="lerobot-v2.1", length_unit="metres", angle_unit="radians",
        state=dict(names=STATE_NAMES, dimension=8, position_frame="maniskill_world",
                   robot_base_world=MANISKILL_ROBOT_BASE.tolist(), position_reference="pen_tip",
                   orientation="recorded end-effector quaternion wxyz, unchanged"),
        action=dict(names=ACTION_NAMES, dimension=8, position_frame="robot_base", position_reference="pen_tip_target",
                    mode="absolute", euler_xyz=PAINTER_TARGET_EULER.tolist(), pen_gripper=1.0),
        transform="translation only: backend robot base = pen_tip_world - pen_tip_painter",
        end_signal=dict(final_steps=end_signal_length, initial_previous_end=0,
                        rule="end[t] = 1 when t >= episode_length - final_steps; previous_end[t] = end[t-1]"),
        image_row_order="top_down", simulator_replay_supported=False,
    )


def painter_state_action(g, T: int):
    """Backend-independent (state[:7], action[:7]) of one canonical episode in the ManiSkill / painter convention.

    state  = [pen tip xyz in ManiSkill world, eef quat wxyz]
    action = [pen-tip target xyz in the robot base frame, pi, 0, 0, 1]
    """
    for k in ("obs/pen_tip_painter", "obs/pen_tip_world", "arrays/targets_world"):
        if k not in g:
            raise KeyError(f"{g.name}: canonical episode lacks {k}; cannot export in the painter convention")
    tip_painter = g["obs/pen_tip_painter"][:T].astype(np.float64)
    if T <= 0 or tip_painter.shape != (T, 3):
        raise ValueError(f"{g.name}: pen_tip_painter must have nonempty shape (T, 3)")
    base = g["obs/pen_tip_world"][0].astype(np.float64) - tip_painter[0]   # backend robot base in its world frame
    targets = g["arrays/targets_world"][:].astype(np.float64)
    if len(targets) != T:
        raise ValueError(f"{g.name}: targets_world has {len(targets)} rows, episode has {T} steps")
    state = np.concatenate([tip_painter + MANISKILL_ROBOT_BASE, g["obs/eef_quat_wxyz"][:T]], 1)
    action = np.concatenate([targets - base, np.repeat(PAINTER_TARGET_EULER[None], T, 0), np.ones((T, 1))], 1)
    return state, action

def write_lerobot(canonical_path: str | Path, out_root: str | Path, repo_name: str, end_signal_length: int = 5, fps: int | None = None,
                  overwrite: bool = False, image_writer_processes: int = 0, image_writer_threads: int = 4) -> dict:
    """Validate first, export to staging, then publish the complete dataset directory."""
    C.validate_file(canonical_path, require_painter=True)
    validate_dataset_name(repo_name)
    if isinstance(end_signal_length, bool) or not isinstance(end_signal_length, int) or end_signal_length < 1:
        raise ValueError("end_signal_length must be a positive integer")
    info = C.read_info(canonical_path)
    if fps is not None and fps != info.control_freq:
        raise ValueError("fps must match canonical control_freq; resampling is not implemented")
    out_root = Path(out_root).resolve()
    output = out_root / repo_name
    if output == Path(canonical_path).resolve() or output in Path(canonical_path).resolve().parents:
        raise ValueError("export output must not contain the canonical source")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if output.is_symlink():
        raise ValueError("export output must not be a symbolic link")
    out_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lerobot-export-", dir=out_root) as staging:
        result = _write_lerobot(canonical_path, staging, repo_name, end_signal_length, fps,
                               image_writer_processes=image_writer_processes, image_writer_threads=image_writer_threads)
        result.update(path=str(output), data_contract=painter_data_contract(end_signal_length),
                      control_freq=info.control_freq, source_backend=info.backend, source_canonical=str(Path(canonical_path).resolve()))
        staged_output = Path(staging) / repo_name
        (staged_output / "painter_export.json").write_text(json.dumps(result, indent=2) + "\n")
        if output.exists():
            shutil.rmtree(output)
        os.replace(staged_output, output)
    return result


def _write_lerobot(canonical_path: str | Path, out_root: str | Path, repo_name: str, end_signal_length: int = 5, fps: int | None = None,
                   overwrite: bool = False, image_writer_processes: int = 0, image_writer_threads: int = 4) -> dict:
    """Write a LeRobot v2.1 dataset in the ManiSkill / painter convention (``lerobot_single``,
    ``scripts/convert_maniskill_to_lerobot.py``) for every backend, so datasets can be mixed:

    * ``observation.image`` (video, agentview), ``observation.wrist_image`` (video, if a wrist camera exists),
    * ``observation.state`` = pen tip xyz in the ManiSkill world (robot base at ``MANISKILL_ROBOT_BASE``) (3)
      + eef quaternion wxyz (4) + previous end-signal (1),
    * ``action`` = absolute pen-tip target xyz in the robot base frame (3) + euler ``(pi, 0, 0)`` (3) + 1 (pen/gripper)
      + end-signal (1 for the last ``end_signal_length`` steps).
    Positions come from ``obs/pen_tip_painter`` and ``arrays/targets_world`` (the robot base is recovered as
    ``pen_tip_world - pen_tip_painter``), so the backend-native action (e.g. LIBERO OSC deltas) is not used.
    Requires the ``lerobot`` package (see ``scripts/setup_export.sh``)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: WPS433 (lazy: only in venvs that have lerobot)

    t0 = time.time()
    info = C.read_info(canonical_path)
    out_root = Path(out_root)
    output_path = out_root / repo_name
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(output_path)
        shutil.rmtree(output_path)
    with h5py.File(canonical_path, "r") as src:
        names = C.episode_names(src)
        first = src["episodes"][names[0]]
        image_shape = tuple(first["obs/agentview_rgb"].shape[1:])
        has_wrist = "eye_in_hand_rgb" in first["obs"]
        wrist_shape = tuple(first["obs/eye_in_hand_rgb"].shape[1:]) if has_wrist else None
        state_dim = 3 + 4 + 1
        action_dim = 3 + 3 + 1 + 1
    features = {
        "observation.image": {"dtype": "video", "shape": tuple(image_shape), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": ACTION_NAMES},
    }
    if has_wrist:
        features["observation.wrist_image"] = {"dtype": "video", "shape": tuple(wrist_shape), "names": ["height", "width", "channel"]}
    dataset = LeRobotDataset.create(repo_id=repo_name, robot_type="panda", fps=int(fps or info.control_freq), root=output_path, features=features,
                                    image_writer_threads=image_writer_threads, image_writer_processes=image_writer_processes, use_videos=True)
    try:
        total = 0
        with h5py.File(canonical_path, "r") as src:
            for name in names:
                g = src["episodes"][name]
                T = int(g.attrs["num_steps"])
                imgs = g["obs/agentview_rgb"][:]
                wrist = g["obs/eye_in_hand_rgb"][:] if has_wrist else None
                state, action = painter_state_action(g, T)
                last_end = np.array([0.0], dtype=np.float32)
                for t in range(T):
                    end_signal = np.array([1.0 if t >= T - end_signal_length else 0.0], dtype=np.float32)
                    frame = {
                        "observation.image": imgs[t],
                        "observation.state": np.concatenate([state[t], last_end]).astype(np.float32),
                        "action": np.concatenate([action[t], end_signal]).astype(np.float32),
                    }
                    if has_wrist:
                        frame["observation.wrist_image"] = wrist[t]
                    dataset.add_frame(frame, task=info.language)
                    last_end = end_signal
                dataset.save_episode()
                total += T
    finally:
        dataset.stop_image_writer()
    return dict(format="lerobot", path=str(output_path), num_episodes=len(names), total_steps=total, write_time_s=time.time() - t0)
