"""Record active ManiSkill trajectories directly into the canonical dataset.

Each row contains the observation before its paired native action. Completed
SVGs are flushed before any reset; idle vector slots never start an episode.
The collector does not depend on RecordEpisode's private buffering internals.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from burger.data import canonical as C


def as_numpy(value):
    """Copy simulator tensors before the next step/reset mutates their storage."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def record_episodes(
    env, writer, svg_files, svg_strings, waypoints, xy_offsets, *, seed=0, log=print
):
    """Write exactly one episode per input SVG, despite uneven vector lengths.

    ``env`` uses the DrawSVGParallelEnv queue API. Keeping the scheduling separate
    from simulator construction lets CPU tests exercise reset and timing behavior.
    """
    core = env.unwrapped
    n_envs = core.num_envs
    buffers = {i: defaultdict(list) for i in range(n_envs)}
    error_buffers = {i: [] for i in range(n_envs)}
    start_state = {}
    obs, _ = env.reset(seed=seed, options={"assign_svg": True})
    written = set()
    while len(written) < len(svg_files):
        active = np.flatnonzero(~as_numpy(core.env_completed))
        ids = as_numpy(core.env_svg_indices).astype(int)
        if len(active) == 0:
            raise RuntimeError(
                "ManiSkill queue became idle before all SVGs were recorded"
            )
        tcp = as_numpy(core.agent.tcp.pose.raw_pose)
        root_pose = as_numpy(core.agent.robot.get_root_pose().raw_pose)
        translation_only = np.isclose(
            np.abs(root_pose[:, 3]), 1.0, atol=1e-6
        ) & np.isclose(root_pose[:, 4:], 0.0, atol=1e-6).all(axis=1)
        if not translation_only.all():
            invalid = np.flatnonzero(~translation_only)
            details = [
                {
                    "env_index": int(i),
                    "svg_index": int(ids[i]),
                    "recorded_rows": len(buffers[i]["actions"]),
                    "root_pose_xyz_wxyz": root_pose[i].tolist(),
                    "tcp_pose_xyz_wxyz": tcp[i].tolist(),
                }
                for i in invalid
            ]
            raise ValueError(
                "Painter export requires translation-only robot bases; "
                f"invalid simulator poses: {details}"
            )
        qpos = as_numpy(core.agent.robot.get_qpos())
        ink = as_numpy(core.env_draw_steps)
        rgb = as_numpy(obs["sensor_data"]["base_camera"]["rgb"])[..., :3]
        wrist = obs["sensor_data"].get("wrist_camera")
        wrist_rgb = as_numpy(wrist["rgb"])[..., :3] if wrist is not None else None
        action = core.get_hardcoded_action()
        action_np = as_numpy(action)
        # get_hardcoded_action marks the final hold before the next physics step.
        # Reject the historical world-as-root command before it moves the robot.
        holding = active[as_numpy(core.env_need_reset)[active].astype(bool)]
        for i in holding:
            expected_hold = tcp[i, :3] - root_pose[i, :3]
            if not np.allclose(action_np[i, :3], expected_hold, atol=1e-5, rtol=0):
                raise ValueError(
                    "Terminal hold action must use robot-base coordinates: "
                    f"env_index={i}, svg_index={ids[i]}, "
                    f"action_xyz={action_np[i, :3].tolist()}, "
                    f"expected_root_xyz={expected_hold.tolist()}"
                )
        for i in active:
            if ids[i] < 0 or ids[i] >= len(svg_files):
                raise RuntimeError(
                    f"Invalid SVG identity in active environment {i}: {ids[i]}"
                )
            row = buffers[i]
            if not row:
                start_state[i] = dict(
                    initial_tcp_world=tcp[i, :3].tolist(),
                    initial_joint_pos=qpos[i, :7].tolist(),
                )
            row["agentview_rgb"].append(rgb[i])
            if wrist_rgb is not None:
                row["eye_in_hand_rgb"].append(wrist_rgb[i])
            row["eef_pos"].append(tcp[i, :3])
            row["eef_quat_wxyz"].append(tcp[i, 3:7])
            row["pen_tip_world"].append(tcp[i, :3])
            row["pen_tip_painter"].append(tcp[i, :3] - root_pose[i, :3])
            row["joint_pos"].append(qpos[i, :7])
            row["ink_count"].append(ink[i])
            row["actions"].append(action_np[i])
            row["targets_world"].append(action_np[i, :3] + root_pose[i, :3])
        next_obs, _, _, truncated, _ = env.step(action)
        if np.any(as_numpy(truncated)[active]):
            raise RuntimeError("ManiSkill truncated an unfinished SVG trajectory")
        tcp_after = as_numpy(core.agent.tcp.pose.p)
        for i in active:
            error_buffers[i].append(
                np.linalg.norm(tcp_after[i] - buffers[i]["targets_world"][-1])
            )
        finished = np.flatnonzero(
            as_numpy(core.env_need_reset) & ~as_numpy(core.env_completed)
        )
        for i in finished:
            svg_idx = int(ids[i])
            if svg_idx in written:
                raise RuntimeError(f"Duplicate completed SVG identity: {svg_idx}")
            data = {key: np.asarray(value) for key, value in buffers[i].items()}
            actions = data.pop("actions")
            targets = data.pop("targets_world")
            expected_steps = (
                len(waypoints[svg_idx]) + 1
            )  # Original collector's final hold-current-TCP row.
            if len(actions) != expected_steps:
                raise RuntimeError(
                    f"SVG {svg_idx}: expected {expected_steps} rows, got {len(actions)}"
                )
            base = data["pen_tip_world"][0] - data["pen_tip_painter"][0]
            errors = np.asarray(error_buffers[i])
            meta = dict(
                svg_file=str(svg_files[svg_idx]),
                svg_index=svg_idx,
                svg_string=svg_strings[svg_idx],
                env_index=int(i),
                seed=int(seed) + svg_idx,
                xy_offset=np.asarray(xy_offsets[svg_idx]).tolist(),
                trajectory_complete=True,
                truncated=False,
                visual_success_evaluated=False,
                completion_criterion="all planned waypoints and one hold action executed",
                observation_timing="before_action",
                final_ink_count=int(as_numpy(core.env_draw_steps)[i]),
                ink_count=int(as_numpy(core.env_draw_steps)[i]),
                recorded_hold_steps=1,
                tip_error_after_step_mean_m=float(errors.mean()),
                tip_error_after_step_p95_m=float(np.percentile(errors, 95)),
                **start_state[i],
            )
            apps = getattr(core, "appearance_list", None)
            if apps is not None:
                meta["appearance"] = apps[i].to_dict()
                meta["appearance_scope"] = "fixed_per_environment"
            writer.add_episode(
                C.Episode(
                    obs=data,
                    actions=actions.astype(np.float64),
                    meta=meta,
                    arrays=dict(
                        targets_world=targets.astype(np.float64),
                        waypoints_painter=waypoints[svg_idx],
                        waypoints_world=waypoints[svg_idx] + base,
                    ),
                ),
                index=svg_idx,
            )
            written.add(svg_idx)
            buffers[i].clear()
            error_buffers[i].clear()
            log(
                f"[maniskill] SVG {svg_idx + 1}/{len(svg_files)}: {len(actions)} rows, {meta['final_ink_count']} ink dots"
            )
        remaining = len(svg_files) - core.next_svg_index
        to_reset = finished[:remaining]
        idle = finished[remaining:]
        if len(idle):
            core.env_completed[idle.tolist()] = True
            core.env_need_reset[idle.tolist()] = False
        if len(to_reset):
            # The next observation covers the whole vector, including unchanged active slots.
            obs, _ = env.reset(options={"env_idx": to_reset, "assign_svg": True})
        else:
            obs = next_obs
    if written != set(range(len(svg_files))):
        raise RuntimeError("ManiSkill output SVG identities do not match its inputs")
    return len(written)
