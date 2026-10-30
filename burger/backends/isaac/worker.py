#!/usr/bin/env python
"""Isaac Sim worker: draws SVG trajectories with the ``IsaacDrawEnv`` and writes canonical episodes.

Runs inside the Isaac Sim virtualenv (Kit owns the process).

  # Driven by burger.backends.isaac.backend.
  .venv-isaac61/bin/python -m burger.backends.isaac.worker --job_json work/isaac_job.json --summary_json work/summary.json --device 0

Use ``python -m burger generate --backend isaac ...`` for generation and export.

Episode structure mirrors ManiSkill's collector: one action per waypoint (absolute ``pd_ee_pose`` target in the
robot root frame, rot = [pi, 0, 0]), then ``hold_steps`` hold steps (target = current TCP) and ``raise_steps``
``raise_arms`` steps (target z = 0.35; the reference data has 1 hold and 0 raise steps), then the env is reset for its next SVG. Observation i is the state *before* action i.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job_json", required=True, help="job description written by the isaac backend")
    ap.add_argument("--summary_json", default=None)
    ap.add_argument("--device", type=int, default=0, help="GPU index for rendering + physics")
    # debugging
    ap.add_argument("--max_steps", type=int, default=None, help="reject SVGs exceeding this waypoint count (debug safety limit; never truncate)")
    ap.add_argument("--debug_dir", default=None, help="save a few rendered frames here (debug)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t_proc = time.time()
    from burger.backends._worker_utils import validate_gpu_selection
    args.device = validate_gpu_selection(args.device, backend="Isaac")
    from isaaclab.app import AppLauncher

    options = json.loads(Path(args.job_json).read_text()).get("options", {})
    kit_args = (str(options.get("extra_kit_args", "")) +
                f" --/renderer/activeGpu={args.device} --/renderer/multiGpu/enabled=false")
    app_launcher = AppLauncher(headless=True, device=f"cuda:{args.device}", enable_cameras=True, kit_args=kit_args)
    app = app_launcher.app
    app_startup_s = time.time() - t_proc
    print(f"[isaac worker] Kit app started in {app_startup_s:.1f}s (device cuda:{args.device})", flush=True)
    rc = 1
    try:
        rc = run(args, app_startup_s)
    except BaseException:
        # Kit can exit directly during close(). Persist and print failure before closing.
        traceback.print_exc()
        if args.summary_json:
            Path(args.summary_json).write_text(json.dumps({"status": "failed", "error": traceback.format_exc()}, indent=2))
    finally:
        app.close(exit_code=rc)
    return rc


def run(args, app_startup_s: float) -> int:
    import numpy as np
    import torch

    from burger.pipeline import svg_trajectory as traj
    from burger.data import canonical as C
    from burger.backends.base import GenerationConfig
    from burger.backends.isaac import geometry as G
    from burger.backends.isaac.environment import IsaacDrawEnv, DEFAULT_ENV_OPTIONS

    # ------------------------------------------------------------------ job
    job = json.loads(Path(args.job_json).read_text())
    cfg = GenerationConfig(**job["cfg"])
    opt = dict(job["options"])
    part_path = Path(job["part"])
    svg_files = [Path(f) for f in job["svg_files"]]
    part_path.parent.mkdir(parents=True, exist_ok=True)
    if not svg_files:
        raise ValueError("Isaac generation requires at least one SVG")
    if int(opt.get("tcp_init_max_trials", 10)) < 1:
        raise ValueError("tcp_init_max_trials must be positive")
    env_opt = {k: opt[k] for k in DEFAULT_ENV_OPTIONS if k in opt}
    # appearance: preset / randomization -> one Appearance per parallel env (fixed for the run; draws differ per env)
    appearances = None
    if opt.get("appearance") is not None or str(opt.get("appearance_random", "none")) != "none":
        from burger.pipeline import appearance as A
        mode = str(opt.get("appearance_random", "none"))
        n_env = max(1, min(int(cfg.parallel), len(svg_files))) if len(svg_files) else int(cfg.parallel)
        appearances = [A.resolve_appearance(dict(appearance=opt.get("appearance", "maniskill"), appearance_random=mode, appearance_seed=int(opt.get("appearance_seed", 0))),
                                            episode_seed=i if mode != "none" else None) for i in range(n_env)]
        env_opt["appearance_list"] = [a.to_dict() for a in appearances]
        env_opt["appearance_seed"] = int(opt.get("appearance_seed", 0))
        for i, a in enumerate(appearances):
            print(f"[isaac worker] env {i} appearance: {A.describe(a)}", flush=True)
        if mode != "none":
            print("[isaac worker] appearance randomization is fixed per vector slot; scene lighting uses slot 0", flush=True)
    if opt.get("camera_preset") not in (None, "", "none", "maniskill"):
        env_opt["camera_preset"] = opt["camera_preset"]
        print(f"[isaac worker] camera preset: {opt['camera_preset']}", flush=True)
    if opt.get("canvas_size") is not None:
        env_opt["canvas_size"] = [float(v) for v in opt["canvas_size"]]
        print(f"[isaac worker] canvas size: {env_opt['canvas_size']}", flush=True)
    compare_wh = None
    if opt.get("compare_render"):
        w, h = str(opt["compare_render"]).lower().split("x")
        compare_wh = (int(w), int(h))
    hold_steps, raise_steps, raise_height = int(opt["hold_steps"]), int(opt["raise_steps"]), float(opt["raise_height"])
    # [isaac v6] real-robot raise tail / random TCP init / approach guard (all off by default; see isaac_backend.DEFAULT_OPTIONS)
    raise_ramp, raise_xy = int(opt.get("raise_ramp", 0)), str(opt.get("raise_xy", "base"))
    tcp_init = int(opt.get("tcp_init_random", 0)) > 0
    guard_xy, guard_z = float(opt.get("approach_guard_xy", 0.0)), float(opt.get("approach_guard_z", 0.04))
    store_states = bool(opt.get("store_states", True))
    num_envs = max(1, min(int(cfg.parallel), len(svg_files))) if len(svg_files) else int(cfg.parallel)
    device = f"cuda:{args.device}"
    log = lambda m: print(m, flush=True)

    # ------------------------------------------------------------------ env
    t0 = time.time()
    env = IsaacDrawEnv(num_envs, camera_width=cfg.camera_width, camera_height=cfg.camera_height, wrist_camera=cfg.wrist_camera, compare_wh=compare_wh,
                       options=env_opt, device=device, log=log)
    scene_build_s = env.scene_build_s
    N = env.num_envs
    from importlib.metadata import version
    revision_file = Path(sys.prefix) / "isaaclab-commit.txt"

    info = C.DatasetInfo(
        backend="isaac", language=cfg.language, control_freq=int(round(1.0 / env.control_dt)), robot="panda_stick",
        cameras=["agentview_rgb"] + (["eye_in_hand_rgb"] if cfg.wrist_camera else []),
        action_space=dict(type="maniskill_pd_ee_pose_abs", dim=6, description="absolute pd_ee_pose target in the robot root frame (base at world [-0.615, 0, 0]): xyz (m) + euler xyz (rad); same as the ManiSkill backend",
                          robot_base_world=list(G.ROBOT_BASE)),
        backend_config=dict(env_name="IsaacDrawSVG", problem_name="isaac_draw_svg", isaac_sim=version("isaacsim"),
                            isaac_lab_commit=revision_file.read_text().strip() if revision_file.exists() else "unknown",
                            isaac_lab=version("isaaclab"), torch=version("torch"), scene=env.spec.to_dict(),
                            options=opt, env_options={k: v for k, v in env.opt.items() if k != "appearance_list"}, appearances=[a.to_dict() for a in appearances] if appearances else None,
                            appearance_scope="per_vector_slot", global_lighting_appearance_index=0 if appearances else None,
                            num_envs=N, device=args.device, robot_urdf=str(Path(env.opt.get("robot_urdf", "")) if env.opt.get("robot_urdf") else ""),
                            hold_steps=hold_steps, raise_steps=raise_steps, raise_height=raise_height),
        generation=cfg.to_json(),
    )

    # ------------------------------------------------------------------ per-env episode state
    class EpState:
        def __init__(self):
            self.job = None; self.wps = None; self.idx = 0; self.phase = "idle"; self.count = 0
            self.frames = []; self.wrist = []; self.compare = []; self.tcp = []; self.joints = []; self.jvel = []; self.actions = []; self.ink = []; self.targets = []
            self.tip_after_actions = []
            self.xy_offset = np.zeros(2); self.t_start = 0.0; self.svg_string = ""; self.hold_target = None
            self.init_q = None; self.init_meta = {}; self.guard_on = guard_xy > 0; self.guarded = 0   # [isaac v6]

    states = [EpState() for _ in range(N)]
    jobs = list(enumerate(svg_files))
    next_job = 0
    done_episodes = 0
    steps_total = 0
    max_steps = args.max_steps
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
    t_sim = time.time()
    timers = dict(render=0.0, control=0.0, ink=0.0, record=0.0)

    def assign(i: int):
        nonlocal next_job
        st = states[i]
        st.__init__()
        if next_job >= len(jobs):
            st.phase = "idle"; return
        svg_index, svg_file = jobs[next_job]; next_job += 1
        svg_string = traj.read_svg_path_string(svg_file)
        rng = np.random.default_rng([int(cfg.seed), int(svg_index), 9011])
        xy_offset = rng.uniform(-cfg.svg_xy_noise, cfg.svg_xy_noise, size=2) if cfg.svg_xy_noise > 0 else np.zeros(2)
        wps = traj.build_ee_waypoints(svg_string, cfg.max_draw_dist, cfg.end_lift_z, rot_image=cfg.rot_image, xy_offset=tuple(xy_offset), scale_mode=cfg.svg_scale_mode)
        if max_steps is not None and len(wps) > max_steps:
            raise ValueError(f"SVG {svg_file} needs {len(wps)} waypoints, exceeding --max_steps={max_steps}; refusing truncation")
        st.job = (svg_index, str(svg_file)); st.wps = wps.astype(np.float64); st.idx = 0; st.phase = "draw"; st.xy_offset = xy_offset; st.svg_string = svg_string
        st.t_start = time.time()
        if tcp_init:   # [isaac v6] seeded per episode (seed, svg_index) exactly like the LIBERO backend's _random_tcp_init
            g = np.random.default_rng([int(cfg.seed), int(svg_index), 7477])
            st.init_meta = dict(tcp_init_target_painter=None, tcp_init_trials=int(opt.get("tcp_init_max_trials", 10)), tcp_init_err_m=float("inf"))
            for trial in range(int(opt.get("tcp_init_max_trials", 10))):
                goal_p = np.array([g.uniform(float(opt["tcp_init_x_min"]), float(opt["tcp_init_x_max"])) - G.ROBOT_BASE[0],   # ManiSkill world -> root
                                   g.uniform(float(opt["tcp_init_y_min"]), float(opt["tcp_init_y_max"])), float(opt["tcp_init_z"])])
                q_init, err = env.tcp_init_qpos(goal_p)
                st.init_meta["tcp_init_err_m"] = err
                if err <= 1e-3:
                    st.init_q = q_init
                    st.init_meta.update(tcp_init_target_painter=goal_p.tolist(), tcp_init_trials=trial + 1)
                    break
            if st.init_q is None:
                raise RuntimeError(f"Could not initialize SVG {svg_index} TCP within 1 mm after {opt.get('tcp_init_max_trials', 10)} trials")

    def reset_qpos(ids):
        """[isaac v6] initial joints of the envs ``ids`` (their TCP-init solution, rest keyframe otherwise); None = all rest."""
        if not tcp_init:
            return None
        return np.stack([states[i].init_q if states[i].init_q is not None else np.asarray(G.REST_QPOS) for i in ids])

    def finish(i: int, writer):
        nonlocal done_episodes, steps_total
        st = states[i]
        T = len(st.actions)
        acts = np.asarray(st.actions, dtype=np.float64)
        tcp = np.asarray(st.tcp, dtype=np.float64)                      # (T, 7) pos + quat wxyz, ManiSkill world frame
        n_wp = len(st.wps)
        targets_world = np.asarray(st.targets, dtype=np.float64) + np.asarray(G.ROBOT_BASE)
        err = np.linalg.norm(np.asarray(st.tip_after_actions) - targets_world, axis=1)
        down = st.wps[:, 2] < G.TOUCH_Z_THRESHOLD
        obs = dict(agentview_rgb=np.stack(st.frames), eef_pos=tcp[:, :3], eef_quat_wxyz=tcp[:, 3:7], joint_pos=np.asarray(st.joints, dtype=np.float64),
                   pen_tip_world=tcp[:, :3], pen_tip_painter=tcp[:, :3] - np.asarray(G.ROBOT_BASE), ink_count=np.asarray(st.ink, dtype=np.int32))
        if st.wrist:
            obs["eye_in_hand_rgb"] = np.stack(st.wrist)
        if st.compare:
            obs["painter_compare_rgb"] = np.stack(st.compare)
        arrays = dict(targets_world=targets_world, commanded_targets_world=acts[:, :3] + np.asarray(G.ROBOT_BASE),
                      waypoints_world=st.wps + np.asarray(G.ROBOT_BASE), waypoints_painter=st.wps, ink_dots_xyz=env.ink_positions_env(i))
        if store_states:
            arrays["backend_state"] = np.concatenate([np.asarray(st.joints), np.asarray(st.jvel)], axis=1).astype(np.float64)
        gen_time = time.time() - st.t_start
        meta = dict(svg_file=st.job[1], svg_index=int(st.job[0]), svg_string=st.svg_string, xy_offset=st.xy_offset.tolist(), ink_count=int(env.draw_count[i]),
                    appearance=appearances[i].to_json() if appearances else "",
                    scene_lighting_appearance=appearances[0].to_json() if appearances else "",
                    pen_down_waypoints=int(down.sum()), num_waypoints=int(n_wp), recorded_hold_and_lift_steps=hold_steps + raise_steps,
                    tip_error_after_step_mean_m=float(err.mean()), tip_error_after_step_p95_m=float(np.percentile(err, 95)),
                    tip_error_pen_down_after_step_mean_m=float(err[:n_wp][down].mean()) if down.any() else 0.0,
                    gen_time_s=gen_time, trajectory_complete=True, truncated=False, visual_success_evaluated=False, env_id=i,
                    episode_seed=int(cfg.seed) + int(st.job[0]),
                    hold_steps=hold_steps, raise_steps=raise_steps, raise_height=raise_height, raise_ramp=raise_ramp, raise_xy=raise_xy,   # [isaac v6]
                    approach_guarded_steps=int(st.guarded), joint_pos_min=np.asarray(st.joints).min(0).tolist(), joint_pos_max=np.asarray(st.joints).max(0).tolist(),
                    **st.init_meta)
        ep = C.Episode(obs=obs, actions=acts, meta=meta, arrays=arrays)
        writer.add_episode(ep, index=int(st.job[0]))
        done_episodes += 1; steps_total += T
        log(f"[isaac worker] env {i} svg {st.job[0]} {Path(st.job[1]).name}: {T} steps, ink {meta['ink_count']}/{meta['pen_down_waypoints']}, "
            f"tip err {meta['tip_error_after_step_mean_m']*1000:.2f} mm (pen down {meta['tip_error_pen_down_after_step_mean_m']*1000:.2f} mm), {gen_time:.1f}s")

    for i in range(N):
        assign(i)
    if tcp_init:   # [isaac v6] first episodes start at their TCP-init pose too
        env.reset_envs(list(range(N)), qpos=reset_qpos(range(N)))
    env.settle()   # renderer only sees kinematic resets after a physics step -> make the first frame fresh
    targets = torch.zeros((N, 3), device=device)
    rot_action = np.array(G.TCP_TARGET_EULER_XYZ, dtype=np.float64)
    with C.CanonicalWriter(part_path, info) as writer:
        step_k = 0
        pending_reset = []
        while any(st.phase != "idle" for st in states):
            t = time.perf_counter()
            frames = env.render()
            timers["render"] += time.perf_counter() - t
            t = time.perf_counter()
            tcp_env = env.tcp_pose_env()
            q, qd = env.joint_state()
            tcp_root = env.tcp_pos_root()
            tcp_root_np = tcp_root.detach().cpu().numpy()
            active = np.zeros(N, dtype=bool)
            tgt = np.zeros((N, 3), dtype=np.float64)
            cmd = {}   # [isaac v6] env -> commanded target overriding tgt (approach guard)
            finished = []
            for i, st in enumerate(states):
                if st.phase == "idle" or i in pending_reset:
                    tgt[i] = tcp_root[i].detach().cpu().numpy(); continue
                active[i] = True
                if st.phase == "draw":
                    tgt[i] = st.wps[st.idx]; st.idx += 1
                    if st.idx >= len(st.wps):
                        st.phase = "hold" if hold_steps > 0 else ("raise" if raise_steps > 0 else "done"); st.count = 0
                elif st.phase == "hold":
                    tgt[i] = tcp_root[i].detach().cpu().numpy(); st.count += 1
                    if st.count >= hold_steps:
                        st.phase = "raise" if raise_steps > 0 else "done"; st.count = 0
                elif st.phase == "raise":
                    st.count += 1
                    xy = (st.wps[-1, 0], st.wps[-1, 1]) if raise_xy == "last" else (0.0, 0.0)
                    if raise_ramp > 0:   # [isaac v6] real robot _record_raise: z linear from the last waypoint to raise_height
                        z0 = float(st.wps[-1, 2]); z1 = max(z0, raise_height)
                        tgt[i] = (xy[0], xy[1], z0 + (z1 - z0) * st.count / raise_steps)
                    else:
                        tgt[i] = (xy[0], xy[1], raise_height)
                    if st.count >= raise_steps:
                        st.phase = "done"
                # record obs (before the action) + action
                st.frames.append(frames["base"][i]); st.tcp.append(tcp_env[i]); st.joints.append(q[i]); st.jvel.append(qd[i]); st.ink.append(int(env.draw_count[i]))
                if "wrist" in frames:
                    st.wrist.append(frames["wrist"][i])
                if "compare" in frames:
                    st.compare.append(frames["compare"][i])
                st.targets.append(tgt[i].copy())
                if st.guard_on and tgt[i, 2] < guard_z:   # approach guard; planned target remains in st.targets
                    if np.linalg.norm(tcp_root_np[i, :2] - tgt[i, :2]) > guard_xy:
                        cmd[i] = (tgt[i, 0], tgt[i, 1], guard_z); st.guarded += 1
                    else:
                        st.guard_on = False
                st.actions.append(np.concatenate([np.asarray(cmd.get(i, tgt[i])), rot_action]))
                if st.phase == "done":
                    finished.append(i)
            timers["record"] += time.perf_counter() - t
            if debug_dir and step_k in (0, 1, 2, 5, 10, 50, 100, 200, 400, 800):
                from PIL import Image
                Image.fromarray(frames["base"][0]).save(debug_dir / f"base_{step_k:04d}.png")
                if "compare" in frames:
                    Image.fromarray(frames["compare"][0]).save(debug_dir / f"compare_{step_k:04d}.png")
                if "wrist" in frames:
                    Image.fromarray(frames["wrist"][0]).save(debug_dir / f"wrist_{step_k:04d}.png")
            if pending_reset:
                # A reset slot gets one unrecorded settling step while other slots continue
                # their recorded actions. Its first observation is captured next iteration.
                env.reset_envs(pending_reset, qpos=reset_qpos(pending_reset))
                tcp_root_new = env.tcp_pos_root().detach().cpu().numpy()
                for i in pending_reset:
                    tgt[i] = tcp_root_new[i]          # hold the rest pose during this control step
                    cmd.pop(i, None)
                    active[i] = False
            for i, c in cmd.items():
                tgt[i] = c
            targets[:] = torch.as_tensor(tgt, device=device, dtype=torch.float32)
            t = time.perf_counter()
            env.control_step(targets, hold_env_ids=np.flatnonzero(~active).tolist())
            timers["control"] += time.perf_counter() - t
            t = time.perf_counter()
            env.place_ink(active)
            after_action = env.tcp_pose_env()
            for i in np.flatnonzero(active):
                states[i].tip_after_actions.append(after_action[i, :3].copy())
            timers["ink"] += time.perf_counter() - t
            # The final recorded action must reach physics and ink before flushing/resetting.
            pending_reset = []
            for i in finished:
                finish(i, writer)
                assign(i)
                if states[i].phase != "idle":
                    pending_reset.append(i)
            step_k += 1
            if step_k % 200 == 0:
                log(f"[isaac worker] step {step_k}: {done_episodes}/{len(jobs)} episodes done, {step_k * N / (time.time() - t_sim):.1f} env-steps/s "
                    f"(render {timers['render']/step_k*1000:.1f} ms, control {timers['control']/step_k*1000:.1f} ms, ink {timers['ink']/step_k*1000:.1f} ms, record {timers['record']/step_k*1000:.1f} ms per step)")
    sim_wall = time.time() - t_sim
    summary = dict(status="complete", run_id=job.get("run_id"),
                   num_episodes=done_episodes, total_steps=steps_total, control_steps=step_k, num_envs=N, app_startup_s=app_startup_s, scene_build_s=scene_build_s,
                   sim_wall_s=sim_wall, backend_wall_s=scene_build_s + sim_wall, steps_per_s=steps_total / max(sim_wall, 1e-9),
                   ms_per_control_step={k: v / max(step_k, 1) * 1000 for k, v in timers.items()}, part=str(part_path), device=args.device)
    log(f"[isaac worker] done: {done_episodes} episodes / {steps_total} steps in {sim_wall:.1f}s sim ({summary['steps_per_s']:.1f} steps/s, {N} envs); "
        f"per control step: " + ", ".join(f"{k} {v:.1f} ms" for k, v in summary["ms_per_control_step"].items()))
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
