"""LIBERO (robosuite + MuJoCo) backend: Franka Panda + pen, OSC_POSE delta actions (or [pd_ee] absolute IK + joint PD)."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

from burger.data import canonical as C
from burger.backends.base import Backend, GenerationConfig, REPO_ROOT, register_backend
from burger.backends._worker_utils import termination_as_interrupt, validate_parts, validate_options, validate_gpu_selection

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from burger.pipeline import svg_trajectory as traj  # noqa: E402

LIBERO_CAMERA_NAMES = ["robot0_eye_in_hand", "agentview"]


def set_egl_device(device):
    """Select EGL without altering a scheduler's CUDA mask, before importing robosuite."""
    if device is None:
        try:
            device = int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"))
        except ValueError as error:
            raise ValueError("MUJOCO_EGL_DEVICE_ID must be an integer") from error
    device = validate_gpu_selection(device, backend="LIBERO")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device)
    return device
DEFAULT_OPTIONS = dict(
    pen_length=0.05, fingertip_offset=0.0095, max_dots=1500, gain_pos=4.0, gain_rot=1.0, end_lift_steps=0,
    end_lift_height=0.35, hard_reset=False, hide_walls=True, hide_mount_visuals=True, store_states=True,
    compare_render="",                 # e.g. "320x240": extra agentview render stored as obs/painter_compare_rgb
    robot_init_qpos="maniskill_rest",
    arm_controller="osc",              # [pd_ee] "osc" (OSC_POSE deltas) | "pd_ee" (absolute target -> IK -> joint PD, like ManiSkill pd_ee_pose)
    # [pd_ee tail] ManiSkill-style episode tail: end_hold_steps x "hold here" (target = the pen tip at that step, like
    # ManiSkill's recorded hold step), then end_lift_steps x lift target;
    # end_lift_xy "last" = lift straight up at the last waypoint, "base" = ManiSkill's [0, 0, end_lift_height] in the base frame
    end_hold_steps=0, end_lift_xy="last",
    # [pd_ee start] 0 = off (every episode starts at the rest pose). N > 0: like ManiSkill's N parallel envs, which do not
    # reset the arm, episodes with svg_index >= N start where episode svg_index - N ended: an unrecorded preamble drives
    # the pen (up) to that episode's last waypoint and replays its hold + lift tail before recording starts
    start_from_prev_lift=0, start_approach_steps=120,
    start_settle_steps=0,              # [pd_ee start] extra unrecorded steps toward the lift target after the replayed tail
    # [pd_ee start] controller-side approach guard (planned target unchanged): until the pen tip first comes within
    # approach_guard_xy (m) in xy of a pen-down target, the commanded tip z is kept at least approach_guard_z (m, base frame); 0 = off
    approach_guard_xy=0.0, approach_guard_z=0.04,
    # [pd_ee real-tail] 1 = real-robot style raise (draw_svg_real_robot_osc_pose_cameras._record_raise): end_lift_steps
    # targets ramping linearly in z from the last waypoint to end_lift_height at the last xy; 0 = constant lift target
    end_lift_ramp=0,
    # [pd_ee tcp_init] native reset-time random TCP init (ManiSkill DrawSVGParallel randomize_tcp_init analogue): after
    # env.reset() the arm is IK-teleported (at rest, unrecorded, no ink) so the pen tip is at a seeded-uniform xy in
    # [tcp_init_x_min, tcp_init_x_max] x [tcp_init_y_min, tcp_init_y_max] (ManiSkill world, i.e. base frame x - 0.615)
    # and tip z tcp_init_z (base frame); recording starts there. 0 = off (rest pose)
    tcp_init_random=0, tcp_init_x_min=-0.05, tcp_init_x_max=0.05, tcp_init_y_min=-0.05, tcp_init_y_max=0.05,
    tcp_init_z=0.12, tcp_init_max_trials=10,
    timeout_s=4 * 3600,
)

EXTRA_OPTIONS = (
    "appearance", "appearance_random", "appearance_seed", "camera_preset", "canvas_size",
    "table_length", "table_width", "table_x_behind_base", "table_y_offset", "table_top_z",
    "camera_fovx_deg", "camera_fovy_deg", "pen_tip_offset", "touch_tolerance_z",
    "scene_textures", "table_texrepeat", "floor_texrepeat", "table_rgba", "floor_rgba", "floor_size",
    "scene_lighting", "light_ambient", "light_diffuse", "fill_diffuse", "light_specular", "headlight_diffuse",
    "headlight_ambient", "pd_ee_kp", "pd_ee_damping_ratio", "pd_ee_ik_lambda", "pd_ee_max_joint_vel", "pd_ee_joint_margin",
)

def _end_tail(layout, wps: np.ndarray, opt: dict) -> np.ndarray:
    """[pd_ee tail] world-frame targets appended after the drawing waypoints ``wps`` (painter/base frame)."""
    parts = []
    if int(opt.get("end_hold_steps", 0)) > 0:
        parts.append(np.full((int(opt["end_hold_steps"]), 3), np.nan))   # filled with the current pen tip at run time
    if int(opt["end_lift_steps"]) > 0:
        xy = (0.0, 0.0) if str(opt.get("end_lift_xy", "last")) == "base" else (wps[-1, 0], wps[-1, 1])
        n = int(opt["end_lift_steps"])
        if int(opt.get("end_lift_ramp", 0)) > 0:  # [pd_ee real-tail] z: last waypoint -> max(last z, end_lift_height)
            z0 = float(wps[-1, 2]); z1 = max(z0, float(opt["end_lift_height"]))
            zs = z0 + (z1 - z0) * np.arange(1, n + 1) / n
            parts.append(layout.painter_to_world(np.stack([np.full(n, xy[0]), np.full(n, xy[1]), zs], 1)))
        else:
            lift = layout.painter_to_world(np.array([xy[0], xy[1], float(opt["end_lift_height"])]))
            parts.append(np.repeat(lift[None], n, 0))
    return np.concatenate(parts, 0) if parts else np.zeros((0, 3))


def _episode_waypoints(svg_file, cfg: GenerationConfig, xy_offset=(0.0, 0.0)) -> np.ndarray:
    return traj.build_ee_waypoints(traj.read_svg_path_string(svg_file), cfg.max_draw_dist, cfg.end_lift_z, rot_image=cfg.rot_image,
                                   xy_offset=tuple(xy_offset), scale_mode=cfg.svg_scale_mode)


def _episode_offset(cfg: GenerationConfig, svg_index: int):
    rng = np.random.default_rng([int(cfg.seed), int(svg_index), 9011])
    return rng.uniform(-cfg.svg_xy_noise, cfg.svg_xy_noise, size=2) if cfg.svg_xy_noise > 0 else np.zeros(2)


def _start_preamble(env, layout, prev_svg_file, cfg: GenerationConfig, opt: dict, prev_svg_index: int):
    """[pd_ee start] unrecorded: from the reset pose, bring the pen (lifted) to the previous episode's last waypoint,
    then run that episode's hold + lift tail, so recording starts from the (moving) post-lift arm like ManiSkill."""
    penv = env.env
    prev_wps = _episode_waypoints(prev_svg_file, cfg, _episode_offset(cfg, prev_svg_index))
    goal = layout.painter_to_world(prev_wps[-1])
    obs, n = None, 0
    for n in range(int(opt.get("start_approach_steps", 120))):
        a, _ = penv.pd_ee_action(goal); obs, _, _, _ = env.step(a)
        if np.linalg.norm(penv.pen_tip_pos() - goal) < 2e-3:
            break
    tail = _end_tail(layout, prev_wps, opt)
    for t in tail:
        a, _ = penv.pd_ee_action(penv.pen_tip_pos().copy() if np.isnan(t).any() else t); obs, _, _, _ = env.step(a)
    for _ in range(int(opt.get("start_settle_steps", 0)) if len(tail) else 0):
        target = penv.pen_tip_pos().copy() if np.isnan(tail[-1]).any() else tail[-1]
        a, _ = penv.pd_ee_action(target); obs, _, _, _ = env.step(a)
    return obs, dict(start_prev_svg=str(prev_svg_file), start_approach_steps=int(n + 1), start_preamble_ink=int(penv.ink_count),
                     start_tip_painter=layout.world_to_painter(penv.pen_tip_pos()).tolist())


def _random_tcp_init(env, layout, cfg: GenerationConfig, opt: dict, svg_index: int):
    """[pd_ee tcp_init] seeded per episode (cfg.seed + svg_index, independent of worker split), like ManiSkill seeding the
    TCP-init generator with the episode seed. Retries a new xy when IK does not reach it (ManiSkill: tcp_init_max_trials)."""
    penv = env.env
    g = np.random.default_rng([int(cfg.seed), int(svg_index), 7477])
    for trial in range(int(opt.get("tcp_init_max_trials", 10))):
        xy = np.array([g.uniform(float(opt["tcp_init_x_min"]), float(opt["tcp_init_x_max"])) + 0.615,  # ManiSkill world -> base
                       g.uniform(float(opt["tcp_init_y_min"]), float(opt["tcp_init_y_max"]))])
        goal_p = np.array([xy[0], xy[1], float(opt["tcp_init_z"])])
        err = penv.teleport_pen_tip(layout.painter_to_world(goal_p))
        if err <= 1e-3:
            return dict(tcp_init_target_painter=goal_p.tolist(), tcp_init_trials=trial + 1, tcp_init_err_m=err)
    raise RuntimeError(f"Could not initialize SVG {svg_index} TCP within 1 mm after {opt.get('tcp_init_max_trials', 10)} trials")


def _env_kwargs(cfg: GenerationConfig, opt: dict, appearance=None) -> dict:
    kw = dict(max_dots=int(opt["max_dots"]), pen_length=float(opt["pen_length"]), fingertip_offset=float(opt["fingertip_offset"]),
              hide_walls=bool(opt["hide_walls"]), hide_mount_visuals=bool(opt["hide_mount_visuals"]), robot_init_qpos=opt["robot_init_qpos"],
              arm_controller=str(opt.get("arm_controller", "osc")))  # [pd_ee]
    if appearance is not None:
        kw["appearance"] = appearance.to_dict() if hasattr(appearance, "to_dict") else appearance
        kw["appearance_seed"] = int(opt.get("appearance_seed", 0))
    if opt.get("camera_preset") not in (None, "", "none", "maniskill"):
        kw["camera_preset"] = opt["camera_preset"]
    if opt.get("canvas_size") is not None:
        kw["canvas_size"] = [float(v) for v in opt["canvas_size"]]
    for k in ("table_length", "table_width", "table_x_behind_base", "table_y_offset", "table_top_z", "camera_fovx_deg", "camera_fovy_deg", "pen_tip_offset", "touch_tolerance_z",
              "scene_textures", "table_texrepeat", "floor_texrepeat", "table_rgba", "floor_rgba", "floor_size",
              "scene_lighting", "light_ambient", "light_diffuse", "fill_diffuse", "light_specular", "headlight_diffuse", "headlight_ambient",
              "pd_ee_kp", "pd_ee_damping_ratio", "pd_ee_ik_lambda", "pd_ee_max_joint_vel", "pd_ee_joint_margin"):  # [pd_ee]
        if k in opt and opt[k] is not None:
            kw[k] = opt[k]
    return kw


def _make_env(bddl_path: str, cfg: GenerationConfig, opt: dict, appearance=None):
    from libero.libero.envs import OffScreenRenderEnv
    import burger.backends.libero.environment  # noqa: F401

    cams = LIBERO_CAMERA_NAMES if cfg.wrist_camera else ["agentview"]
    return OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=cfg.camera_height, camera_widths=cfg.camera_width, camera_names=cams,
                              hard_reset=bool(opt["hard_reset"]), ignore_done=True, horizon=10**9, **_env_kwargs(cfg, opt, appearance))


def _uses_appearance(opt: dict) -> bool:
    return opt.get("appearance") is not None or str(opt.get("appearance_random", "none")) != "none"


def _episode_appearance(opt: dict, svg_index: int):
    """Appearance for one episode: the preset, or a per-episode draw when ``appearance_random`` is mild / wide."""
    if not _uses_appearance(opt):
        return None
    from burger.pipeline import appearance as A
    mode = str(opt.get("appearance_random", "none"))
    return A.resolve_appearance(dict(opt, appearance=opt.get("appearance", "maniskill")), episode_seed=svg_index if mode != "none" else None)


def run_episode(env, layout, svg_file: Path, cfg: GenerationConfig, opt: dict, rng: np.random.Generator, svg_index: int) -> C.Episode:
    penv = env.env
    svg_string = traj.read_svg_path_string(svg_file)
    # A worker's scheduling and the lengths of previous SVGs must not change this episode.
    xy_offset = _episode_offset(cfg, svg_index)
    wps = traj.build_ee_waypoints(svg_string, cfg.max_draw_dist, cfg.end_lift_z, rot_image=cfg.rot_image, xy_offset=tuple(xy_offset), scale_mode=cfg.svg_scale_mode)
    wps_world = layout.painter_to_world(wps)
    targets = np.concatenate([wps_world, _end_tail(layout, wps, opt)], 0)  # [pd_ee tail] (unchanged when hold/lift steps are 0)
    compare_wh = None
    if opt.get("compare_render"):
        w, h = str(opt["compare_render"]).lower().split("x"); compare_wh = (int(w), int(h))

    obs = env.reset()
    start_meta = {}
    if int(opt.get("start_from_prev_lift", 0)) > 0 and svg_index >= int(opt["start_from_prev_lift"]):  # [pd_ee start]
        penv.set_ink_enabled(False)
        previous_index = svg_index - int(opt["start_from_prev_lift"])
        obs, start_meta = _start_preamble(env, layout, cfg.svg_files[previous_index], cfg, opt, previous_index)
        penv.set_ink_enabled(True)
    if int(opt.get("tcp_init_random", 0)) > 0:  # [pd_ee tcp_init] native reset-time teleport, then fresh obs
        start_meta.update(_random_tcp_init(env, layout, cfg, opt, svg_index))
        obs = env.env._get_observations(force_update=True)
    penv.set_target_waypoints(wps_world)
    model_xml = env.sim.model.get_xml()
    T_steps = len(targets)
    H, W = cfg.camera_height, cfg.camera_width
    agent = np.zeros((T_steps, H, W, 3), np.uint8)
    wrist = np.zeros((T_steps, H, W, 3), np.uint8) if cfg.wrist_camera else None
    eef_pos = np.zeros((T_steps, 3)); eef_quat_xyzw = np.zeros((T_steps, 4)); joint = np.zeros((T_steps, 7)); grip = np.zeros((T_steps, 2))
    pd_ee = str(opt.get("arm_controller", "osc")) == "pd_ee"  # [pd_ee]
    tip_w = np.zeros((T_steps, 3)); ink = np.zeros(T_steps, np.int32); actions = np.zeros((T_steps, 8 if pd_ee else 7))
    goal_clipped = 0
    guard_xy = float(opt.get("approach_guard_xy", 0.0)); guarded = 0; guard_on = guard_xy > 0  # [pd_ee start]
    guard_zw = float(layout.painter_to_world(np.array([0.0, 0.0, float(opt.get("approach_guard_z", 0.04))]))[2])
    states = [] if opt["store_states"] else None
    compare = np.zeros((T_steps, compare_wh[1], compare_wh[0], 3), np.uint8) if compare_wh else None
    tip_err = np.zeros(T_steps)
    commanded_targets = np.zeros_like(targets)
    t0 = time.time()
    for i in range(T_steps):
        agent[i] = obs["agentview_image"][::-1]                     # OpenGL -> top-down
        if wrist is not None:
            wrist[i] = obs["robot0_eye_in_hand_image"][::-1]
        eef_pos[i] = obs["robot0_eef_pos"]; eef_quat_xyzw[i] = obs["robot0_eef_quat"]
        joint[i] = obs["robot0_joint_pos"]; grip[i] = obs["robot0_gripper_qpos"]
        tip_w[i] = penv.pen_tip_pos(); ink[i] = penv.ink_count
        if states is not None:
            states.append(env.get_sim_state())
        if compare is not None:
            compare[i] = penv.render_frame("agentview", compare_wh[0], compare_wh[1])[::-1]
        if np.isnan(targets[i]).any():  # [pd_ee tail] hold step: target = current pen tip
            targets[i] = penv.pen_tip_pos()
        cmd = targets[i]
        if guard_on and cmd[2] < guard_zw:  # [pd_ee start] approach: keep the pen up until it has caught the target once (latched)
            if np.linalg.norm(tip_w[i, :2] - cmd[:2]) > guard_xy:
                cmd = np.array([cmd[0], cmd[1], guard_zw]); guarded += 1
            else:
                guard_on = False
        if pd_ee:  # [pd_ee]
            a, clipped = penv.pd_ee_action(cmd); goal_clipped += int(clipped)
        else:
            a = penv.waypoint_action(cmd, gain_pos=float(opt["gain_pos"]), gain_rot=float(opt["gain_rot"]))
        commanded_targets[i] = cmd
        actions[i] = a
        obs, _, _, _ = env.step(a)
        tip_err[i] = np.linalg.norm(penv.pen_tip_pos() - targets[i])
    penv.mark_episode_complete(True)
    down = targets[:, 2] < penv.touch_z_threshold
    ep_obs = dict(agentview_rgb=agent, eef_pos=eef_pos, eef_quat_wxyz=C.quat_xyzw_to_wxyz(eef_quat_xyzw), joint_pos=joint, gripper_qpos=grip,
                  pen_tip_world=tip_w, pen_tip_painter=layout.world_to_painter(tip_w), ink_count=ink)
    if wrist is not None:
        ep_obs["eye_in_hand_rgb"] = wrist
    if compare is not None:
        ep_obs["painter_compare_rgb"] = compare
    arrays = dict(waypoints_world=wps_world, waypoints_painter=wps, targets_world=targets,
                  commanded_targets_world=commanded_targets, ink_dots_xyz=penv.ink_positions())
    if states is not None:
        arrays["backend_state"] = np.stack(states, 0)
    meta = dict(svg_file=str(svg_file), svg_index=int(svg_index), svg_string=svg_string, xy_offset=xy_offset.tolist(), ink_count=int(penv.ink_count),
                pen_down_waypoints=int(down.sum()), ink_coverage=float(penv.ink_coverage()), tip_error_mean_m=float(tip_err.mean()),
                tip_error_p95_m=float(np.percentile(tip_err, 95)), tip_error_max_m=float(tip_err.max()),
                tip_error_pen_down_mean_m=float(tip_err[down].mean()) if down.any() else 0.0, gen_time_s=time.time() - t0,
                model_file=model_xml, end_lift_steps=int(opt["end_lift_steps"]), trajectory_complete=True, truncated=False,
                episode_seed=int(cfg.seed) + int(svg_index), visual_success_evaluated=False,
                arm_controller=str(opt.get("arm_controller", "osc")), pd_ee_goal_clipped_steps=int(goal_clipped),  # [pd_ee]
                joint_pos_min=joint.min(0).tolist(), joint_pos_max=joint.max(0).tolist(),
                end_hold_steps=int(opt.get("end_hold_steps", 0)), end_lift_xy=str(opt.get("end_lift_xy", "last")), end_lift_ramp=int(opt.get("end_lift_ramp", 0)), approach_guarded_steps=int(guarded), **start_meta)  # [pd_ee tail/start]
    return C.Episode(obs=ep_obs, actions=actions, meta=meta, arrays=arrays)


def _worker(rank: int, jobs: list, bddl_path: str, layout_dict: dict, cfg_dict: dict, opt: dict, part_path: str, info_attrs: dict, log_path: str):
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    cfg = GenerationConfig(**cfg_dict)
    cfg.device = set_egl_device(cfg.device)
    from burger.backends.libero.environment import SceneLayout
    layout = SceneLayout(**layout_dict)
    info = C.DatasetInfo(**info_attrs)
    env = None
    env_app_json = None
    try:
        with C.CanonicalWriter(part_path, info) as w, open(log_path, "w") as log:
            for svg_index, svg_file in jobs:
                t0 = time.time()
                app = _episode_appearance(opt, svg_index)
                app_json = app.to_json() if app is not None else None
                if env is None or app_json != env_app_json:
                    if env is not None:
                        env.close()
                        env = None
                    env = _make_env(bddl_path, cfg, opt, app)
                    env_app_json = app_json
                env.seed(int(cfg.seed) + int(svg_index))
                ep = run_episode(env, layout, Path(svg_file), cfg, opt, None, svg_index)
                if app is not None:
                    ep.meta["appearance"] = app_json
                w.add_episode(ep, index=svg_index)
                m = ep.meta
                msg = (f"[libero worker {rank}] svg {svg_index} {Path(svg_file).name}: {ep.num_steps} steps, "
                       f"ink {m['ink_count']}/{m['pen_down_waypoints']}, coverage {m['ink_coverage']:.3f}, "
                       f"tip err {m['tip_error_mean_m']*1000:.2f} mm, {time.time()-t0:.1f}s")
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
    finally:
        if env is not None:
            env.close()


@register_backend
class LiberoBackend(Backend):
    name = "libero"
    import_check = "robosuite"

    def generate(self, cfg: GenerationConfig, out_canonical: Path, work_dir: Path, log=print) -> dict:
        opt = validate_options(cfg.backend_options, DEFAULT_OPTIONS, EXTRA_OPTIONS)
        os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        cfg.device = set_egl_device(cfg.device)
        import robosuite as suite
        import burger.backends.libero.environment  # noqa: F401
        from burger.backends.libero.bddl_utils import write_bddl
        from burger.backends.libero.environment import ENV_NAME, PROBLEM_NAME, compute_scene_layout

        if not cfg.svg_files:
            raise ValueError("LIBERO generation requires at least one SVG")
        cfg.svg_files = [str(Path(f).resolve()) for f in cfg.svg_files]
        work_root = Path(work_dir).resolve(); work_root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="libero-", dir=work_root))
        canvas_size = tuple(float(v) for v in opt["canvas_size"]) if opt.get("canvas_size") is not None else None
        if _uses_appearance(opt):
            # the appearance's table extent is geometry shared by the BDDL layout and every (randomized) episode
            from burger.pipeline import appearance as A
            base_app = A.resolve_appearance(dict(opt, appearance=opt.get("appearance", "maniskill")), episode_seed=None)
            for k, v in A.table_layout_kwargs(base_app, canvas_size).items():
                opt.setdefault(k, float(v))
        if opt.get("camera_preset") not in (None, "", "none", "maniskill"):
            log(f"[libero backend] camera preset: {opt['camera_preset']}")
        layout_kwargs = {k: opt[k] for k in ("table_length", "table_width", "table_x_behind_base", "table_y_offset", "table_top_z") if k in opt and opt[k] is not None}
        if opt.get("camera_fovx_deg") is not None:
            layout_kwargs["camera_fovx_rad"] = np.radians(float(opt["camera_fovx_deg"]))
        if opt.get("camera_fovy_deg") is not None:
            layout_kwargs["camera_fovy_rad"] = np.radians(float(opt["camera_fovy_deg"]))
        if canvas_size is not None:
            layout_kwargs["canvas_half_size"] = (canvas_size[0] / 2, canvas_size[1] / 2)
            log(f"[libero backend] canvas size: {list(canvas_size)}")
        layout = compute_scene_layout(camera_aspect=cfg.camera_width / cfg.camera_height, **layout_kwargs)
        bddl_path = write_bddl(work_dir / f"{cfg.object_name}.bddl", cfg.language, layout)
        cams = LIBERO_CAMERA_NAMES if cfg.wrist_camera else ["agentview"]
        env_kwargs = dict(robots=["Panda"], controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),  # [pd_ee] env replaces it when arm_controller=pd_ee
                          bddl_file_name=str(bddl_path),
                          has_renderer=False, has_offscreen_renderer=True, ignore_done=True, use_camera_obs=True, camera_depths=False, camera_names=cams,
                          reward_shaping=True, control_freq=20, camera_heights=cfg.camera_height, camera_widths=cfg.camera_width, camera_segmentations=None)
        env_kwargs.update(_env_kwargs(cfg, opt))
        env_args = {"type": 1, "env_name": ENV_NAME, "problem_name": PROBLEM_NAME, "bddl_file": str(bddl_path), "env_kwargs": env_kwargs}
        info = C.DatasetInfo(
            backend="libero", language=cfg.language, control_freq=20, robot="panda_gripper_pen" if float(opt["pen_length"]) > 0 else "panda_gripper",
            cameras=["agentview_rgb"] + (["eye_in_hand_rgb"] if cfg.wrist_camera else []),
            action_space=(dict(type="libero_osc_pose_delta", dim=7, description="robosuite OSC_POSE deltas in [-1, 1]: dx dy dz (x0.05 m) droll dpitch dyaw (x0.5 rad) gripper (+1 closed)",
                               gain_pos=float(opt["gain_pos"]), gain_rot=float(opt["gain_rot"]))
                          if str(opt.get("arm_controller", "osc")) != "pd_ee" else  # [pd_ee]
                          dict(type="libero_pd_ee_joint_position", dim=8, description="pd_ee: absolute pen-tip target (arrays/targets_world) -> 1 DLS IK step -> "
                               "robosuite JOINT_POSITION delta goal (rad, x1.0) for 7 joints + gripper (+1 closed)", controller="JOINT_POSITION")),
            backend_config=dict(env_args=env_args, problem_info={"problem_name": PROBLEM_NAME, "domain_name": "robosuite", "language_instruction": cfg.language},
                                bddl_file=str(bddl_path), bddl_file_content=Path(bddl_path).read_text(), robosuite_version=suite.__version__, scene_layout=layout.to_dict(),
                                options=opt, parallel_workers=int(cfg.parallel), device=cfg.device),
            generation=cfg.to_json(),
        )
        jobs = list(enumerate([str(f) for f in cfg.svg_files]))
        n_workers = max(1, min(int(cfg.parallel), len(jobs)))
        parts_dir = work_dir / "parts"; parts_dir.mkdir(exist_ok=True)
        info_attrs = dict(backend=info.backend, language=info.language, control_freq=info.control_freq, robot=info.robot, cameras=info.cameras,
                          action_space=info.action_space, backend_config=info.backend_config, generation=info.generation)
        t_start = time.time()
        parts = []
        ctx = mp.get_context("spawn"); procs = []
        with termination_as_interrupt():
            try:
                for rank in range(n_workers):
                    part = parts_dir / f"part_{rank}.h5"; parts.append(part)
                    log_path = work_dir / f"worker_{rank}.log"
                    p = ctx.Process(target=_worker, args=(rank, jobs[rank::n_workers], str(bddl_path), layout.to_dict(), cfg.to_json(), opt, str(part), info_attrs, str(log_path)))
                    p.start(); procs.append(p)
                deadline = time.monotonic() + float(opt["timeout_s"])
                while any(p.is_alive() for p in procs):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"LIBERO workers exceeded {opt['timeout_s']} s; see {work_dir}")
                    for p in procs:
                        p.join(timeout=0.1)
                        if p.exitcode not in (None, 0):
                            raise RuntimeError(f"LIBERO worker exited with code {p.exitcode}; see {work_dir}")
                if any(p.exitcode != 0 for p in procs):
                    raise RuntimeError(f"LIBERO worker failed; see {work_dir}")
            finally:
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                for p in procs:
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=5)
        gen_wall = time.time() - t_start
        validate_parts(parts, cfg.svg_files)
        C.merge_parts(parts, out_canonical, info)
        for part in parts:
            Path(part).unlink(missing_ok=True)
        summ = C.summarize(out_canonical)
        summ.update(dict(backend_wall_s=gen_wall, parallel=n_workers, steps_per_s=summ["total_steps"] / gen_wall, bddl=str(bddl_path)))
        log(f"[libero backend] {summ['num_episodes']} episodes, {summ['total_steps']} steps in {gen_wall:.1f}s ({summ['steps_per_s']:.1f} steps/s, {n_workers} workers)")
        return summ
