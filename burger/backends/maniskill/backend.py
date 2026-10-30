"""ManiSkill/SAPIEN GPU generation with direct, per-SVG canonical recording."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from burger.data import canonical as C
from burger.pipeline import svg_trajectory as traj
from burger.backends.base import (
    Backend,
    GenerationConfig,
    fovy_for,
    register_backend,
)
from burger.backends._worker_utils import validate_gpu_selection

DEFAULT_OPTIONS = dict(
    max_dots=1500,
    camera_view="front",
    camera_fov=None,
    randomize_tcp_init=False,
    robot_init_qpos_noise=0.0,
    robot_init_qpos=None,
    tcp_init_x_min=-0.05,
    tcp_init_x_max=0.05,
    tcp_init_y_min=-0.05,
    tcp_init_y_max=0.05,
    tcp_init_z=0.12,
    tcp_init_max_trials=10,
    appearance=None,
    appearance_random="none",
    appearance_seed=0,
    camera_preset=None,
    canvas_size=None,
)


def validate_options(options):
    unknown = set(options) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(
            f"Unknown ManiSkill backend options: {', '.join(sorted(unknown))}"
        )
    opt = dict(DEFAULT_OPTIONS, **options)
    if not isinstance(opt["randomize_tcp_init"], bool):
        raise ValueError("randomize_tcp_init must be a JSON boolean")
    for key in ("max_dots", "tcp_init_max_trials"):
        if isinstance(opt[key], bool) or not isinstance(opt[key], int) or opt[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (
        isinstance(opt["appearance_seed"], bool)
        or not isinstance(opt["appearance_seed"], int)
        or opt["appearance_seed"] < 0
    ):
        raise ValueError("appearance_seed must be a nonnegative integer")
    if opt["camera_view"] not in ("front", "side"):
        raise ValueError("camera_view must be front or side")
    if opt["appearance_random"] not in ("none", "mild", "wide"):
        raise ValueError("appearance_random must be none, mild or wide")
    if opt["appearance_random"] != "none" and opt["appearance"] is None:
        raise ValueError("appearance_random requires an explicit appearance preset")
    for key in (
        "tcp_init_x_min",
        "tcp_init_x_max",
        "tcp_init_y_min",
        "tcp_init_y_max",
        "tcp_init_z",
        "robot_init_qpos_noise",
    ):
        if (
            isinstance(opt[key], bool)
            or not isinstance(opt[key], (int, float))
            or not np.isfinite(opt[key])
        ):
            raise ValueError(f"{key} must be a finite JSON number")
    if opt["robot_init_qpos_noise"] < 0:
        raise ValueError("robot_init_qpos_noise cannot be negative")
    if (
        opt["tcp_init_x_min"] >= opt["tcp_init_x_max"]
        or opt["tcp_init_y_min"] >= opt["tcp_init_y_max"]
    ):
        raise ValueError("TCP sampling minimum must be below maximum")
    if opt["camera_fov"] is not None:
        if (
            isinstance(opt["camera_fov"], bool)
            or not isinstance(opt["camera_fov"], (int, float))
            or not 0 < opt["camera_fov"] < np.pi
        ):
            raise ValueError(
                "camera_fov must be a JSON number between 0 and pi radians"
            )
    if opt["canvas_size"] is not None:
        size = np.asarray(opt["canvas_size"], dtype=float)
        if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError("canvas_size must contain two positive metres")
    if opt["robot_init_qpos"] is not None:
        qpos = np.asarray(opt["robot_init_qpos"], dtype=float)
        if qpos.shape != (7,) or not np.isfinite(qpos).all():
            raise ValueError("robot_init_qpos must contain seven finite joint angles")
    return opt


@register_backend
class ManiSkillBackend(Backend):
    name = "maniskill"
    import_check = "mani_skill"

    def generate(
        self, cfg: GenerationConfig, out_canonical: Path, work_dir: Path, log=print
    ):
        opt = validate_options(cfg.backend_options)
        if not cfg.svg_files:
            raise ValueError("At least one SVG is required")
        cfg.device = validate_gpu_selection(cfg.device, backend="ManiSkill")
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import torch
        from burger.backends.maniskill.environment import DrawSVGParallelEnv  # noqa: F401
        from burger.backends.maniskill.collector import (
            record_episodes,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "ManiSkill generation requires an NVIDIA CUDA GPU and Vulkan renderer"
            )
        n_envs = min(max(1, int(cfg.parallel)), len(cfg.svg_files))
        svg_strings = [traj.read_svg_path_string(path) for path in cfg.svg_files]
        offsets = [
            np.random.default_rng([int(cfg.seed), i, 9011]).uniform(
                -cfg.svg_xy_noise, cfg.svg_xy_noise, 2
            )
            for i in range(len(svg_strings))
        ]
        waypoints = [
            traj.build_ee_waypoints(
                s,
                cfg.max_draw_dist,
                cfg.end_lift_z,
                cfg.rot_image,
                offset,
                cfg.svg_scale_mode,
            )
            for s, offset in zip(svg_strings, offsets)
        ]
        appearances = None
        if opt["appearance"] is not None:
            from burger.pipeline import appearance as A

            appearances = [
                A.resolve_appearance(
                    opt, episode_seed=i if opt["appearance_random"] != "none" else None
                )
                for i in range(n_envs)
            ]
        kwargs = {
            k: opt[k]
            for k in (
                "max_dots",
                "camera_view",
                "randomize_tcp_init",
                "robot_init_qpos_noise",
                "robot_init_qpos",
                "tcp_init_x_min",
                "tcp_init_x_max",
                "tcp_init_y_min",
                "tcp_init_y_max",
                "tcp_init_z",
                "tcp_init_max_trials",
                "camera_preset",
                "appearance_seed",
            )
        }
        kwargs.update(
            waypoints_list=waypoints,
            generation_seed=cfg.seed,
            appearance_list=appearances,
            camera_width=cfg.camera_width,
            camera_height=cfg.camera_height,
            camera_fov=opt["camera_fov"]
            if opt["camera_fov"] is not None
            else fovy_for(cfg.camera_width, cfg.camera_height),
            canvas_size=opt["canvas_size"] or (0.8, 1.2),
            wrist_camera=cfg.wrist_camera,
        )
        t0 = time.monotonic()
        env = gym.make(
            "DrawSVGParallel-v1",
            num_envs=n_envs,
            obs_mode="state_dict+rgb",
            control_mode="pd_ee_pose",
            render_mode="rgb_array",
            sim_backend=f"physx_cuda:{cfg.device}",
            render_backend=f"sapien_cuda:{cfg.device}",
            sensor_configs=dict(shader_pack="default"),
            **kwargs,
        )
        try:
            base = (
                env.unwrapped.agent.robot.get_root_pose()
                .p[0]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            info = C.DatasetInfo(
                backend="maniskill",
                language=cfg.language,
                control_freq=int(env.unwrapped.control_freq),
                robot="panda_stick",
                cameras=["agentview_rgb"]
                + (["eye_in_hand_rgb"] if cfg.wrist_camera else []),
                action_space=dict(
                    type="maniskill_pd_ee_pose_abs",
                    dim=6,
                    robot_base_world=base,
                    description="absolute robot-root-frame TCP target xyz (metres), Euler xyz (radians)",
                ),
                backend_config=dict(
                    env_id="DrawSVGParallel-v1",
                    options=opt,
                    num_envs=n_envs,
                    observation_timing="before_action",
                    appearance_scope="fixed_per_environment",
                    appearances=[a.to_dict() for a in appearances]
                    if appearances
                    else None,
                ),
                generation=cfg.to_json(),
            )
            with C.CanonicalWriter(out_canonical, info) as writer:
                record_episodes(
                    env,
                    writer,
                    cfg.svg_files,
                    svg_strings,
                    waypoints,
                    offsets,
                    seed=cfg.seed,
                    log=log,
                )
        finally:
            env.close()
        summary = C.summarize(out_canonical)
        elapsed = time.monotonic() - t0
        summary.update(
            backend_wall_s=elapsed,
            parallel=n_envs,
            steps_per_s=summary["total_steps"] / max(elapsed, 1e-9),
        )
        log(
            f"[maniskill] {summary['num_episodes']} episodes generated in {elapsed:.1f}s"
        )
        return summary
