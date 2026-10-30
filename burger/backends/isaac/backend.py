"""Isaac Sim backend (PhysX + RTX through Isaac Lab): ManiSkill's ``DrawSVGParallel-v1`` scene rebuilt
in Isaac Sim with the same ``panda_stick`` robot, table, canvas, camera and ink-dot mechanism.

The simulation itself runs in a separate worker process (``python -m burger.backends.isaac.worker``)
inside the Isaac Sim virtualenv, because Omniverse Kit must own the process (it is started once per process
and exits hard on shutdown). The worker writes a canonical part file that is merged here, so this module
only needs h5py/numpy and can be driven from any of the pipeline virtualenvs.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import tempfile
import uuid
from pathlib import Path

from burger.data import canonical as C
from burger.backends.base import Backend, GenerationConfig, REPO_ROOT, register_backend
from burger.backends._worker_utils import stop_process_group, termination_as_interrupt, validate_parts, validate_options, validate_gpu_selection

WORKER_MODULE = "burger.backends.isaac.worker"

# Backend-specific knobs (all overridable with --backend_option key=value). Physics values mirror
# ManiSkill's DrawSVGParallel-v1 / panda_stick pd_ee_pose controller.
DEFAULT_OPTIONS = dict(
    max_dots=1500,                 # ink dots pre-allocated per env (ManiSkill MAX_DOTS)
    ik="maniskill_lm",             # "maniskill_lm": one Levenberg-Marquardt step (J^T J + lambda I)^-1 J^T dx like ManiSkill; "dls": Isaac Lab damped least squares
    ik_lambda=1e-4,                # ManiSkill's regularisation (Isaac Lab dls uses lambda_val^2 = 1e-4 as well)
    stiffness=1000.0, damping=100.0, effort_limit=100.0,   # PhysX joint drive (ManiSkill arm_stiffness/damping/force_limit)
    disable_robot_gravity=True,    # ManiSkill's BaseAgent disables gravity on robot links (no gravity sag with pure PD)
    sim_dt=0.01, decimation=5,     # sim_freq 100 Hz, control_freq 20 Hz
    solver_position_iterations=4, solver_velocity_iterations=0,
    contact_offset=0.01, rest_offset=0.0,
    env_spacing=12.0,              # metres between cloned envs (keeps neighbours out of the 84.7 deg camera)
    hold_steps=1, raise_steps=0, raise_height=0.35,  # the ManiSkill reference trajectories contain the waypoints + 1 hold step (raise_arms steps are not recorded)
    # [isaac v6] alignment with LIBERO v6 / real_3class_30 (all off by default = previous behaviour):
    # raise_xy "base" = ManiSkill's [0, 0, raise_height] target, "last" = straight up at the last waypoint;
    # raise_ramp 1 = real-robot _record_raise: z linear from the last waypoint to raise_height over raise_steps
    raise_xy="base", raise_ramp=0,
    # reset-time random TCP init (ManiSkill randomize_tcp_init / LIBERO tcp_init_random): seeded by (seed, svg_index), TCP xy uniform
    # in the ManiSkill-world box, z = tcp_init_z (root frame), stick down; joints solved by iterated LM and written at reset
    tcp_init_random=0, tcp_init_x_min=-0.05, tcp_init_x_max=0.05, tcp_init_y_min=-0.05, tcp_init_y_max=0.05, tcp_init_z=0.12, tcp_init_max_trials=10,
    # controller-side approach guard (LIBERO approach_guard_xy): until the TCP is within approach_guard_xy (m) in xy of a pen-down
    # target, its commanded z is kept >= approach_guard_z; planned targets stay separately recorded. 0 = off
    approach_guard_xy=0.0, approach_guard_z=0.04,
    max_joint_vel=0.0, joint_margin=0.0,   # Franka joint-velocity cap (x datasheet) / joint margin (rad) on the IK goal; 0 = off
    render_mode="RealTimePathTracing",  # Kit 110's RTX real-time renderer (the classic RaytracedLighting mode is no longer selectable)
    light_shadows=False, light_angle_deg=0.0,  # directional lights cast no shadows (UsdLux ShadowAPI) and have no penumbra, like ManiSkill's rasteriser
    antialiasing="FXAA",           # no temporal history from a previous episode; RTX sampling noise may remain
    sampled_lighting=True, samples_per_pixel=1, tonemap_op=1,  # RTPT requires sampled direct lighting; samples_per_pixel controls direct-light samples only; tonemap 1 = Linear
    enable_shadows=False,          # ManiSkill's default lighting renders without shadows
    ambient_intensity=1.0, dome_intensity=0.0, sun_intensity=900.0, fill_intensity=0.0,   # RTX flat ambient + ManiSkill's (1,1,-1) directional light, tuned so canvas/table/floor brightness match the ManiSkill renders
    table_visual=True,             # ManiSkill's table.glb (converted to USD) vs a plain box
    ground_texture=True,           # ManiSkill grid_texture.png ground vs Isaac default grid
    store_states=True,             # joint pos/vel per step in arrays/backend_state
    compare_render="",             # e.g. "320x240": extra base-camera render stored as obs/painter_compare_rgb
    keep_work=True,
    timeout_s=4 * 3600,            # kill the worker if it exceeds this
    extra_kit_args="",             # forwarded to Kit, e.g. "--/rtx/sceneDb/maxInstances=500000"
)

EXTRA_OPTIONS = ("appearance", "appearance_random", "appearance_seed", "camera_preset", "canvas_size", "canvas_roughness")


@register_backend
class IsaacBackend(Backend):
    name = "isaac"
    import_check = "isaaclab.app"

    @classmethod
    def available(cls) -> bool:
        # the simulation runs in a subprocess with the Isaac venv, so this process only needs the venv to exist
        return cls.venv_python().exists()

    def generate(self, cfg: GenerationConfig, out_canonical: Path, work_dir: Path, log=print) -> dict:
        opt = validate_options(cfg.backend_options, DEFAULT_OPTIONS, EXTRA_OPTIONS)
        cfg.device = validate_gpu_selection(cfg.device, backend="Isaac")
        if not cfg.svg_files:
            raise ValueError("Isaac generation requires at least one SVG")
        work_root = Path(work_dir).resolve(); work_root.mkdir(parents=True, exist_ok=True)
        # Never reuse a previous run's summary or partially written HDF5 file.
        work_dir = Path(tempfile.mkdtemp(prefix="isaac-", dir=work_root))
        parts_dir = work_dir / "parts"; parts_dir.mkdir(exist_ok=True)
        part = parts_dir / "part_0.h5"
        cfg.svg_files = [str(Path(f).resolve()) for f in cfg.svg_files]
        run_id = uuid.uuid4().hex
        job = dict(run_id=run_id, svg_files=cfg.svg_files, cfg=cfg.to_json(), options=opt, part=str(part))
        job_path = work_dir / "isaac_job.json"; job_path.write_text(json.dumps(job, indent=2))
        summary_path = work_dir / "isaac_worker_summary.json"
        log_path = work_dir / "isaac_backend.log"
        py = self.venv_python()
        if not py.exists():
            raise SystemExit(f"Isaac Sim virtualenv {py} not found; run scripts/setup_isaac.sh")
        cmd = [str(py), "-m", WORKER_MODULE, "--job_json", str(job_path), "--summary_json", str(summary_path)]
        if cfg.device is not None:
            cmd += ["--device", str(cfg.device)]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        log(f"[isaac backend] launching worker: {' '.join(cmd)} (log: {log_path})")
        t_start = time.time()
        with open(log_path, "w") as lf, termination_as_interrupt():
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                rc = proc.wait(timeout=float(opt["timeout_s"]))
            except subprocess.TimeoutExpired:
                stop_process_group(proc)
                raise RuntimeError(f"Isaac worker exceeded {opt['timeout_s']} s; see {log_path}")
            except BaseException:
                stop_process_group(proc)
                raise
        process_wall = time.time() - t_start
        if rc != 0 or not summary_path.exists() or not part.exists():
            tail = "".join(Path(log_path).read_text(errors="replace").splitlines(True)[-30:])
            raise RuntimeError(f"Isaac worker failed (exit code {rc}); see {log_path}\n{tail}")
        wsum = json.loads(summary_path.read_text())
        if wsum.get("status") != "complete" or wsum.get("run_id") != run_id:
            raise RuntimeError(f"Isaac worker did not report completion for this run; see {summary_path} and {log_path}")
        # Kit exits hard; the worker's HDF5 part may still be flushing / locked for a moment after the process is gone
        info = None
        for attempt in range(15):
            try:
                info = C.read_info(part)
                break
            except OSError as e:
                if attempt == 14:
                    raise
                log(f"[isaac backend] part file not readable yet ({str(e)[:80]}...); retrying in 2 s")
                time.sleep(2.0)
        actual = validate_parts([part], cfg.svg_files)
        if any(int(wsum.get(key, -1)) != value for key, value in actual.items()):
            raise RuntimeError(f"Isaac worker summary disagrees with its output: {actual}; see {log_path}")
        C.merge_parts([part], out_canonical, info)
        if not bool(opt["keep_work"]):
            part.unlink(missing_ok=True)
        summ = C.summarize(out_canonical)
        summ.update(dict(
            backend_wall_s=wsum["backend_wall_s"], sim_wall_s=wsum["sim_wall_s"], scene_build_s=wsum["scene_build_s"], app_startup_s=wsum["app_startup_s"],
            process_wall_s=process_wall, parallel=int(cfg.parallel), steps_per_s=summ["total_steps"] / max(wsum["backend_wall_s"], 1e-9),
            sim_steps_per_s=summ["total_steps"] / max(wsum["sim_wall_s"], 1e-9), worker_summary=wsum, log=str(log_path),
        ))
        log(f"[isaac backend] {summ['num_episodes']} episodes, {summ['total_steps']} steps in {wsum['backend_wall_s']:.1f}s "
            f"({summ['steps_per_s']:.1f} steps/s incl. scene build; {summ['sim_steps_per_s']:.1f} steps/s sim loop; {cfg.parallel} envs; "
            f"Kit startup {wsum['app_startup_s']:.1f}s)")
        return summ
