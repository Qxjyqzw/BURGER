#!/usr/bin/env python
"""Unified painter data generation: choose the simulation *backend* (physics + rendering) and the
output *format* independently.

  python -m burger.pipeline.generate --backend libero --svg_dir SVG --output_dir OUT --object_name house --source_desc "draw a house"
  python -m burger.pipeline.generate --backend isaac --svg_dir SVG --output_dir OUT --object_name house --source_desc "draw a house" --parallel 2 --device 0

The script re-executes itself in the virtualenv that has the requested backend
(.venv-maniskill / .venv-libero / .venv-isaac61) when needed, and delegates the LeRobot writer to
.venv-lerobot-export (or PAINTER_LEROBOT_PYTHON). Output layout:

  OUT/<object_name>/<backend>/<object_name>.canonical.h5      backend-agnostic episodes (kept for re-export)
  OUT/<object_name>/<backend>/<object_name>_demo.hdf5         LIBERO format
  OUT/<object_name>/<backend>/<lerobot repo>/                 LeRobot v2.1 dataset
  OUT/<object_name>/<backend>/generation_summary.json         timings + settings
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from burger._paths import PROJECT_ROOT as _REPO_ROOT

from burger.data.exporters import export as export_mod  # noqa: E402
from burger.backends import GenerationConfig, get_backend  # noqa: E402
from burger.backends.base import VENV_FOR_BACKEND  # noqa: E402


def parse_kv(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise argparse.ArgumentTypeError(f"--backend_option expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        if not k or k in out:
            raise ValueError(f"backend options must have nonempty, unique keys: {k!r}")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def build_arg_parser():
    p = argparse.ArgumentParser(description="Generate BURGER drawing data with a chosen backend and output format(s)")
    p.add_argument("--backend", choices=sorted(VENV_FOR_BACKEND), required=True)
    p.add_argument("--format", default="canonical,lerobot", help="comma separated output formats: canonical, lerobot, libero")
    p.add_argument("--svg_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--object_name", default="drawing")
    p.add_argument("--source_desc", required=True, help="exact language instruction for every SVG in this run")
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    # trajectory
    p.add_argument("--max_draw_dist", type=float, default=0.008)
    p.add_argument("--end_lift_z", type=float, default=0.085)
    p.add_argument("--svg_xy_noise", type=float, default=0.0)
    p.add_argument("--svg_scale_mode", choices=["fixed", "max"], default="fixed")
    p.add_argument("--no_rot_image", action="store_true")
    # cameras
    p.add_argument("--camera_height", type=int, default=128)
    p.add_argument("--camera_width", type=int, default=128)
    p.add_argument("--no_wrist_camera", action="store_true")
    # execution
    p.add_argument("--parallel", type=int, default=1, help="ManiSkill/Isaac: GPU envs; LIBERO: worker processes")
    p.add_argument("--device", type=int, default=None, help="CUDA device for ManiSkill/Isaac, EGL device for LIBERO")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend_option", action="append", default=[], help="backend-specific key=value (json values), e.g. pen_length=0.05, max_dots=1500, compare_render=320x240")
    # appearance (shared by all backends, see burger/pipeline/appearance.py)
    p.add_argument("--appearance", default=None, help="appearance preset (maniskill | real) or a JSON Appearance dict; default: the backend's original look")
    p.add_argument("--appearance_random", choices=["none", "mild", "wide"], default="none", help="randomize the appearance around the preset: mild = lighting/texture details only, wide = large range")
    p.add_argument("--appearance_seed", type=int, default=0)
    p.add_argument("--scene", choices=["auto", "maniskill", "real"], default="auto",
                   help="canvas geometry (burger/pipeline/appearance.py SCENE_PRESETS): maniskill = 0.8 x 1.2 m canvas, real = the lab's 0.6 x 0.9 m board (table and real camera rescaled to it; robot and drawings unchanged); auto = real when the appearance preset is real")
    p.add_argument("--camera", choices=["auto", "maniskill", "real"], default="auto",
                   help="main-camera preset (burger/pipeline/appearance.py CAMERA_PRESETS): real = legacy forward-camera preset; auto = real when the appearance preset is real")
    p.add_argument("--no_auto_venv", action="store_true", help="do not re-exec in the backend's virtualenv")
    p.add_argument("--summary_json", default=None, help="write the run summary to this path as well")
    export_mod.add_writer_args(p)
    return p


def validate_args(args):
    """Fail before allocating a simulator, and preserve caller-relative paths."""
    args.svg_dir = str(Path(args.svg_dir).expanduser().resolve())
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    if args.summary_json:
        args.summary_json = str(Path(args.summary_json).expanduser().resolve())
    if not Path(args.svg_dir).is_dir():
        raise ValueError(f"SVG directory does not exist: {args.svg_dir}")
    if not args.object_name or args.object_name in {".", ".."} or Path(args.object_name).name != args.object_name or "\\" in args.object_name:
        raise ValueError("--object_name must be a single directory name")
    export_mod.writers.validate_dataset_name(args.object_name, argument="object_name")
    if not args.source_desc.strip():
        raise ValueError("--source_desc must be nonempty")
    if args.appearance_random != "none" and args.appearance is None:
        raise ValueError("--appearance_random requires an explicit --appearance preset")
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    if not formats or set(formats) - set(export_mod.FORMATS):
        raise ValueError(f"--format must select from {', '.join(export_mod.FORMATS)}")
    if len(formats) != len(set(formats)):
        raise ValueError("--format must not contain duplicates")
    if args.end_signal_length < 1:
        raise ValueError("--end_signal_length must be positive")
    if args.image_hw is not None and min(args.image_hw) < 1:
        raise ValueError("--image_hw must contain positive dimensions")
    if not math.isfinite(args.delta_gain) or args.delta_gain <= 0:
        raise ValueError("--delta_gain must be finite and positive")
    if args.lerobot_repo_name is not None:
        export_mod.writers.validate_dataset_name(args.lerobot_repo_name, argument="lerobot_repo_name")
    if args.parallel < 1 or min(args.camera_height, args.camera_width) < 1:
        raise ValueError("--parallel and camera dimensions must be positive")
    if args.num_samples == 0 or args.num_samples < -1 or args.start_idx < 0:
        raise ValueError("--num_samples must be positive or -1; --start_idx must be nonnegative")
    if args.end_idx is not None and args.end_idx <= args.start_idx:
        raise ValueError("--end_idx must be greater than --start_idx")
    if args.seed < 0 or args.appearance_seed < 0 or (args.device is not None and args.device < 0):
        raise ValueError("seeds and device indices must be nonnegative")
    for name in ("max_draw_dist", "end_lift_z", "svg_xy_noise"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == "max_draw_dist" and value == 0):
            raise ValueError(f"--{name} must be finite and {'positive' if name == 'max_draw_dist' else 'nonnegative'}")
    parse_kv(args.backend_option)


def maybe_reexec(args, backend_cls, argv):
    if args.no_auto_venv or backend_cls.available():
        return None
    py = backend_cls.venv_python()
    if not py.exists():
        setup = f"scripts/setup_{args.backend}.sh"
        raise SystemExit(f"backend {args.backend!r} is not importable here and {py} does not exist; run bash {setup}")
    # Override path arguments after the original argv so that cwd changes cannot
    # redirect inputs or outputs.
    cmd = [str(py), "-m", "burger.pipeline.generate", *argv, "--no_auto_venv",
           "--svg_dir", args.svg_dir, "--output_dir", args.output_dir]
    if args.summary_json:
        cmd += ["--summary_json", args.summary_json]
    print(f"[generate] backend {args.backend!r} not importable in {sys.executable}; re-running with {py}", flush=True)
    return subprocess.call(cmd, cwd=str(_REPO_ROOT))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    backend_cls = get_backend(args.backend)
    rc = maybe_reexec(args, backend_cls, argv)
    if rc is not None:
        sys.exit(rc)
    from burger.pipeline import svg_trajectory as traj

    language = args.source_desc
    svg_files = [str(f) for f in traj.list_svg_files(args.svg_dir, args.num_samples, args.start_idx, args.end_idx)]
    if not svg_files:
        parser.error("no SVG files matched the requested selection")
    for svg_file in svg_files:
        traj.read_svg_path_string(svg_file)
    cfg = GenerationConfig(
        svg_files=svg_files, language=language, object_name=args.object_name, max_draw_dist=args.max_draw_dist, end_lift_z=args.end_lift_z,
        svg_xy_noise=args.svg_xy_noise, svg_scale_mode=args.svg_scale_mode, rot_image=not args.no_rot_image, camera_height=args.camera_height,
        camera_width=args.camera_width, wrist_camera=not args.no_wrist_camera, parallel=args.parallel, device=args.device, seed=args.seed,
        backend_options=parse_kv(args.backend_option),
    )
    if args.appearance is not None:
        cfg.backend_options.setdefault("appearance", args.appearance)
    if args.appearance_random != "none":
        cfg.backend_options.setdefault("appearance_random", args.appearance_random)
        cfg.backend_options.setdefault("appearance", cfg.backend_options.get("appearance", "real"))
    if args.appearance_seed:
        cfg.backend_options.setdefault("appearance_seed", args.appearance_seed)
    camera = args.camera
    if camera == "auto":
        camera = "real" if str(cfg.backend_options.get("appearance", "")) == "real" else "maniskill"
    if camera != "maniskill":
        cfg.backend_options.setdefault("camera_preset", camera)
    scene = args.scene
    if scene == "auto":
        scene = "real" if str(cfg.backend_options.get("appearance", "")) == "real" else "maniskill"
    if scene != "maniskill":
        from burger.pipeline import appearance as _A
        cfg.backend_options.setdefault("canvas_size", list(_A.SCENE_PRESETS[scene]))
    out_dir = Path(args.output_dir) / args.object_name / args.backend
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical = out_dir / f"{args.object_name}.canonical.h5"
    if canonical.exists() and not args.overwrite:
        raise SystemExit(f"{canonical} exists; pass --overwrite")
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    print(f"[generate] backend={args.backend} formats={formats} episodes={len(svg_files)} parallel={args.parallel} camera={args.camera_height}x{args.camera_width} -> {out_dir}", flush=True)
    t0 = time.time()
    backend = backend_cls()
    from burger.data import canonical as C
    pending = canonical.with_name(f".{canonical.name}.{uuid.uuid4().hex}.pending")
    try:
        summary = backend.generate(cfg, pending, out_dir / "work", log=lambda m: print(m, flush=True))
        validated = C.validate_file(pending, require_painter=True)
        if validated["num_episodes"] != len(svg_files):
            raise ValueError(f"expected {len(svg_files)} episodes, got {validated['num_episodes']}")
        if summary["num_episodes"] != validated["num_episodes"] or summary["total_steps"] != validated["total_steps"]:
            raise ValueError("backend summary does not match its recorded episodes")
        pending.replace(canonical)
    finally:
        pending.unlink(missing_ok=True)
    summary["generation_wall_s"] = time.time() - t0
    summary["path"] = str(canonical)
    summary["config"] = cfg.to_json()
    summary["python"] = sys.executable
    t1 = time.time()
    try:
        exports = export_mod.export(canonical, formats, out_dir, args, args.object_name, log=lambda m: print(m, flush=True))
    except Exception as exc:
        summary.update(status="export_failed", export_error=str(exc), export_wall_s=time.time() - t1)
        write_summary(out_dir / "generation_summary.json", summary)
        if args.summary_json:
            write_summary(Path(args.summary_json), summary)
        raise
    summary["export_wall_s"] = time.time() - t1
    summary["exports"] = exports
    summary["status"] = "complete"
    write_summary(out_dir / "generation_summary.json", summary)
    if args.summary_json:
        write_summary(Path(args.summary_json), summary)
    print(f"[generate] done: {summary['num_episodes']} episodes / {summary['total_steps']} steps | backend {summary['backend_wall_s']:.1f}s "
          f"({summary['steps_per_s']:.1f} steps/s) | export {summary['export_wall_s']:.1f}s | {out_dir}", flush=True)


def write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        temporary.write_text(json.dumps(summary, indent=2, default=str))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
