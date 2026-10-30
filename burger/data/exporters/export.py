#!/usr/bin/env python
"""Export a canonical episode file to LIBERO and/or LeRobot format.

  python -m burger.data.exporters.export --canonical out/house/libero/house.canonical.h5 --format libero,lerobot --out_dir out/house/libero
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from burger._paths import PROJECT_ROOT as _REPO_ROOT

from burger.data import canonical as C  # noqa: E402
from burger.data.exporters import writers  # noqa: E402

FORMATS = ("libero", "lerobot", "canonical")


def add_writer_args(p: argparse.ArgumentParser):
    p.add_argument("--image_hw", type=int, nargs=2, default=None, metavar=("H", "W"), help="resize images for the LIBERO file (default: native)")
    p.add_argument("--libero_actions", choices=writers.LIBERO_ACTION_MODES, default="auto",
                   help="LIBERO file actions: 'auto' (native for the LIBERO backend, OSC-style deltas derived from the absolute targets for ManiSkill), 'native', or 'libero_delta'")
    p.add_argument("--delta_gain", type=float, default=4.0, help="gain used when deriving LIBERO deltas from absolute targets")
    p.add_argument("--end_signal_length", type=int, default=5, help="LeRobot: number of final steps with end signal = 1")
    p.add_argument("--lerobot_repo_name", default=None, help="LeRobot dataset folder name (default: <object>-<backend>-<timestamp>)")
    p.add_argument("--libero_compression", default=None, help="HDF5 compression for LIBERO image datasets (default none, like the official files; e.g. lzf, gzip)")
    p.add_argument("--overwrite", action="store_true")


def export(canonical: Path, formats: list, out_dir: Path, args, object_name: str | None = None, log=print) -> dict:
    import time
    # Resolve against the caller's directory before delegating with cwd=_REPO_ROOT.
    canonical = Path(canonical).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if not formats or any(fmt not in FORMATS for fmt in formats):
        raise ValueError(f"formats must be a nonempty selection from {FORMATS}")
    if len(formats) != len(set(formats)):
        raise ValueError("formats must not contain duplicates")
    object_name = Path(canonical).stem.replace(".canonical", "") if object_name is None else object_name
    writers.validate_dataset_name(object_name, "object_name")
    if args.lerobot_repo_name is not None:
        writers.validate_dataset_name(args.lerobot_repo_name, "lerobot_repo_name")
    if isinstance(args.end_signal_length, bool) or not isinstance(args.end_signal_length, int) or args.end_signal_length < 1:
        raise ValueError("end_signal_length must be a positive integer")
    if isinstance(args.delta_gain, bool) or not isinstance(args.delta_gain, (int, float)) or not math.isfinite(args.delta_gain) or args.delta_gain <= 0:
        raise ValueError("delta_gain must be positive and finite")
    if args.image_hw is not None and (len(args.image_hw) != 2 or min(args.image_hw) < 1):
        raise ValueError("image_hw must contain positive height and width")
    results = {}
    info = C.read_info(canonical)
    if "lerobot" in formats:
        # Validate the default-derived name before any other requested format is written.
        repo = args.lerobot_repo_name or f"{object_name}-{info.backend}-{time.strftime('%Y%m%d_%H%M%S')}"
        writers.validate_dataset_name(repo, "lerobot_repo_name")
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        if fmt == "canonical":
            C.validate_file(canonical)
            results[fmt] = dict(format="canonical", path=str(canonical))
            continue
        if fmt == "libero":
            path = Path(out_dir) / f"{object_name}_demo.hdf5"
            results[fmt] = writers.write_libero_hdf5(canonical, path, image_hw=tuple(args.image_hw) if args.image_hw else None,
                                                     action_mode=args.libero_actions, delta_gain=args.delta_gain, overwrite=args.overwrite,
                                                     image_compression=args.libero_compression)
        elif fmt == "lerobot":
            try:
                import lerobot  # noqa: F401
            except ImportError:
                # Use an independent export environment; no simulator installation is required.
                import subprocess
                py = Path(os.environ.get("PAINTER_LEROBOT_PYTHON", str(_REPO_ROOT / ".venv-lerobot-export" / "bin" / "python"))).expanduser().absolute()
                if not py.is_file():
                    raise RuntimeError("LeRobot is unavailable; run scripts/setup_export.sh or set PAINTER_LEROBOT_PYTHON")
                if os.environ.get("PAINTER_EXPORT_DELEGATED") == "1" or py == Path(sys.executable).absolute():
                    raise RuntimeError(f"LeRobot is not installed in the export interpreter: {py}")
                cmd = [str(py), "-m", "burger.data.exporters.export", "--canonical", str(canonical), "--format", "lerobot", "--out_dir", str(out_dir),
                       "--end_signal_length", str(args.end_signal_length), "--lerobot_repo_name", repo, "--object_name", object_name] + (["--overwrite"] if args.overwrite else [])
                log(f"[export] lerobot not importable here; delegating to {py}")
                subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT), env=dict(os.environ, PAINTER_EXPORT_DELEGATED="1"))
                results[fmt] = json.loads((Path(out_dir) / repo / "painter_export.json").read_text())
                continue
            results[fmt] = writers.write_lerobot(canonical, out_dir, repo, end_signal_length=args.end_signal_length, overwrite=args.overwrite)
        else:
            raise ValueError(f"unknown format {fmt}")
        log(f"[export] {fmt}: {results[fmt]}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export canonical BURGER episodes to LIBERO / LeRobot")
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--format", default="libero", help="comma separated: libero, lerobot")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--object_name", default=None)
    add_writer_args(ap)
    args = ap.parse_args(argv)
    res = export(Path(args.canonical), [f.strip() for f in args.format.split(",") if f.strip()], Path(args.out_dir), args, args.object_name)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
