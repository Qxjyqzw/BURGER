"""SVG -> end-effector waypoint construction (simulator independent).

This is a faithful, numpy-only port of the trajectory logic in
``burger/backends/maniskill/environment.py``
(``_parse_svg_path_string`` + ``_get_end_effector_trajectory``), so LIBERO/robosuite
and ManiSkill draw *exactly* the same waypoints from the same SVG.

Coordinate convention of the returned waypoints ("painter frame"):
  * expressed in the robot base frame used by ManiSkill's ``pd_ee_pose`` controller
    (x forward, y left, z up, origin at the robot base on the table surface);
  * the canvas top surface is at ``z == CANVAS_TOP_Z`` (0.02) and the drawing height is
    ``DRAW_Z`` (0.02); between strokes the pen is lifted to ``LIFT_Z`` (0.11);
  * the canvas centre is at ``(CANVAS_CENTER_X, 0)`` = (0.515, 0).

In ManiSkill the base sits at world (-0.615, 0, 0); in the LIBERO port the same
waypoints are re-anchored to the LIBERO robot base (see ``burger.backends.libero.environment``).
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    import svgpathtools
except ImportError as exc:  # pragma: no cover
    raise ImportError("svgpathtools not installed. Install with `pip install svgpathtools`") from exc

# --- constants shared with DrawSVGParallelEnv -----------------------------------------
SVG_IMAGE_BASELINE_PX = 1024.0   # fixed image-space baseline used when scaling SVG px -> m
SVG_SCALE_M = 0.45               # 1024 px map to 0.45 m
CANVAS_THICKNESS = 0.02          # canvas box thickness (m); its top is at z = 0.02
DOT_THICKNESS = 0.003            # ink dot thickness (m)
BRUSH_RADIUS = 0.004             # ink dot radius (m)
TOUCH_TOLERANCE_Z = 0.012        # extra tolerance above the canvas that still counts as touching
CANVAS_TOP_Z = CANVAS_THICKNESS
TOUCH_Z_THRESHOLD = CANVAS_THICKNESS + DOT_THICKNESS + TOUCH_TOLERANCE_Z  # 0.035
LOCAL_DRAW_Z = 0.01              # drawing height before the +0.01 shift
LOCAL_LIFT_Z = 0.1               # lift height before the +0.01 shift
Z_SHIFT = 0.01
DRAW_Z = LOCAL_DRAW_Z + Z_SHIFT  # 0.02  (== canvas top)
LIFT_Z = LOCAL_LIFT_Z + Z_SHIFT  # 0.11
CANVAS_CENTER_X = 0.515          # canvas centre x in the robot base frame
CANVAS_HALF_SIZE = (0.4, 0.6)    # canvas half extents (x, y) in m
CONNECT_THRESHOLD = 0.01         # strokes closer than this are drawn without lifting
DEFAULT_MAX_DRAW_DISTANCE = 0.008
DEFAULT_END_LIFT_Z = 0.085


def read_svg_path_string(svg_file: str | Path) -> str:
    """Read a line-path SVG, rejecting unsupported geometry instead of dropping it."""
    return parse_svg_document(Path(svg_file).read_text(encoding="utf-8"))


def _parse_line_path(path_data: str):
    try:
        parsed = svgpathtools.parse_path(path_data)
    except (ValueError, TypeError, IndexError, AssertionError) as exc:
        raise ValueError(f"invalid SVG path data: {exc}") from exc
    if not parsed:
        raise ValueError("SVG path has no drawable segments")
    if any(not isinstance(segment, svgpathtools.Line) for segment in parsed):
        raise ValueError("Only line segments are supported; flatten SVG curves/arcs to lines before generation")
    if not np.isfinite([[s.start.real, s.start.imag, s.end.real, s.end.imag] for s in parsed]).all():
        raise ValueError("SVG path coordinates must be finite")
    return parsed


def parse_svg_document(content: str) -> str:
    """Extract absolute line paths in document order using an XML parser.

    Each path is parsed separately, so a relative ``m`` starts at that path's own
    origin. The pipeline draws path centerlines in source coordinates; it does not
    render fills or apply viewport scaling. Transforms, clipping, CSS and non-path
    geometry require preprocessing and are rejected with an actionable error.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError(f"invalid SVG XML: {exc}") from exc
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise ValueError("expected an SVG root element")
    paths = []

    def visit(element, is_root=False):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in ("metadata", "title", "desc", "defs"):
            return
        if tag not in ("svg", "g", "path") or (tag == "svg" and not is_root):
            raise ValueError(f"Unsupported SVG element <{tag}>; convert drawing geometry to line paths")
        for key in ("transform", "clip-path", "mask", "filter"):
            if key in element.attrib:
                raise ValueError(f"Unsupported SVG {key}; bake it into line-path coordinates before generation")
        styles = {}
        for item in element.get("style", "").split(";"):
            if ":" in item:
                key, value = item.split(":", 1)
                styles[key.strip().lower()] = value.strip().lower()
        for key in ("transform", "clip-path", "mask", "filter", "d"):
            if key in styles:
                raise ValueError(f"Unsupported SVG style {key}; bake it into line-path coordinates")
        for key, hidden in (("display", "none"), ("visibility", "hidden")):
            if element.get(key, "").strip().lower() == hidden or styles.get(key) == hidden:
                raise ValueError("Hidden SVG geometry is unsupported; remove hidden paths before generation")
        if tag == "path":
            if not element.get("d", "").strip():
                raise ValueError("SVG path is missing a nonempty d attribute")
            paths.append(_parse_line_path(element.attrib["d"]).d())
        for child in element:
            visit(child)

    visit(root, is_root=True)
    if not paths:
        raise ValueError("No drawable <path d=...> found in SVG")
    return " ".join(paths)


def list_svg_files(folder: str | Path, num_samples: int = -1, start_idx: int = 0, end_idx: int | None = None) -> list[Path]:
    """Sorted ``*.svg`` selection with the same slicing semantics as ``load_svg_from_folder``."""
    files = sorted(Path(folder).glob("*.svg"))
    if not files:
        raise ValueError(f"No SVG files found in folder: {folder}")
    if end_idx is None:
        end_idx = len(files) if num_samples == -1 else start_idx + num_samples
    if start_idx < 0 or start_idx >= len(files):
        raise ValueError(f"start_idx {start_idx} is out of range [0, {len(files)})")
    end_idx = min(end_idx, len(files))
    if start_idx >= end_idx:
        raise ValueError(f"start_idx {start_idx} must be less than end_idx {end_idx}")
    return files[start_idx:end_idx]


SCALE_MODES = ("fixed", "max")


def parse_svg_path_string(svg_string: str, scale_mode: str = "fixed"):
    """Port of ``DrawSVGParallelEnv._parse_svg_path_string``.

    ``scale_mode``: ``"fixed"`` divides pixel coordinates by a 1024 px baseline (pbs_data
    default); ``"max"`` divides by the largest coordinate of the SVG (legacy main-branch
    behaviour that produced the ``pipeline_test_run_20260907`` reference data).

    Returns ``(trajectory_points (N+1,3), lines (N,2,3), discont_indices, is_continuous)``
    in the *local* SVG frame (metres, x mirrored, centred on the mean segment start).
    """
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {SCALE_MODES}, got {scale_mode!r}")
    parsed_svg = _parse_line_path(svg_string)
    lines = []
    for seg in parsed_svg:
        lines.append([[p.real, p.imag] for p in seg.bpoints()])
    lines = np.array(lines)  # (n, 2, 2) for Line segments
    if scale_mode == "max":
        scale = np.max(lines)
        if scale <= 0:
            raise ValueError("legacy max scaling requires a positive largest SVG coordinate; use fixed scaling")
        lines = (lines / scale) * SVG_SCALE_M
    else:
        lines = (lines / SVG_IMAGE_BASELINE_PX) * SVG_SCALE_M
    lines[..., 0] *= -1
    lines = np.concatenate([lines, np.ones((*lines.shape[:-1], 1)) * LOCAL_DRAW_Z], -1)
    center = lines[:, :1, :].mean(axis=0) * np.array([[1, 1, 0]])
    lines = lines - center
    is_continuous = parsed_svg.iscontinuous()
    discont_indices: list[int] = []
    if not is_continuous:
        disconts = lines[1:, 0] - lines[:-1, 1]
        discont_indices = list(np.nonzero(np.logical_or(disconts[:, 0], disconts[:, 1]))[0])
    trajectory_points = np.concatenate((lines[:1, 0], lines[:, 1, :]))
    return trajectory_points, lines, discont_indices, is_continuous


def _construct_stroke_points_with_lift(lines: np.ndarray, disconts, lift_z=LOCAL_LIFT_Z, connect_threshold=CONNECT_THRESHOLD):
    disconts = set(int(i) for i in disconts)
    points = []
    for i, seg in enumerate(lines):
        start, end = seg
        if i == 0 or (i - 1) in disconts:
            if i > 0:
                prev_end = lines[i - 1][1]
                dist = np.linalg.norm(start - prev_end)
                if dist < connect_threshold:
                    points.append([start[0], start[1], start[2]])
                else:
                    points.append([start[0], start[1], lift_z])
                    points.append([start[0], start[1], start[2]])
            else:
                points.append([start[0], start[1], lift_z])
                points.append([start[0], start[1], start[2]])
        points.append([end[0], end[1], end[2]])
        if i == len(lines) - 1:
            points.append([end[0], end[1], lift_z])
        elif i in disconts:
            next_start = lines[i + 1][0]
            dist = np.linalg.norm(end - next_start)
            if dist >= connect_threshold:
                points.append([end[0], end[1], lift_z])
    return np.array(points)


def densify_points(points_array: np.ndarray, max_distance: float) -> np.ndarray:
    """Insert linearly interpolated points so consecutive points are <= ``max_distance`` apart."""
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be positive and finite")
    if len(points_array) == 0:
        raise ValueError("cannot densify an empty trajectory")
    new_points = [points_array[0]]
    for i in range(1, len(points_array)):
        p1 = points_array[i - 1]
        p2 = points_array[i]
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        if dist > max_distance:
            num_insert = int(np.floor(dist / max_distance))
            for j in range(1, num_insert + 1):
                alpha = j / (num_insert + 1)
                new_points.append((1 - alpha) * p1 + alpha * p2)
        new_points.append(p2)
    return np.array(new_points)


def _rotate_xy(points: np.ndarray, rad: float) -> np.ndarray:
    rot = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
    return points @ rot.T


def build_ee_waypoints(
    svg_string: str,
    max_draw_distance: float = DEFAULT_MAX_DRAW_DISTANCE,
    end_lift_z: float = DEFAULT_END_LIFT_Z,
    rot_image: bool = True,
    xy_offset=(0.0, 0.0),
    scale_mode: str = "fixed",
) -> np.ndarray:
    """Port of ``DrawSVGParallelEnv._assign_svg_to_env`` + ``_get_end_effector_trajectory``.

    Args:
        svg_string: concatenated ``d`` strings of one SVG (see ``read_svg_path_string``).
        max_draw_distance: densification spacing (m). 0.008 reproduces the reference data.
        end_lift_z: local z of the final lift point for *continuous* single-path SVGs.
        rot_image: rotate the drawing by 270 deg (True, default of the collector) or 90 deg.
        xy_offset: deterministic (dx, dy) translation in the local frame (the collector's
            ``svg_xy_noise`` draws this uniformly from [-noise, noise]).

    Returns:
        (T, 3) float64 waypoints in the robot base frame; ``T`` equals the number of
        actions ManiSkill records for the episode.
    """
    trajectory_points, lines, discont_indices, is_continuous = parse_svg_path_string(svg_string, scale_mode)
    offset = np.array([xy_offset[0], xy_offset[1], 0.0])
    trajectory_points = trajectory_points + offset
    lines = lines + offset
    if not is_continuous:
        points_array = _construct_stroke_points_with_lift(lines, discont_indices)
    else:
        start = np.array([[trajectory_points[0][0], trajectory_points[0][1], LOCAL_LIFT_Z]])
        end = np.array([[trajectory_points[-1][0], trajectory_points[-1][1], end_lift_z]])
        points_array = np.concatenate([start, trajectory_points, end], axis=0)
    points_array = densify_points(points_array, max_distance=max_draw_distance)
    points_array = _rotate_xy(points_array, 3 * np.pi / 2 if rot_image else np.pi / 2)
    points_array[:, 0] += CANVAS_CENTER_X
    points_array[:, 2] += Z_SHIFT
    return points_array


def pen_down_mask(waypoints: np.ndarray, z_threshold: float = TOUCH_Z_THRESHOLD) -> np.ndarray:
    """Boolean mask of waypoints at which the pen is expected to touch the canvas."""
    return waypoints[:, 2] < z_threshold


def _main():
    ap = argparse.ArgumentParser(description="Build painter waypoints from an SVG and optionally compare with a ManiSkill H5.")
    ap.add_argument("--svg", required=True, help="SVG file")
    ap.add_argument("--max_draw_dist", type=float, default=DEFAULT_MAX_DRAW_DISTANCE)
    ap.add_argument("--end_lift_z", type=float, default=DEFAULT_END_LIFT_Z)
    ap.add_argument("--no_rot_image", action="store_true")
    ap.add_argument("--scale_mode", choices=SCALE_MODES, default="fixed")
    ap.add_argument("--compare_h5", default=None, help="ManiSkill trajectory .h5 to compare actions[:, :3] against")
    ap.add_argument("--traj", default="traj_0")
    ap.add_argument("--save_npy", default=None)
    args = ap.parse_args()
    wps = build_ee_waypoints(read_svg_path_string(args.svg), args.max_draw_dist, args.end_lift_z, rot_image=not args.no_rot_image, scale_mode=args.scale_mode)
    down = pen_down_mask(wps)
    print(f"waypoints: {wps.shape} | pen-down: {int(down.sum())} | x[{wps[:,0].min():.4f},{wps[:,0].max():.4f}] y[{wps[:,1].min():.4f},{wps[:,1].max():.4f}] z[{wps[:,2].min():.4f},{wps[:,2].max():.4f}]")
    if args.save_npy:
        np.save(args.save_npy, wps)
    if args.compare_h5:
        import h5py
        with h5py.File(args.compare_h5, "r") as f:
            acts = f[args.traj]["actions"][:]
        n = min(len(acts), len(wps))
        diff = np.abs(acts[:n, :3] - wps[:n].astype(np.float32))
        print(f"h5 actions: {acts.shape} | compared first {n} | max |action_xyz - waypoint| = {diff.max():.2e} m | rot part unique: {np.unique(acts[:, 3:].round(4), axis=0).tolist()}")
        if len(acts) != len(wps):
            print(f"NOTE: length mismatch (h5 {len(acts)} vs waypoints {len(wps)}); extra h5 steps are typically the recorded arm-raise.")


if __name__ == "__main__":
    _main()
