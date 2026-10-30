import argparse
import os
import math

from burger.preprocess.extract_paths import extract_paths_from_skeleton
from skimage.io import imread


def path_to_svg_d(points):
    """
    Convert a list of (x, y) points to an SVG path string.
    """
    if not points:
        return ""
    d = f"M {points[0][0]:.2f},{points[0][1]:.2f}"
    for x, y in points[1:]:
        d += f" L {x:.2f},{y:.2f}"
    return d


def paths_to_svg(paths, width, height):
    """
    Convert a list of paths into a full SVG string with multiple <path> tags.

    Each path is drawn in a different color for visualization clarity.
    """
    svg_header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
    )

    svg_body = ""
    color_palette = ["black", "gray", "blue", "red", "green", "orange"]
    color_index = 0

    for i, path in enumerate(paths):
        if not path:
            continue
        d_str = path_to_svg_d(path)
        if not d_str.strip():
            continue
        color = color_palette[color_index % len(color_palette)]
        svg_body += (
            f'  <path d="{d_str}" stroke="{color}" fill="none" '
            f'stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>\n'
        )
        color_index += 1

    svg_footer = "</svg>"
    return svg_header + svg_body + svg_footer


def process_skeleton_folder(input_folder, output_folder, spacing=5, smooth_circular_paths=True):
    """
    Process all skeleton PNG files in input_folder, extract paths,
    and write SVG files to output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    filenames = sorted(name for name in os.listdir(input_folder) if name.lower().endswith(".png"))
    if not filenames:
        raise ValueError(f"No PNG inputs in {input_folder}")

    for filename in filenames:
        if not filename.lower().endswith(".png"):
            continue

        input_path = os.path.join(input_folder, filename)
        skeleton = imread(input_path) > 0

        if skeleton.ndim != 2:
            raise ValueError(f"Expected a grayscale skeleton: {input_path}")
        paths = extract_paths_from_skeleton(skeleton, spacing=spacing, smooth_circular_paths=smooth_circular_paths)
        if not paths or not any(len(path) >= 2 for path in paths):
            raise ValueError(f"No drawable paths in {input_path}; no SVG written.")
        h, w = skeleton.shape
        svg_string = paths_to_svg(paths, width=w, height=h)

        output_name = os.path.splitext(filename)[0] + ".svg"
        output_path = os.path.join(output_folder, output_name)
        with open(output_path, "w") as f:
            f.write(svg_string)

        print(
            f"[✓] Saved SVG: {output_path} — {len(paths)} paths, "
            f"{sum(len(p) for p in paths)} total points."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert skeleton PNGs to SVG paths.")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing skeleton .png images.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder to save .svg output files.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=5.0,
        help="Spacing between sampled points along paths.",
    )
    parser.add_argument("--no_smooth_circular_paths", action="store_true",
                        help="Disable the historical circular-path smoothing heuristic.")
    args = parser.parse_args()

    process_skeleton_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        spacing=args.spacing,
        smooth_circular_paths=not args.no_smooth_circular_paths,
    )
