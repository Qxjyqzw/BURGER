import argparse
import os

import cv2
import numpy as np
from PIL import Image
from skimage.io import imsave
from skimage.morphology import skeletonize
from skimage.transform import resize


def remove_filled_black_areas(img: np.ndarray) -> np.ndarray:
    """
    Remove small filled black circular-like regions by drawing a small white dot
    at the center of each detected region.

    Parameters:
        img (np.ndarray): Binary image with black regions on white background (0 for black, 255 for white)

    Returns:
        np.ndarray: Modified image with white dots drawn over selected black regions
    """
    img = img.copy().astype(np.uint8)

    # Find all contours in the image
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Compute the area of the contour
        area = cv2.contourArea(cnt)
        if area < 10 or area > 2000:
            continue  # Ignore too small or too large regions

        # Compute the perimeter (arc length) of the contour
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # Calculate circularity to filter round-like shapes
        circularity = 4 * np.pi * area / (perimeter**2)
        if circularity > 0.6:
            # Compute the center (centroid) of the contour
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Determine the desired fill area as a ratio of the original region
            fill_ratio = 0.1
            fill_area = area * fill_ratio

            # Compute radius from the desired fill area
            radius = int(np.sqrt(fill_area / np.pi))

            # Draw a white circle at the region center to remove the dark region
            cv2.circle(
                img, (cx, cy), radius, 0, -1
            )  # 0 means white (in inverted image)

    return img


def load_and_skeletonize(image_path, target_size=512, do_resize=False, invert=False):
    """
    Load a binary (black and white) image, resize it to target_size (max side),
    and extract its skeleton.
    """
    with Image.open(image_path) as source:
        if source.mode.startswith("I;16") or source.mode == "I":
            image = np.asarray(source, dtype=np.float64) / 65535.0
        else:
            # Transparent pixels are paper; handle both LA and RGBA inputs.
            rgba = source.convert("RGBA")
            background = 0 if invert else 255
            paper = Image.new("RGBA", rgba.size, (background, background, background, 255))
            image = np.asarray(Image.alpha_composite(paper, rgba).convert("L"), dtype=np.float64) / 255.0
    if invert:
        image = 1.0 - image

    # Resize image before skeletonization
    if do_resize:
        H, W = image.shape
        scale = target_size / max(H, W)
        new_H = int(round(H * scale))
        new_W = int(round(W * scale))

        resized_image = resize(
            image, (new_H, new_W), order=1, preserve_range=True, anti_aliasing=False
        )

        # Pad to square
        pad_H = target_size - new_H
        pad_W = target_size - new_W
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left

        binary_image = np.pad(
            resized_image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=1.0,
        )
    else:
        binary_image = image

    # Convert to uint8 binary image: 0 for black, 255 for white
    binary_uint8 = binary_image < 0.5
    cleaned = remove_filled_black_areas(binary_uint8)  # Remove solid black regions
    skeleton = skeletonize(cleaned)
    if not skeleton.any():
        raise ValueError(f"No foreground strokes found in {image_path}; use dark lines on a light background or --invert.")

    return skeleton


def process_folder(input_folder, output_folder, invert=False):
    """
    Process all PNG images in the input_folder,
    generate skeleton images, and save them in the output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    filenames = sorted(name for name in os.listdir(input_folder) if name.lower().endswith(".png"))
    if not filenames:
        raise ValueError(f"No PNG inputs in {input_folder}")
    for filename in filenames:
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            skeleton = load_and_skeletonize(input_path, invert=invert)
            # Save skeleton image as uint8 (0 or 255)
            skeleton_uint8 = (skeleton.astype(np.uint8)) * 255
            output_path = os.path.join(output_folder, filename)
            imsave(output_path, skeleton_uint8)
            print(f"Saved skeleton: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch skeletonize all PNG images in a folder."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing input PNG images.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder to save skeletonized images.",
    )
    parser.add_argument("--invert", action="store_true", help="Input has light strokes on a dark background.")
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, invert=args.invert)
