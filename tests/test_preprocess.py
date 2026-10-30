"""Regressions for lost grayscale strokes, alpha handling, and path direction."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PNG = ROOT / "burger/preprocess"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, PNG / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def skeletonizer():
    pytest.importorskip("cv2")
    pytest.importorskip("skimage")
    return load("skeletonizer", "skeletonize.py")


def test_gray_rgb_alpha_and_16bit_preserve_same_strokes(tmp_path, skeletonizer):
    image = np.full((64, 64), 255, dtype=np.uint8)
    image[30:33, 10:54] = 40
    reference = None
    for mode in ("L", "RGB", "LA", "RGBA", "I;16"):
        source = Image.fromarray(image).convert(mode) if mode != "I;16" else Image.fromarray(image.astype(np.uint16) * 257)
        path = tmp_path / f"stroke-{mode.replace(';', '-')}.png"
        source.save(path)
        actual = skeletonizer.load_and_skeletonize(path)
        assert actual.sum() > 30
        if reference is None:
            reference = actual
        np.testing.assert_array_equal(actual, reference)


def test_invert_is_explicit_and_empty_is_rejected(tmp_path, skeletonizer):
    image = np.zeros((64, 64), dtype=np.uint8)
    image[30:33, 10:54] = 255
    path = tmp_path / "light-stroke.png"
    Image.fromarray(image).save(path)
    actual = skeletonizer.load_and_skeletonize(path, invert=True)
    assert actual[:, :8].sum() == 0
    assert actual.sum() > 30
    Image.new("L", (64, 64), 255).save(path)
    with pytest.raises(ValueError, match="No foreground"):
        skeletonizer.load_and_skeletonize(path)


def test_transparent_white_stroke_with_invert(tmp_path, skeletonizer):
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[30:33, 10:54] = 255
    path = tmp_path / "transparent-stroke.png"
    Image.fromarray(rgba).save(path)
    actual = skeletonizer.load_and_skeletonize(path, invert=True)
    assert actual.sum() > 30
    assert actual[:, :8].sum() == 0


def test_duplicate_paths_leave_one_representative():
    pytest.importorskip("networkx")
    module = load("extract_paths", "extract_paths.py")
    path = [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert module.remove_redundant_paths([path, path.copy()]) == [path]


def test_merges_collinear_paths_without_losing_gap_endpoints():
    pytest.importorskip("networkx")
    module = load("extract_paths", "extract_paths.py")
    left, right = [(0, 0), (10, 0)], [(12, 0), (25, 0)]
    for a, b in ((left, right), (left[::-1], right), (left, right[::-1])):
        merged = module.merge_paths([a, b], dist_thresh=3)
        assert len(merged) == 1
        assert set(merged[0]) == set(left + right)


def test_rejects_hairpin_merge():
    pytest.importorskip("networkx")
    module = load("extract_paths", "extract_paths.py")
    assert len(module.merge_paths([[(0, 0), (10, 0)], [(12, 0), (-10, 0)]], dist_thresh=3)) == 2


def test_short_stroke_survives_sampling():
    pytest.importorskip("networkx")
    module = load("extract_paths", "extract_paths.py")
    skeleton = np.zeros((30, 30), dtype=bool)
    skeleton[10, 5:19] = True
    paths = module.extract_paths_from_skeleton(skeleton, spacing=5, smooth_circular_paths=False)
    assert paths
    assert len(paths[0]) >= 4
    assert np.max(np.linalg.norm(np.diff(paths[0], axis=0), axis=1)) <= 5.001


def test_no_empty_svg_written(tmp_path, monkeypatch):
    pytest.importorskip("networkx")
    pytest.importorskip("skimage")
    monkeypatch.syspath_prepend(str(PNG))
    converter = load("svg_converter", "svg_convert.py")
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    Image.new("L", (20, 20), 0).save(source / "empty.png")
    with pytest.raises(ValueError, match="No drawable paths"):
        converter.process_skeleton_folder(source, output)
    assert list(output.iterdir()) == []
