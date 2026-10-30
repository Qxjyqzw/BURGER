import numpy as np
import pytest
import svgpathtools

from burger.pipeline.svg_trajectory import build_ee_waypoints, parse_svg_document, parse_svg_path_string, read_svg_path_string


def test_xml_attributes_namespaces_and_relative_path_origins(tmp_path):
    source = tmp_path / "drawing.svg"
    source.write_text("""<svg xmlns="http://www.w3.org/2000/svg">
      <path d='M 100 100 L 200 100' id='this-is-not-path-data'/>
      <g><path id='second' d='m 10 20 l 30 0'/></g>
    </svg>""")
    parsed = svgpathtools.parse_path(read_svg_path_string(source))
    assert parsed[0].start == 100 + 100j
    assert parsed[1].start == 10 + 20j
    assert parsed[1].end == 40 + 20j


@pytest.mark.parametrize("body", [
    '<g transform="translate(5)"><path d="M0 0L1 1"/></g>',
    '<path d="M0 0Q1 2 3 4"/>',
    '<path d="M0 0A1 1 0 0 0 2 2"/>',
    '<circle cx="0" cy="0" r="1"/>',
    '<use href="#shape"/>',
    '<path d="M0 0L1 1" style="display:none"/>',
    '<style>path { transform: translate(1px); }</style><path d="M0 0L1 1"/>',
    '<path id="M0 0L1 1"/>',
])
def test_unsupported_geometry_fails_clearly(body):
    with pytest.raises(ValueError):
        parse_svg_document(f'<svg>{body}</svg>')


def test_defs_are_not_drawn():
    content = '<svg><defs><path d="M100 100L200 200"/></defs><path d="M0 0L1 1"/></svg>'
    assert len(svgpathtools.parse_path(parse_svg_document(content))) == 1


def test_line_trajectory_retains_existing_scaling_and_spacing():
    path_data = "M 100 100 L 200 100 L 200 200"
    direct = build_ee_waypoints(path_data)
    xml = build_ee_waypoints(parse_svg_document(f'<svg><path d="{path_data}"/></svg>'))
    np.testing.assert_allclose(xml, direct)
    assert np.isfinite(xml).all()
    assert np.max(np.linalg.norm(np.diff(xml, axis=0), axis=1)) <= 0.008 + 1e-12


@pytest.mark.parametrize("data", ["", "M0 0", "M0 0C1 2 3 4 5 6"])
def test_raw_paths_have_same_validation(data):
    with pytest.raises(ValueError):
        parse_svg_path_string(data)


@pytest.mark.parametrize("spacing", [0, -1, float("nan")])
def test_invalid_spacing_is_rejected(spacing):
    with pytest.raises(ValueError, match="positive"):
        build_ee_waypoints("M0 0L100 100", max_draw_distance=spacing)
