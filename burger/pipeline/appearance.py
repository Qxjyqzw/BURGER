"""Backend-independent description of the scene *appearance* (colours, textures, lights) with presets and randomizers.

The geometry of the drawing scene (robot, table footprint, canvas, camera) is fixed by ManiSkill's
``DrawSVGParallel-v1``; this module only describes how the surfaces and lights look, so that the ManiSkill, LIBERO
and Isaac backends can render the same appearance and randomize it the same way.

Presets
    ``maniskill``  the original ManiSkill look (wood table, grey grid floor, white canvas, red ink)
    ``real``       matched to the lab photos 678-681.JPG (green mat on a 1.40 x 1.64 m table, white board with a dark rim,
                   black surround, soft overhead light with a light shadow right under the arm, red marker, blue status
                   LEDs on the arm). The matching *camera* is a separate run-level preset (``CAMERA_PRESETS["real"]``).
Randomization modes (``sample_appearance``)
    ``none``  the preset itself
    ``mild``  lighting and texture *details* only (intensity, direction jitter, colour temperature, texture pattern and
              scale, small colour jitter): every sample still looks like the real setting
    ``wide``  large range: arbitrary table / floor / canvas colours, any texture family, lights from anywhere in the
              upper hemisphere with strong intensity and colour changes, ink colour

All colours are sRGB in [0, 1]. Light intensities are relative to the ManiSkill rasteriser (a directional light of
intensity 1 in ManiSkill); each backend converts them into its own units (see the backends).
"""
from __future__ import annotations

import colorsys
import dataclasses
import hashlib
import json
import math
import os  # [pd_ee run] atomic texture writes
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets"
TEXTURE_CACHE = ASSET_DIR / "textures_generated"   # procedural textures (git-ignored)
MANISKILL_WOOD = ASSET_DIR / "textures" / "maniskill_table_wood.png"
MANISKILL_GRID = ASSET_DIR / "textures" / "maniskill_grid.png"

TEXTURE_FAMILIES = ("plain", "noise", "grid", "stripes", "checker", "wood", "maniskill_grid")


@dataclasses.dataclass
class Appearance:
    """One concrete appearance. ``*_texture`` is a family name from ``TEXTURE_FAMILIES`` or a PNG path; the texture is
    blended over ``*_color`` with ``*_texture_strength`` (0 = flat colour) and repeated ``*_texture_scale`` times per m."""

    name: str = "maniskill"
    # surfaces
    table_color: tuple = (0.55, 0.36, 0.22)
    table_texture: str | None = "wood"
    table_texture_scale: float = 1.0
    table_texture_repeats: tuple | None = (1.0, 3.94)   # explicit (x, y) repeats over the table top; None -> scale * size
    table_texture_strength: float = 1.0
    table_roughness: float = 0.7
    floor_color: tuple = (0.48, 0.49, 0.52)
    floor_texture: str | None = "maniskill_grid"
    floor_texture_scale: float = 0.25
    floor_texture_strength: float = 1.0
    canvas_color: tuple = (1.0, 1.0, 1.0)
    canvas_rim_color: tuple = (0.15, 0.16, 0.16)
    canvas_rim_width: float = 0.0             # m; 0 = no rim
    ink_color: tuple = (0.8, 0.2, 0.2)
    background_color: tuple = (0.48, 0.49, 0.52)   # walls / sky where a backend shows any
    # lights (ManiSkill units)
    ambient: float = 0.3
    key_dir: tuple = (1.0, 1.0, -1.0)         # direction the light travels (world frame: x toward the robot's front, z up)
    key_intensity: float = 1.0
    key_color: tuple = (1.0, 1.0, 1.0)
    fill_dir: tuple = (0.0, 0.0, -1.0)
    fill_intensity: float = 1.0
    fill_color: tuple = (1.0, 1.0, 1.0)
    key2_dir: tuple | None = None             # optional second shadow-casting light (e.g. the lamp on the other side of the arm)
    key2_intensity: float = 0.0
    key2_color: tuple = (1.0, 1.0, 1.0)
    light_angle_deg: float = 0.0              # angular size of the key lights (soft shadows); honoured by Isaac Sim's RTX lights only
    shadows: bool = False
    exposure: float = 1.0                      # global multiplier applied by the backends
    # geometry-ish extras
    table_extent: tuple | None = None          # (x_min, x_max, y_min, y_max) of the table top in the ManiSkill world frame; None = ManiSkill's table
    table_apron_color: tuple | None = (0.02, 0.02, 0.02)   # dark band below the 1 cm table top (the real table's edge); None = table colour
    led_color: tuple | None = None             # emissive colour of the robot's status LEDs (``LED_MARKERS``); None = no LEDs
    led_intensity: float = 1.0
    pen_style: str | None = None               # "marker": black finger holder + 18 mm marker body (``MARKER_PARTS``); None = the backend's thin stick
    photo_scaled: bool = False                 # table_extent was measured on the lab photos in board units (1.2 m canvas) and scales with the simulated canvas

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Appearance":
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


PRESETS: dict[str, Appearance] = {
    "maniskill": Appearance(name="maniskill"),
    # measured on the 320x240 lab frames 678-681.JPG (the dataset's forward camera): green mat rgb (27, 98, 87), white board
    # 180 (195 lit / 150 in the arm's shadow), 1.2 cm dark rim, surround 5-7, red marker, blue LEDs on the arm; the table
    # top spans x[-1.08, 0.32] y[-0.82, 0.82] in the ManiSkill world frame (see docs/PIPELINE.md)
    "real": Appearance(
        name="real",
        table_color=(0.11, 0.53, 0.47), table_texture="noise", table_texture_scale=2.0, table_texture_repeats=None, table_texture_strength=0.08, table_roughness=0.9,
        floor_color=(0.004, 0.005, 0.004), floor_texture="plain", floor_texture_scale=1.0, floor_texture_strength=0.0,
        canvas_color=(0.92, 0.93, 0.92), canvas_rim_color=(0.12, 0.13, 0.13), canvas_rim_width=0.01,
        ink_color=(0.75, 0.08, 0.10), background_color=(0.0, 0.0, 0.0),
        # diffuse room light with a light, soft shadow right under the arm: strong ambient + fill and two weak, large key lights,
        # one behind and one in front of the robot, whose shadows overlap (soft penumbra where the renderer supports area lights)
        ambient=0.6, key_dir=(0.22, 0.05, -0.97), key_intensity=0.17, key_color=(1.0, 0.99, 0.97),
        key2_dir=(-0.30, -0.05, -0.95), key2_intensity=0.17, key2_color=(1.0, 0.99, 0.97), light_angle_deg=20.0,
        fill_dir=(0.0, 0.0, -1.0), fill_intensity=0.3, fill_color=(1.0, 1.0, 1.0), shadows=True,
        table_extent=(-1.084, 0.315, -0.82, 0.82), photo_scaled=True, led_color=(0.25, 0.55, 1.0), led_intensity=1.0, pen_style="marker",
    ),
}

# Canvas (board) geometry. ManiSkill's task canvas is 0.8 x 1.2 m; the lab's board is 0.6 x 0.9 m. The photo calibration
# (camera preset "real", REAL_TABLE_EXTENT) was done in *board units* of the 1.2 m canvas and is rescaled about the canvas
# centre to the canvas that is actually simulated (scene preset "real" -> factor 0.75, robot and drawings unchanged).
CANVAS_CENTER = (-0.1, 0.0, 0.02)          # canvas top centre in the ManiSkill world frame
DEFAULT_CANVAS_SIZE = (0.8, 1.2)           # (x depth, y width)
SCENE_PRESETS = {"maniskill": DEFAULT_CANVAS_SIZE, "real": (0.6, 0.9)}
PHOTO_CANVAS_WIDTH = 1.2


def resolve_canvas_size(options: dict | None) -> tuple:
    """``canvas_size`` option ((x, y) in m) or the ``scene`` preset name; default ManiSkill's canvas."""
    options = options or {}
    c = options.get("canvas_size")
    if c is None:
        scene = options.get("scene") or "maniskill"
        if scene not in SCENE_PRESETS:
            raise ValueError(f"unknown scene preset {scene!r}; choose from {list(SCENE_PRESETS)}")
        c = SCENE_PRESETS[scene]
    return (float(c[0]), float(c[1]))


def scene_scale(canvas_size) -> float:
    return float(canvas_size[1]) / PHOTO_CANVAS_WIDTH


def scale_about_canvas(p, s: float) -> tuple:
    """Scale a world point about the canvas top centre (photo-calibrated lengths -> the simulated canvas size)."""
    c = CANVAS_CENTER
    return tuple(float(c[i] + s * (float(p[i]) - c[i])) for i in range(len(p)))

# ManiSkill's TableSceneBuilder table top (world AABB) and the real table measured on the lab frames
MANISKILL_TABLE_EXTENT = (-0.7402, 0.4687, -1.2149, 1.2031)
REAL_TABLE_EXTENT = (-1.084, 0.315, -0.82, 0.82)
MANISKILL_ROBOT_BASE_X = -0.615
# emissive status-LED markers: (URDF link name, offset in the link frame [m], shape, size) with shape "sphere" (size = radius),
# "box" (half extents) or "capsule" / "cylinder" ((radius, half_length) along the link's z axis). Measured on the lab frames:
# two vertical strips on both sides of the robot base, right above the table on the front corners of the base block
# (z 2-9 cm, alternately hidden by the arm) and one dot at the centre of the dark cover on top of the wrist cylinder
# (joint-7 axis, ~8 cm above the flange). The same link frames exist in ManiSkill's panda_stick URDF, robosuite's Panda
# MJCF (robot0_<link>) and the Isaac URDF import.
LED_MARKERS = (
    ("panda_link0", (0.06, 0.082, 0.055), "capsule", (0.006, 0.035)),
    ("panda_link0", (0.06, -0.082, 0.055), "capsule", (0.006, 0.035)),
    ("panda_link7", (0.0, 0.0, -0.092), "sphere", 0.01),
)
# "marker" pen style, in the panda_hand frame (z towards the pen tip, TCP at z = 0.15, fingers open along y): black finger
# holder + 18 mm marker body; the backends' thin stick keeps providing the last few mm of the tip. (shape, centre, size) with
# size = half extents for boxes and (radius, half_length along z) for cylinders.
MARKER_PARTS = (
    ("box", (0.0, 0.0, 0.083), (0.012, 0.03, 0.025)),
    ("cylinder", (0.0, 0.0, 0.122), (0.009, 0.024)),
)
MARKER_COLOR = (0.03, 0.03, 0.035)


def table_extent_of(a: Appearance, canvas_size=None) -> tuple:
    """(x_min, x_max, y_min, y_max) of the table top in world metres; photo-measured extents follow the canvas size."""
    if a.table_extent is None:
        return tuple(float(v) for v in MANISKILL_TABLE_EXTENT)
    e = [float(v) for v in a.table_extent]
    if getattr(a, "photo_scaled", False) and canvas_size is not None:
        s = scene_scale(canvas_size); cx, cy = CANVAS_CENTER[0], CANVAS_CENTER[1]
        e = [cx + s * (e[0] - cx), cx + s * (e[1] - cx), cy + s * (e[2] - cy), cy + s * (e[3] - cy)]
    return tuple(e)


def table_size_of(a: Appearance, canvas_size=None) -> tuple:
    """(depth along x, width along y) of the table top."""
    e = table_extent_of(a, canvas_size)
    return (e[1] - e[0], e[3] - e[2])


def table_layout_kwargs(a: Appearance, canvas_size=None) -> dict:
    """LIBERO ``compute_scene_layout`` table parameters for the appearance's table extent (relative to the robot base)."""
    e = table_extent_of(a, canvas_size)
    return dict(table_length=e[1] - e[0], table_width=e[3] - e[2], table_x_behind_base=MANISKILL_ROBOT_BASE_X - e[0], table_y_offset=(e[2] + e[3]) / 2.0)


# ------------------------------------------------------------------------------------------
# camera presets (run-level: the viewpoint is the same for all envs / episodes of a run)
# ------------------------------------------------------------------------------------------
@dataclasses.dataclass
class CameraSpec:
    """Pinhole camera in the ManiSkill world frame with square pixels and intrinsics *relative to the frame width*
    (``fx = fy = f_rel * width``, ``cx = cx_rel * width``, ``cy = cy_rel * height``), so that any resolution keeps the
    same horizontal coverage; 4:3 frames reproduce the calibrated photo exactly."""

    name: str
    eye: tuple
    target: tuple
    f_rel: float
    cx_rel: float = 0.5
    cy_rel: float = 0.5
    photo_scaled: bool = False   # eye / target were calibrated in board units (1.2 m canvas) and scale with the simulated canvas

    def intrinsics(self, width: int, height: int) -> tuple:
        f = self.f_rel * width
        return f, f, self.cx_rel * width, self.cy_rel * height

    def fovy_rad(self, width: int, height: int) -> float:
        return 2.0 * math.atan(height / (2.0 * self.f_rel * width))

    def fovx_rad(self, width: int) -> float:
        return 2.0 * math.atan(1.0 / (2.0 * self.f_rel))

    def render_geometry(self, width: int) -> tuple:
        """For renderers without a principal-point offset: render a centred frame of ``render_width`` columns and keep
        columns ``[x0, x0 + width)``. Returns (render_width, x0)."""
        if abs(self.cx_rel - 0.5) < 1e-6:
            return int(width), 0
        render_width = int(round(2.0 * max(self.cx_rel, 1.0 - self.cx_rel) * width))
        x0 = int(round(render_width / 2.0 - self.cx_rel * width))
        return render_width, max(0, x0)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


CAMERA_PRESETS: dict[str, CameraSpec] = {
    # ManiSkill base_camera ("front" view): fovy 1.2 rad at 4:3, centred principal point
    "maniskill": CameraSpec(name="maniskill", eye=(0.3, 0.0, 0.8), target=(0.0, 0.0, 0.1), f_rel=0.75 / math.tan(0.6) / 2.0),
    # calibrated on 678-681.JPG (320x240, board corners reprojected to < 0.1 px): the frames are an off-centre crop of the
    # webcam image (principal point at 65.3 % of the width), camera 1.6 m from the board centre, 31 deg elevation
    "real": CameraSpec(name="real", eye=(1.155, 0.202, 0.993), target=(-0.424, 0.174, 0.02), f_rel=1.01084, cx_rel=0.65294, cy_rel=0.5, photo_scaled=True),
}


def resolve_camera(options: dict) -> CameraSpec | None:
    """``camera_preset`` option: None / "maniskill" -> None (the backend's native ManiSkill camera), a preset name, or a JSON
    / dict ``CameraSpec``."""
    c = options.get("camera_preset")
    if c is None or c in ("", "none", "maniskill"):
        return None
    if isinstance(c, CameraSpec):
        spec = c
    elif isinstance(c, dict):
        spec = CameraSpec(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()})
    elif isinstance(c, str) and c.strip().startswith("{"):
        spec = resolve_camera(dict(camera_preset=json.loads(c)))
    elif c in CAMERA_PRESETS:
        spec = CAMERA_PRESETS[c]
    else:
        raise ValueError(f"unknown camera preset {c!r}; choose from {list(CAMERA_PRESETS)}")
    if spec.photo_scaled and (options.get("canvas_size") is not None or options.get("scene")):
        s = scene_scale(resolve_canvas_size(options))
        if abs(s - 1.0) > 1e-9:
            spec = dataclasses.replace(spec, eye=scale_about_canvas(spec.eye, s), target=scale_about_canvas(spec.target, s), photo_scaled=False)
    return spec


# ------------------------------------------------------------------------------------------
# procedural textures
# ------------------------------------------------------------------------------------------
def _hash(*parts) -> str:
    return hashlib.md5(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def _save_png_atomic(img, out: Path) -> None:
    """[pd_ee run] write ``img`` to ``out`` via a per-process temp file + rename, so concurrent workers that share
    TEXTURE_CACHE never open a half-written PNG (PIL.UnidentifiedImageError seen with several processes)."""
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp.png")
    img.save(tmp)
    os.replace(tmp, out)


def texture_path(family: str | None, color, strength: float, seed: int = 0, size: int = 512, tile=None) -> Path | None:
    """Return a PNG for ``family`` blended over ``color`` (written to ``TEXTURE_CACHE`` on first use).

    ``plain``: flat colour; ``noise``: low-frequency colour noise; ``grid``: light grid lines on the colour; ``stripes``:
    plank-like stripes; ``checker``: two-tone checkerboard; ``wood``: ManiSkill's wood photo tinted toward ``color``;
    ``maniskill_grid``: ManiSkill's ground texture tinted toward ``color``. Returns None for ``None``/``plain`` with
    strength 0 (the backends then use a flat colour).
    """
    if family is None or (family == "plain" and strength <= 0):
        return None
    if family not in TEXTURE_FAMILIES:
        p = Path(family)
        if p.exists():
            return p
        raise ValueError(f"unknown texture family {family!r}")
    from PIL import Image

    TEXTURE_CACHE.mkdir(parents=True, exist_ok=True)
    tile = None if tile is None else (float(tile[0]), float(tile[1]))
    key = _hash(family, [round(float(c), 4) for c in color], round(float(strength), 4), int(seed), size, tile)
    out = TEXTURE_CACHE / f"{family}_{key}.png"
    if out.exists():
        return out
    if tile is not None:
        # bake ``tile`` = (repeats along u, repeats along v) of the base pattern into one image (for surfaces whose UVs
        # span 0..1 once, e.g. box tops in MuJoCo / SAPIEN / USD cubes); fractional repeats are cropped
        base_png = texture_path(family, color, strength, seed, size=256)
        base = np.asarray(Image.open(base_png).convert("RGB"))
        nu, nv = max(tile[0], 1e-3), max(tile[1], 1e-3)
        reps_u, reps_v = int(math.ceil(nu)), int(math.ceil(nv))
        big = np.tile(base, (reps_v, reps_u, 1))[: int(round(256 * nv)), : int(round(256 * nu))]
        h, w = big.shape[:2]
        scale = min(1.0, 2048.0 / max(h, w))
        img = Image.fromarray(big)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        _save_png_atomic(img, out)  # [pd_ee run]
        return out
    rng = np.random.default_rng(int(seed))
    base = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    if family == "plain":
        img = np.broadcast_to(base, (size, size, 3)).copy()
    elif family == "noise":
        # sum of a few low-frequency random cosines + fine grain, as brightness modulation
        mod = np.zeros((size, size), np.float32)
        for k in range(6):
            fx, fy = rng.uniform(1, 6, size=2); ph = rng.uniform(0, 2 * np.pi, size=2)
            mod += np.cos(2 * np.pi * fx * x + ph[0]) * np.cos(2 * np.pi * fy * y + ph[1]) / (k + 1)
        mod = mod / np.abs(mod).max() * 0.5 + rng.normal(0, 0.15, (size, size)).astype(np.float32)
        img = np.clip(base * (1 + strength * mod[..., None]), 0, 1)
    elif family == "grid":
        cells = 4
        line = (np.abs((x * cells) % 1 - 0.5) > 0.5 - 0.02) | (np.abs((y * cells) % 1 - 0.5) > 0.5 - 0.02)
        img = np.where(line[..., None], np.clip(base + strength * 0.35, 0, 1), base * np.ones((size, size, 3), np.float32))
    elif family == "stripes":
        planks = 6
        idx = np.floor(y * planks); tone = rng.uniform(-1, 1, planks)[idx.astype(int)]
        grain = np.cos(2 * np.pi * x * rng.uniform(15, 40)) * 0.3 + rng.normal(0, 0.1, (size, size))
        seam = (np.abs((y * planks) % 1 - 0.5) > 0.5 - 0.015)
        mod = 0.35 * tone + 0.25 * grain
        img = np.clip(base * (1 + strength * mod[..., None]), 0, 1)
        img = np.where(seam[..., None], img * 0.55, img)
    elif family == "checker":
        cells = 8
        chk = ((np.floor(x * cells) + np.floor(y * cells)) % 2).astype(np.float32)
        img = np.clip(base * (1 + strength * 0.6 * (chk[..., None] - 0.5)), 0, 1)
    elif family in ("wood", "maniskill_grid"):
        src = np.asarray(Image.open(MANISKILL_WOOD if family == "wood" else MANISKILL_GRID).convert("RGB").resize((size, size))).astype(np.float32) / 255
        lum = src.mean(2, keepdims=True); mean = lum.mean()
        tinted = np.clip(base * (lum / max(mean, 1e-3)), 0, 1)             # texture brightness pattern on the requested colour
        img = (1 - strength) * np.broadcast_to(base, src.shape) + strength * tinted
    _save_png_atomic(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)), out)  # [pd_ee run]
    return out


# ------------------------------------------------------------------------------------------
# randomization
# ------------------------------------------------------------------------------------------
def _jitter_color(rng, color, hue=0.0, sat=0.0, val=0.0):
    h, s, v = colorsys.rgb_to_hsv(*[float(c) for c in color])
    h = (h + rng.uniform(-hue, hue)) % 1.0
    s = float(np.clip(s * (1 + rng.uniform(-sat, sat)), 0, 1))
    v = float(np.clip(v * (1 + rng.uniform(-val, val)), 0, 1))
    return tuple(round(c, 4) for c in colorsys.hsv_to_rgb(h, s, v))


def _random_color(rng, v_range=(0.15, 0.95), s_range=(0.0, 0.9)):
    return tuple(round(c, 4) for c in colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(*s_range), rng.uniform(*v_range)))


def _jitter_dir(rng, d, max_deg):
    d = np.asarray(d, dtype=np.float64); d /= np.linalg.norm(d)
    ang = math.radians(rng.uniform(0, max_deg)); phi = rng.uniform(0, 2 * math.pi)
    a = np.cross(d, [0, 0, 1.0]); a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else np.array([1.0, 0, 0])
    b = np.cross(d, a)
    v = math.cos(ang) * d + math.sin(ang) * (math.cos(phi) * a + math.sin(phi) * b)
    v[2] = -abs(v[2]) if v[2] > -0.15 else v[2]     # keep it coming from above
    return tuple(round(float(c), 4) for c in v / np.linalg.norm(v))


def _random_dir_upper(rng, min_elev_deg=25):
    elev = math.radians(rng.uniform(min_elev_deg, 85)); az = rng.uniform(0, 2 * math.pi)
    return (round(math.cos(elev) * math.cos(az), 4), round(math.cos(elev) * math.sin(az), 4), round(-math.sin(elev), 4))


def _color_temperature(rng, spread):
    """white with a warm/cool tint of the given spread"""
    t = rng.uniform(-spread, spread)
    return (round(1.0 + max(t, 0) * 0.35, 4), round(1.0 - abs(t) * 0.1, 4), round(1.0 + max(-t, 0) * 0.35, 4))


def sample_appearance(mode: str, seed: int, base: str | Appearance = "real") -> Appearance:
    """Draw one appearance. ``mode``: none | mild | wide."""
    base_app = PRESETS[base] if isinstance(base, str) else base
    if mode in (None, "none", "fixed"):
        return dataclasses.replace(base_app)
    rng = np.random.default_rng(int(seed))
    a = dataclasses.replace(base_app, name=f"{base_app.name}-{mode}-{int(seed)}")
    if mode == "mild":
        a.table_color = _jitter_color(rng, base_app.table_color, hue=0.02, sat=0.15, val=0.2)
        a.table_texture = rng.choice(["plain", "noise", "noise", "grid"]) if base_app.table_texture in (None, "plain", "noise", "grid") else base_app.table_texture
        a.table_texture_scale = float(round(base_app.table_texture_scale * rng.uniform(0.6, 1.6), 3))
        if base_app.table_texture_repeats is not None:
            a.table_texture_repeats = tuple(float(round(r * rng.uniform(0.7, 1.4), 3)) for r in base_app.table_texture_repeats)
        a.table_texture_strength = float(round(np.clip(base_app.table_texture_strength * rng.uniform(0.5, 1.8), 0, 1), 3))
        a.floor_color = _jitter_color(rng, base_app.floor_color, hue=0.03, sat=0.3, val=0.35)
        a.floor_texture_strength = float(round(np.clip(base_app.floor_texture_strength * rng.uniform(0.5, 1.8), 0, 1), 3))
        a.canvas_color = _jitter_color(rng, base_app.canvas_color, hue=0.02, sat=0.5, val=0.08)
        a.canvas_rim_color = _jitter_color(rng, base_app.canvas_rim_color, hue=0.05, sat=0.3, val=0.3)
        a.ambient = float(round(base_app.ambient * rng.uniform(0.7, 1.3), 4))
        a.key_dir = _jitter_dir(rng, base_app.key_dir, 12)
        a.key_intensity = float(round(base_app.key_intensity * rng.uniform(0.7, 1.3), 4))
        a.key_color = _color_temperature(rng, 0.25)
        if base_app.key2_dir is not None:
            a.key2_dir = _jitter_dir(rng, base_app.key2_dir, 12)
            a.key2_intensity = float(round(base_app.key2_intensity * rng.uniform(0.7, 1.3), 4))
            a.key2_color = _color_temperature(rng, 0.25)
        a.light_angle_deg = float(round(base_app.light_angle_deg * rng.uniform(0.6, 1.4), 2))
        a.fill_intensity = float(round(base_app.fill_intensity * rng.uniform(0.6, 1.4), 4))
        a.exposure = float(round(rng.uniform(0.9, 1.1), 4))
        if base_app.led_color is not None:
            a.led_color = _jitter_color(rng, base_app.led_color, hue=0.03, sat=0.15, val=0.15)
            a.led_intensity = float(round(base_app.led_intensity * rng.uniform(0.7, 1.3), 3))
    elif mode == "wide":
        a.table_color = _random_color(rng, v_range=(0.1, 0.9))
        a.table_texture = str(rng.choice(["plain", "noise", "grid", "stripes", "checker", "wood"]))
        a.table_texture_scale = float(round(rng.uniform(0.3, 3.0), 3))
        a.table_texture_repeats = None
        a.table_texture_strength = float(round(rng.uniform(0.0, 1.0), 3))
        a.floor_color = _random_color(rng, v_range=(0.03, 0.8))
        a.floor_texture = str(rng.choice(["plain", "noise", "grid", "checker", "maniskill_grid", "stripes"]))
        a.floor_texture_scale = float(round(rng.uniform(0.1, 1.5), 3))
        a.floor_texture_strength = float(round(rng.uniform(0.0, 1.0), 3))
        a.canvas_color = _random_color(rng, v_range=(0.7, 1.0), s_range=(0.0, 0.25))
        a.canvas_rim_color = _random_color(rng, v_range=(0.05, 0.6))
        a.canvas_rim_width = float(rng.choice([0.0, 0.008, 0.012, 0.02]))
        a.ink_color = _random_color(rng, v_range=(0.05, 0.6), s_range=(0.3, 1.0))
        a.background_color = _random_color(rng, v_range=(0.02, 0.8))
        a.ambient = float(round(rng.uniform(0.1, 0.5), 4))
        a.key_dir = _random_dir_upper(rng)
        a.key_intensity = float(round(rng.uniform(0.4, 1.8), 4))
        a.key_color = _color_temperature(rng, 0.6)
        a.fill_dir = _random_dir_upper(rng, min_elev_deg=45)
        a.fill_intensity = float(round(rng.uniform(0.0, 1.2), 4))
        a.fill_color = _color_temperature(rng, 0.4)
        a.key2_dir = _random_dir_upper(rng) if rng.uniform() < 0.6 else None
        a.key2_intensity = float(round(rng.uniform(0.2, 1.2), 4)) if a.key2_dir is not None else 0.0
        a.key2_color = _color_temperature(rng, 0.6)
        a.light_angle_deg = float(round(rng.uniform(0.0, 30.0), 2))
        a.shadows = bool(rng.uniform() < 0.7)
        a.exposure = float(round(rng.uniform(0.8, 1.25), 4))
        a.led_color = None if rng.uniform() < 0.3 else _random_color(rng, v_range=(0.6, 1.0), s_range=(0.4, 1.0))
        a.led_intensity = float(round(rng.uniform(0.5, 1.5), 3))
    else:
        raise ValueError(f"unknown randomization mode {mode!r} (none | mild | wide)")
    return a


def resolve_appearance(options: dict, episode_seed: int | None = None) -> Appearance:
    """Build the appearance from backend options: ``appearance`` (preset name or JSON dict), ``appearance_random``
    (none | mild | wide) and ``appearance_seed`` (+ ``episode_seed`` for per-episode / per-env draws)."""
    base = options.get("appearance", "maniskill")
    if isinstance(base, dict):
        base = Appearance.from_dict(base)
    elif isinstance(base, str) and base.strip().startswith("{"):
        base = Appearance.from_dict(json.loads(base))
    elif base not in PRESETS:
        raise ValueError(f"unknown appearance preset {base!r}; choose from {list(PRESETS)} or pass a JSON dict")
    mode = options.get("appearance_random", "none")
    seed = int(options.get("appearance_seed", 0)) + (int(episode_seed) if episode_seed is not None else 0)
    return sample_appearance(mode, seed, base)


def table_texture_png(a: Appearance, table_size_xy=None, seed: int = 0) -> Path | None:
    """Texture for the table top with the repeats baked in (u along the table's x/depth, v along y/width)."""
    if table_size_xy is None:
        table_size_xy = table_size_of(a)   # NOTE: callers with a non-default canvas pass the scaled size explicitly
    reps = a.table_texture_repeats if a.table_texture_repeats is not None else (a.table_texture_scale * table_size_xy[0], a.table_texture_scale * table_size_xy[1])
    return texture_path(a.table_texture, a.table_color, a.table_texture_strength, seed, tile=reps)


def floor_texture_png(a: Appearance, seed: int = 0) -> Path | None:
    """Untiled floor pattern; one image spans ``1 / floor_texture_scale`` metres."""
    return texture_path(a.floor_texture, a.floor_color, a.floor_texture_strength, seed)


def describe(a: Appearance) -> str:
    return (f"{a.name}: table {tuple(round(c, 2) for c in a.table_color)} {a.table_texture}x{a.table_texture_scale:g}/{a.table_texture_strength:g}, "
            f"floor {tuple(round(c, 2) for c in a.floor_color)} {a.floor_texture}, canvas {tuple(round(c, 2) for c in a.canvas_color)} rim {a.canvas_rim_width:g}, "
            f"ink {tuple(round(c, 2) for c in a.ink_color)}, ambient {a.ambient:g}, key {tuple(round(c, 2) for c in a.key_dir)} x{a.key_intensity:g} {tuple(round(c, 2) for c in a.key_color)}, "
            f"key2 {None if a.key2_dir is None else tuple(round(c, 2) for c in a.key2_dir)} x{a.key2_intensity:g}, angle {a.light_angle_deg:g} deg, "
            f"fill x{a.fill_intensity:g}, shadows {a.shadows}, exposure {a.exposure:g}, pen {a.pen_style}, "
            f"table extent {tuple(round(v, 2) for v in table_extent_of(a))}, led {None if a.led_color is None else tuple(round(c, 2) for c in a.led_color)}")
