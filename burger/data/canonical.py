"""Canonical, backend-agnostic episode container for the painter data pipeline.

A *backend* (ManiSkill or LIBERO) turns SVG waypoints into episodes; a *format writer*
(LIBERO ``*_demo.hdf5`` or LeRobot v2.1) consumes them. Both sides only depend on the schema
below, which is stored as one HDF5 file per generation run (``*.canonical.h5``):

file attrs
    painter_canonical_version, backend, language, control_freq, robot, cameras (json list),
    action_space (json), backend_config (json), generation (json), num_episodes, total_steps,
    image_row_order = "top_down"  (all images are stored in display orientation)
episodes/ep_{i}/
    obs/agentview_rgb        (T, H, W, 3) uint8     third-person camera (ManiSkill base_camera / LIBERO agentview)
    obs/eye_in_hand_rgb      (T, h, w, 3) uint8     optional wrist camera
    obs/eef_pos              (T, 3)  world frame     ManiSkill tcp / LIBERO grip site
    obs/eef_quat_wxyz        (T, 4)
    obs/joint_pos            (T, 7)  optional        arm joints
    obs/gripper_qpos         (T, 2)  optional
    obs/pen_tip_painter      (T, 3)  optional        pen tip in the robot base ("painter") frame
    obs/ink_count            (T,)    optional
    actions                  (T, A)  native backend actions (see action_space attr)
    arrays/*                 episode-level arrays: waypoints_world/painter, ink_dots_xyz,
                             backend_state (T, S) (LIBERO MuJoCo state), targets_world (T, 3) ...
    attrs                    json-serialisable metadata: svg_file, svg_string, num_steps, ink_count,
                             tip errors, gen_time_s, model_file / init_state (LIBERO), ...
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

CANONICAL_VERSION = "1.0"
IMAGE_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
REQUIRED_OBS = ("agentview_rgb", "eef_pos", "eef_quat_wxyz")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not json serialisable: {type(o)}")


def dumps(obj) -> str:
    return json.dumps(obj, default=_json_default)


@dataclasses.dataclass
class DatasetInfo:
    backend: str                    # "maniskill" | "libero" | "isaac"
    language: str
    control_freq: int
    robot: str                      # e.g. "panda_stick", "panda_gripper_pen"
    cameras: list                   # canonical camera keys present, e.g. ["agentview_rgb", "eye_in_hand_rgb"]
    action_space: dict              # {"type": ..., "dim": ..., "description": ...}
    backend_config: dict = dataclasses.field(default_factory=dict)
    generation: dict = dataclasses.field(default_factory=dict)

    def to_attrs(self) -> dict:
        return dict(
            painter_canonical_version=CANONICAL_VERSION, backend=self.backend, language=self.language,
            control_freq=int(self.control_freq), robot=self.robot, cameras=dumps(self.cameras),
            action_space=dumps(self.action_space), backend_config=dumps(self.backend_config),
            generation=dumps(self.generation), image_row_order="top_down",
        )

    @classmethod
    def from_attrs(cls, a) -> "DatasetInfo":
        return cls(
            backend=str(a["backend"]), language=str(a["language"]), control_freq=int(a["control_freq"]), robot=str(a["robot"]),
            cameras=json.loads(a["cameras"]), action_space=json.loads(a["action_space"]),
            backend_config=json.loads(a["backend_config"]), generation=json.loads(a["generation"]),
        )


@dataclasses.dataclass
class Episode:
    obs: dict                       # name -> (T, ...) array
    actions: np.ndarray             # (T, A)
    meta: dict = dataclasses.field(default_factory=dict)
    arrays: dict = dataclasses.field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return int(self.actions.shape[0])

    def validate(self):
        actions = np.asarray(self.actions)
        if actions.ndim != 2 or not all(actions.shape):
            raise ValueError("actions must be a nonempty (T, A) numeric array")
        _check_finite(actions, "actions")
        T = actions.shape[0]
        for k in REQUIRED_OBS:
            if k not in self.obs:
                raise ValueError(f"episode is missing required obs '{k}'")
        for k, v in self.obs.items():
            _validate_observation(k, np.asarray(v), T)
        for k, v in self.arrays.items():
            v = np.asarray(v)
            _check_finite(v, f"arrays/{k}")
            if k == "targets_world" and v.shape != (T, 3):
                raise ValueError("arrays/targets_world must be (T, 3)")
            if k == "backend_state" and (v.ndim != 2 or v.shape[0] != T):
                raise ValueError("arrays/backend_state must be (T, S)")
        return self


def _check_finite(value, name: str):
    """Validate low-dimensional numeric arrays without silently coercing corrupt strings."""
    if (not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_) or np.issubdtype(value.dtype, np.complexfloating):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")


def _validate_observation(name: str, value, steps: int):
    if value.ndim == 0 or value.shape[0] != steps:
        raise ValueError(f"obs/{name} must have {steps} steps")
    if name in IMAGE_KEYS or name.endswith("_rgb"):
        if value.dtype != np.uint8 or value.ndim != 4 or value.shape[-1] != 3 or min(value.shape[1:3]) < 1:
            raise ValueError(f"obs/{name} must be (T, H, W, 3) uint8")
        return
    suffixes = {"eef_pos": (3,), "eef_quat_wxyz": (4,), "pen_tip_painter": (3,), "pen_tip_world": (3,),
                "joint_pos": (7,), "gripper_qpos": (2,)}
    if name in suffixes and value.shape[1:] != suffixes[name]:
        raise ValueError(f"obs/{name} must have shape (T, {suffixes[name][0]})")
    _check_finite(value, f"obs/{name}")
    if name == "eef_quat_wxyz" and not np.allclose(np.linalg.norm(value, axis=1), 1.0, atol=1e-3, rtol=0):
        raise ValueError("obs/eef_quat_wxyz must contain unit quaternions")


class CanonicalWriter:
    """Appends episodes to a canonical HDF5 file (images gzip-compressed)."""

    def __init__(self, path: str | Path, info: DatasetInfo, mode: str = "w", image_compression: str | None = "lzf"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.path, mode)
        self.info = info
        # lzf: fast (~10x quicker than gzip-4 on 128x128 frames) with a good ratio on mostly flat renders
        self.image_compression = image_compression
        self.grp = self.f.require_group("episodes")
        for k, v in info.to_attrs().items():
            self.f.attrs[k] = v
        self.f.attrs["painter_write_complete"] = False
        self._n = len(self.grp.keys())
        self._steps = int(self.f.attrs.get("total_steps", 0))

    def add_episode(self, ep: Episode, index: int | None = None) -> str:
        ep.validate()
        if set(k for k in ep.obs if k in IMAGE_KEYS) != set(self.info.cameras):
            raise ValueError("episode cameras do not match DatasetInfo.cameras")
        if ep.actions.shape[1] != int(self.info.action_space["dim"]):
            raise ValueError("episode action dimension does not match DatasetInfo.action_space")
        name = f"ep_{self._n if index is None else index}"
        g = self.grp.create_group(name)
        og = g.create_group("obs")
        for k, v in ep.obs.items():
            v = np.asarray(v)
            kw = dict(compression=self.image_compression, chunks=(min(16, len(v)),) + v.shape[1:]) if (v.ndim >= 3 and self.image_compression) else {}
            og.create_dataset(k, data=v, **kw)
        g.create_dataset("actions", data=np.asarray(ep.actions))
        ag = g.create_group("arrays")
        for k, v in ep.arrays.items():
            ag.create_dataset(k, data=np.asarray(v))
        for k, v in ep.meta.items():
            if isinstance(v, (dict, list, tuple)):
                g.attrs[k] = dumps(v)
            elif isinstance(v, np.ndarray):
                g.attrs[k] = v
            elif v is None:
                continue
            else:
                g.attrs[k] = v
        g.attrs["num_steps"] = ep.num_steps
        self._n += 1
        self._steps += ep.num_steps
        self.f.attrs["num_episodes"] = self._n
        self.f.attrs["total_steps"] = self._steps
        self.f.flush()
        return name

    def close(self, complete: bool = True):
        if not self.f.id.valid:
            return
        self.f.attrs["num_episodes"] = self._n
        self.f.attrs["total_steps"] = self._steps
        self.f.attrs["painter_write_complete"] = complete
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(complete=exc_type is None)


def episode_names(f: h5py.File) -> list:
    if "episodes" not in f or any(re.fullmatch(r"ep_\d+", name) is None for name in f["episodes"]):
        raise ValueError("canonical episodes must be named ep_<nonnegative integer>")
    return sorted(f["episodes"].keys(), key=lambda s: int(s.split("_")[1]))


def read_info(path: str | Path) -> DatasetInfo:
    with h5py.File(path, "r") as f:
        _validate_header(f)
        return DatasetInfo.from_attrs(f.attrs)


def _validate_header(f):
    if f.attrs.get("painter_canonical_version") != CANONICAL_VERSION:
        raise ValueError(f"unsupported canonical version: {f.attrs.get('painter_canonical_version')!r}")
    if f.attrs.get("image_row_order") != "top_down":
        raise ValueError("canonical image_row_order must be top_down")
    if not f.attrs.get("painter_write_complete", True):
        raise ValueError("canonical file was not completed successfully")
    required = ("backend", "language", "control_freq", "robot", "cameras", "action_space",
                "backend_config", "generation", "num_episodes", "total_steps")
    missing = [k for k in required if k not in f.attrs]
    if missing:
        raise ValueError(f"canonical file is missing attributes: {missing}")
    info = DatasetInfo.from_attrs(f.attrs)
    if info.control_freq <= 0 or not info.language.strip():
        raise ValueError("canonical data requires a positive control_freq and nonempty language")
    if not isinstance(info.cameras, list) or "agentview_rgb" not in info.cameras or len(info.cameras) != len(set(info.cameras)):
        raise ValueError("canonical cameras must be a unique list containing agentview_rgb")
    if set(info.cameras) - set(IMAGE_KEYS):
        raise ValueError(f"unsupported canonical cameras: {info.cameras}")
    if int(info.action_space.get("dim", 0)) <= 0:
        raise ValueError("canonical action_space.dim must be positive")
    return info


def validate_file(path: str | Path, require_painter: bool = False) -> dict:
    """Check every episode before export, including files written by older versions.

    Images are inspected by shape/dtype without loading the video-sized arrays. Numeric
    observations are checked episode by episode. Legacy files without a completion flag
    remain readable if their schema, counts and values are consistent.
    """
    with h5py.File(path, "r") as f:
        info = _validate_header(f)
        names = episode_names(f)
        if not names:
            raise ValueError("cannot export an empty canonical dataset")
        total = 0
        image_shapes = {}
        for name in names:
            g = f["episodes"][name]
            if "actions" not in g or "obs" not in g:
                raise ValueError(f"{name}: missing actions or obs")
            actions = g["actions"][:]
            if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != int(info.action_space["dim"]):
                raise ValueError(f"{name}: invalid actions shape {actions.shape}")
            _check_finite(actions, f"{name}/actions")
            steps = actions.shape[0]
            if int(g.attrs.get("num_steps", -1)) != steps:
                raise ValueError(f"{name}: num_steps differs from action rows")
            if any(k not in g["obs"] for k in REQUIRED_OBS):
                raise ValueError(f"{name}: missing required observations")
            if set(k for k in g["obs"] if k in IMAGE_KEYS) != set(info.cameras):
                raise ValueError(f"{name}: camera presence differs from declared cameras")
            for key, ds in g["obs"].items():
                is_image = key in IMAGE_KEYS or key.endswith("_rgb")
                _validate_observation(key, ds if is_image else ds[:], steps)
                if is_image:
                    if key in image_shapes and ds.shape[1:] != image_shapes[key]:
                        raise ValueError(f"{name}: image shape changes between episodes for {key}")
                    image_shapes[key] = ds.shape[1:]
            if "arrays" in g:
                for key, ds in g["arrays"].items():
                    _check_finite(ds[:], f"{name}/arrays/{key}")
                    if key == "targets_world" and ds.shape != (steps, 3):
                        raise ValueError(f"{name}: targets_world must have shape (T, 3)")
                    if key == "backend_state" and (ds.ndim != 2 or ds.shape[0] != steps):
                        raise ValueError(f"{name}: backend_state must have shape (T, S)")
            if require_painter:
                for key in ("obs/pen_tip_painter", "obs/pen_tip_world", "arrays/targets_world"):
                    if key not in g:
                        raise ValueError(f"{name}: {key} is required for painter absolute-target export")
                offsets = g["obs/pen_tip_world"][:] - g["obs/pen_tip_painter"][:]
                if not np.allclose(offsets, offsets[0], rtol=0, atol=1e-5):
                    raise ValueError(f"{name}: painter and world positions must differ by a constant translation")
            total += steps
        if int(f.attrs["num_episodes"]) != len(names) or int(f.attrs["total_steps"]) != total:
            raise ValueError("canonical episode/step totals do not match stored data")
        return dict(num_episodes=len(names), total_steps=total, cameras=info.cameras)


def load_episode(g: h5py.Group) -> Episode:
    obs = {k: g["obs"][k][:] for k in g["obs"].keys()}
    arrays = {k: g["arrays"][k][:] for k in g["arrays"].keys()} if "arrays" in g else {}
    meta = {}
    for k, v in g.attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        meta[k] = v
    return Episode(obs=obs, actions=g["actions"][:], meta=meta, arrays=arrays)


def iter_episodes(path: str | Path) -> Iterator[tuple]:
    with h5py.File(path, "r") as f:
        for name in episode_names(f):
            yield name, load_episode(f["episodes"][name])


def merge_parts(part_paths: list, out_path: str | Path, info: DatasetInfo, order_key: str = "svg_index") -> Path:
    """Merge canonical part files (from parallel workers) into one file, episodes ordered by ``order_key``."""
    if not part_paths:
        raise ValueError("cannot merge an empty list of canonical parts")
    if any(Path(part).resolve() == Path(out_path).resolve() for part in part_paths):
        raise ValueError("merged output must differ from all canonical input parts")
    items = []
    for part in part_paths:
        validate_file(part)
        part_info = read_info(part)
        for key in ("backend", "language", "control_freq", "robot", "cameras", "action_space"):
            if json.loads(dumps(getattr(info, key))) != json.loads(dumps(getattr(part_info, key))):
                raise ValueError(f"canonical part {part} has incompatible {key}")
        with h5py.File(part, "r") as pf:
            for name in episode_names(pf):
                items.append((int(pf["episodes"][name].attrs.get(order_key, 0)), str(part), name))
    items.sort()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with h5py.File(out_path, "w") as out:
        grp = out.create_group("episodes")
        for i, (_, part, name) in enumerate(items):
            with h5py.File(part, "r") as pf:
                pf.copy(pf["episodes"][name], grp, name=f"ep_{i}")
                total += int(pf["episodes"][name].attrs["num_steps"])
        for k, v in info.to_attrs().items():
            out.attrs[k] = v
        out.attrs["num_episodes"] = len(items)
        out.attrs["total_steps"] = total
        out.attrs["painter_write_complete"] = True
    return out_path


def summarize(path: str | Path) -> dict:
    with h5py.File(path, "r") as f:
        info = DatasetInfo.from_attrs(f.attrs)
        names = episode_names(f)
        steps = [int(f["episodes"][n].attrs["num_steps"]) for n in names]
        img = f["episodes"][names[0]]["obs/agentview_rgb"].shape if names else None
        return dict(path=str(path), backend=info.backend, language=info.language, robot=info.robot, cameras=info.cameras,
                    action_space=info.action_space, num_episodes=len(names), total_steps=int(sum(steps)), steps_per_episode=steps,
                    agentview_shape=list(img) if img else None, control_freq=info.control_freq)


# ------------------------------------------------------------------------------------------
# helpers shared by backends / writers
# ------------------------------------------------------------------------------------------
def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return np.concatenate([q[..., 1:4], q[..., :1]], axis=-1)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def quat_xyzw_to_axisangle(q: np.ndarray) -> np.ndarray:
    """Batched robosuite-style quat (x, y, z, w) -> axis-angle (3,)."""
    q = np.asarray(q, dtype=np.float64)
    xyz, w = q[..., :3], np.clip(q[..., 3], -1.0, 1.0)
    den = np.sqrt(np.clip(1.0 - w**2, 0.0, None))
    angle = 2.0 * np.arccos(w)
    scale = np.where(den < 1e-8, 0.0, angle / np.where(den < 1e-8, 1.0, den))
    return xyz * scale[..., None]


def derive_libero_delta_actions(target_pos_world: np.ndarray, eef_pos_world: np.ndarray, gain_pos: float = 4.0,
                                pos_scale: float = 0.05, gripper: float = 1.0) -> np.ndarray:
    """LIBERO-style OSC_POSE deltas in [-1, 1] from absolute position targets: the same
    closed-loop rule the LIBERO backend applies (``clip(gain * (target - eef) / 0.05)``), with
    zero rotation deltas (the orientation is held) and a closed gripper."""
    dpos = np.clip(gain_pos * (np.asarray(target_pos_world) - np.asarray(eef_pos_world)) / pos_scale, -1.0, 1.0)
    T = len(dpos)
    return np.concatenate([dpos, np.zeros((T, 3)), np.full((T, 1), float(gripper))], axis=1)


def resize_images(imgs: np.ndarray, hw) -> np.ndarray:
    """Resize (T, H, W, 3) uint8 frames to (h, w) with bilinear filtering (aspect is squashed if it differs)."""
    from PIL import Image
    h, w = int(hw[0]), int(hw[1])
    if h < 1 or w < 1:
        raise ValueError("image height and width must be positive")
    if imgs.shape[1] == h and imgs.shape[2] == w:
        return imgs
    return np.stack([np.asarray(Image.fromarray(x).resize((w, h), Image.BILINEAR)) for x in imgs])
