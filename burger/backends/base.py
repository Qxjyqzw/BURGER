from __future__ import annotations

import dataclasses
import importlib
import json
from pathlib import Path

from burger._paths import PROJECT_ROOT as REPO_ROOT
VENV_FOR_BACKEND = {
    name: REPO_ROOT / venv
    for name, venv in (("maniskill", ".venv-maniskill"), ("libero", ".venv-libero"), ("isaac", ".venv-isaac61"))
    if (Path(__file__).parent / name / "backend.py").is_file()
}
# ManiSkill base_camera: 320x240 with fovy 1.2 rad -> horizontal FOV 84.74 deg, shared by all backends
FIXED_HORIZONTAL_FOV_RAD = 1.4790102243423462


def fovy_for(width: int, height: int, fovx_rad: float = FIXED_HORIZONTAL_FOV_RAD) -> float:
    import math
    return 2 * math.atan(math.tan(fovx_rad / 2) * height / width)


@dataclasses.dataclass
class GenerationConfig:
    """Shared generation settings passed to each simulator backend."""

    svg_files: list                      # ordered SVG paths (one episode each)
    language: str                        # task instruction
    object_name: str = "drawing"
    max_draw_dist: float = 0.008
    end_lift_z: float = 0.085
    svg_xy_noise: float = 0.0
    svg_scale_mode: str = "fixed"        # "fixed" | "max"
    rot_image: bool = True
    camera_height: int = 128
    camera_width: int = 128
    wrist_camera: bool = True
    parallel: int = 1                    # Isaac: vectorized envs; LIBERO: worker processes
    device: int | None = None            # CUDA device for Isaac, EGL device for LIBERO
    seed: int = 0
    backend_options: dict = dataclasses.field(default_factory=dict)  # backend-specific knobs

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["svg_files"] = [str(f) for f in self.svg_files]
        return d


class Backend:
    name: str = "base"
    import_check: str = ""           # module whose import decides availability

    @classmethod
    def available(cls) -> bool:
        try:
            importlib.import_module(cls.import_check)
            return True
        except Exception:
            return False

    @classmethod
    def venv_python(cls) -> Path:
        return VENV_FOR_BACKEND[cls.name] / "bin" / "python"

    def generate(self, cfg: GenerationConfig, out_canonical: Path, work_dir: Path, log=print) -> dict:
        """Run all episodes, write ``out_canonical`` and return a summary dict with timings."""
        raise NotImplementedError


BACKENDS = {}


def register_backend(cls):
    BACKENDS[cls.name] = cls
    return cls


def get_backend(name: str) -> type:
    # import lazily so that a venv without one simulator can still import the package
    if name not in VENV_FOR_BACKEND:
        raise KeyError(f"unknown backend {name!r}; choose from {', '.join(VENV_FOR_BACKEND)}")
    importlib.import_module(f".{name}.backend", package=__package__)
    return BACKENDS[name]
