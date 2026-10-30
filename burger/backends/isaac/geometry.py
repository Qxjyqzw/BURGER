"""Simulator-independent description of ManiSkill's ``DrawSVGParallel-v1`` scene, used to rebuild it in Isaac Sim.

Everything is expressed in ManiSkill's world frame (robot base at ``(-0.615, 0, 0)``, table top at z = 0), which
the Isaac env uses as the per-environment local frame (env origin = ManiSkill world origin). Quaternions returned
here are ``(x, y, z, w)`` as used by Isaac Lab 3.x / USD.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.spatial.transform import Rotation

# --- ManiSkill TableSceneBuilder / DrawSVGParallelEnv constants -----------------------------------
ROBOT_BASE = (-0.615, 0.0, 0.0)                    # panda_stick root pose (identity rotation)
REST_QPOS = (0.0, math.pi / 8, 0.0, -math.pi * 5 / 8, 0.0, math.pi * 3 / 4, math.pi / 4)
TABLE_HEIGHT = 0.9196429                           # table.glb (scale 1.75) top at z = 0
TABLE_POS = (-0.12, 0.0, -TABLE_HEIGHT)            # visual + collision box pose, yawed 90 deg
TABLE_YAW = math.pi / 2                            # ManiSkill's yaw of the glb (SAPIEN keeps the glTF axes as-is)
# Kit's asset converter already lands the converted table with its long axis along y and its height along z, so the
# Isaac scene needs a yaw of 180 deg (not 90) to reproduce ManiSkill's world AABB x[-0.740, 0.469] y[-1.215, 1.203]
ISAAC_TABLE_YAW = math.pi
TABLE_SCALE = 1.75
TABLE_COLLISION_SIZE = (1.209, 2.418, TABLE_HEIGHT)  # world-aligned extents of ManiSkill's box collider
TABLE_COLLISION_CENTER = (-0.12, 0.0, -TABLE_HEIGHT / 2)
GROUND_Z = -TABLE_HEIGHT
GROUND_SIZE = 100.0
GROUND_TEXTURE_SQUARE_M = 4.0                      # build_ground(texture_square_len=4)
CANVAS_SIZE = (0.8, 1.2, 0.02)                     # half_size (0.4, 0.6, 0.01)
CANVAS_CENTER = (-0.1, 0.0, 0.01)
CANVAS_TOP_Z = 0.02
DOT_RADIUS = 0.004
DOT_THICKNESS = 0.003
DOT_COLOR = (0.8, 0.2, 0.2)
DOT_HIDDEN_Z = -DOT_THICKNESS                      # dots wait inside the table
DOT_PLACED_Z = CANVAS_TOP_Z + DOT_THICKNESS / 2
TOUCH_TOLERANCE_Z = 0.012
TOUCH_Z_THRESHOLD = CANVAS_TOP_Z + DOT_THICKNESS + TOUCH_TOLERANCE_Z   # 0.035: tcp below this leaves ink
CAMERA_EYE = (0.3, 0.0, 0.8)
CAMERA_TARGET = (0.0, 0.0, 0.1)
CAMERA_FOVY_320x240 = 1.2
CAMERA_FOVX = 2 * math.atan(math.tan(CAMERA_FOVY_320x240 / 2) * 320.0 / 240.0)   # 1.4790 rad = 84.74 deg, kept fixed
WRIST_CAMERA_POS = (-0.1, 0.0, -0.05)             # in panda_hand, SAPIEN x-forward camera convention
WRIST_CAMERA_EULER_SXYZ = (0.0, -math.pi / 2, 0.0)
TCP_TARGET_EULER_XYZ = (math.pi, 0.0, 0.0)        # pd_ee_pose orientation target for every action
# ManiSkill default lighting: ambient 0.3 + directional (1, 1, -1) + directional (0, 0, -1)
SUN_DIRECTION = (1.0, 1.0, -1.0)
FILL_DIRECTION = (0.0, 0.0, -1.0)
# ManiSkill's pd_ee_pose controller (panda_stick)
ARM_STIFFNESS, ARM_DAMPING, ARM_FORCE_LIMIT = 1000.0, 100.0, 100.0
SIM_DT, CONTROL_DECIMATION = 0.01, 5              # sim_freq 100, control_freq 20
IK_LAMBDA = 1e-4                                   # Levenberg-Marquardt regularisation of Kinematics.compute_ik


def quat_wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 1:4], q[..., :1]], axis=-1)


def quat_xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def look_at_quat_xyzw(eye, target, up=(0.0, 0.0, 1.0)):
    """Orientation of an x-forward / y-left / z-up frame (SAPIEN camera convention, Isaac Lab ``convention="world"``)
    looking from ``eye`` at ``target``; identical to ``sapien_utils.look_at``."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    left = np.cross(up, f)
    left /= np.linalg.norm(left)
    u = np.cross(f, left)
    return tuple(float(v) for v in Rotation.from_matrix(np.stack([f, left, u], axis=1)).as_quat())


def euler_sxyz_quat_xyzw(rx, ry, rz):
    """transforms3d ``euler2quat(rx, ry, rz)`` (static xyz) as (x, y, z, w)."""
    return tuple(float(v) for v in Rotation.from_euler("xyz", [rx, ry, rz]).as_quat())


def yaw_quat_xyzw(yaw):
    return tuple(float(v) for v in Rotation.from_euler("z", yaw).as_quat())


def distant_light_quat_xyzw(direction):
    """USD distant lights emit along their local -Z; rotate -Z onto ``direction``."""
    d = np.asarray(direction, dtype=np.float64)
    d /= np.linalg.norm(d)
    src = np.array([0.0, 0.0, -1.0])
    axis = np.cross(src, d)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0) if d[2] < 0 else (1.0, 0.0, 0.0, 0.0)
    angle = math.atan2(n, float(np.dot(src, d)))
    return tuple(float(v) for v in Rotation.from_rotvec(axis / n * angle).as_quat())


def horizontal_aperture_for(fovx_rad: float, focal_length: float = 24.0) -> float:
    """USD pinhole: fovx = 2 atan(aperture / (2 f)). The vertical FOV then follows the render aspect."""
    return 2.0 * focal_length * math.tan(fovx_rad / 2.0)


def fovy_for(width: int, height: int, fovx_rad: float = CAMERA_FOVX) -> float:
    return 2 * math.atan(math.tan(fovx_rad / 2) * height / width)


@dataclasses.dataclass
class SceneSpec:
    """Resolved numbers that go into the dataset's backend_config (for reproducibility / comparison)."""

    robot_base_xyz: tuple = ROBOT_BASE
    rest_qpos: tuple = REST_QPOS
    table_pos: tuple = TABLE_POS
    table_yaw: float = ISAAC_TABLE_YAW
    table_scale: float = TABLE_SCALE
    canvas_center_xyz: tuple = CANVAS_CENTER
    canvas_size: tuple = CANVAS_SIZE
    camera_eye_xyz: tuple = CAMERA_EYE
    camera_target_xyz: tuple = CAMERA_TARGET
    camera_quat_xyzw: tuple = dataclasses.field(default_factory=lambda: look_at_quat_xyzw(CAMERA_EYE, CAMERA_TARGET))
    camera_fovx_deg: float = math.degrees(CAMERA_FOVX)
    camera_width: int = 128
    camera_height: int = 128
    camera_fovy_deg: float = math.degrees(fovy_for(128, 128))
    wrist_camera_pos: tuple = WRIST_CAMERA_POS
    wrist_camera_quat_xyzw: tuple = dataclasses.field(default_factory=lambda: euler_sxyz_quat_xyzw(*WRIST_CAMERA_EULER_SXYZ))
    touch_z_threshold: float = TOUCH_Z_THRESHOLD
    dot_radius: float = DOT_RADIUS
    dot_thickness: float = DOT_THICKNESS

    @classmethod
    def for_camera(cls, width: int, height: int) -> "SceneSpec":
        return cls(camera_width=int(width), camera_height=int(height), camera_fovy_deg=math.degrees(fovy_for(width, height)))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
