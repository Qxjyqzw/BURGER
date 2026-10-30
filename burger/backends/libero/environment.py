"""LIBERO / robosuite port of ManiSkill's ``DrawSVGParallel-v1`` drawing scene.

What is copied from the ManiSkill environment
(``burger/backends/maniskill/environment.py``):

* a white canvas (0.8 m x 1.2 m) whose centre is 0.515 m in front of the robot base and
  whose top surface is 0.02 m above the robot base plane;
* a pool of ``max_dots`` kinematic red ink dots (radius 4 mm, thickness 3 mm). At every
  control step, if the pen tip is within ``DOT_THICKNESS + TOUCH_TOLERANCE_Z`` (= 15 mm) of
  the canvas top, one dot is moved from its hidden pose to the pen-tip xy on the canvas —
  this is the "drawing appears once the arm is near the surface" mechanism;
* the ``base_camera`` viewpoint (``look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1])`` relative
  to a base at ``[-0.615, 0, 0]``, fovy = 1.2 rad), re-anchored to the LIBERO robot base.

Differences that are intrinsic to LIBERO and are *not* hidden:

* the robot is LIBERO's standard mounted Franka Panda with the two-finger gripper
  (kept closed) instead of ManiSkill's ``panda_stick``; the "pen tip" is the closed
  fingertip (``gripper0_grip_site`` + ``pen_tip_offset`` along the approach axis);
* actions are LIBERO's 7-D ``OSC_POSE`` deltas in [-1, 1] (0.05 m / 0.5 rad per unit,
  +1 = gripper closed) instead of ManiSkill's absolute ``pd_ee_pose`` targets;
* the robot is LIBERO's standard mounted Franka Panda with the two-finger gripper (see above); by default the
  LIBERO table and floor are re-textured with ManiSkill's wood (``table.glb``) and grey-grid textures and lit like
  ManiSkill (``scene_textures="libero"`` / ``scene_lighting="libero"`` restore the stock robosuite look); the room
  walls are hidden.
"""
from __future__ import annotations

import dataclasses
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, new_body, new_element, new_geom

import libero.libero.envs  # noqa: F401  (populates TASK_MAPPING / OBJECTS_DICT)
from libero.libero.envs.base_object import register_object
from libero.libero.envs.bddl_base_domain import TASK_MAPPING, BDDLBaseDomain, register_problem
from libero.libero.envs.objects import get_object_fn

# LIBERO's @register_problem decorator returns None, so the class symbol exported by
# libero.libero.envs.problems is None; fetch the parent class from the registry instead.
Libero_Tabletop_Manipulation = TASK_MAPPING["libero_tabletop_manipulation"]

from burger.pipeline import svg_trajectory as traj

PROBLEM_NAME = "libero_draw_tabletop_manipulation"
ENV_NAME = "Libero_Draw_Tabletop_Manipulation"

ASSET_DIR = Path(__file__).resolve().parent / "assets"

# ---------------------------------------------------------------------------------------
# ManiSkill scene constants (TableSceneBuilder + DrawSVGParallelEnv)
# ---------------------------------------------------------------------------------------
MANISKILL_ROBOT_BASE = np.array([-0.615, 0.0, 0.0])
MANISKILL_CAMERA_EYE = np.array([0.3, 0.0, 0.8])
MANISKILL_CAMERA_TARGET = np.array([0.0, 0.0, 0.1])
MANISKILL_CAMERA_FOVY_RAD = 1.2                 # vertical FOV of the 320x240 base_camera
MANISKILL_CAMERA_FOVX_RAD = 2 * np.arctan(np.tan(MANISKILL_CAMERA_FOVY_RAD / 2) * 320.0 / 240.0)  # 1.4790 rad
# panda_stick "rest" keyframe used by TableSceneBuilder.initialize()
PANDA_STICK_REST_QPOS = [0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, np.pi / 4]
# ManiSkill TableSceneBuilder table: world AABB x[-0.740, 0.469] y[-1.215, 1.203] z[-0.92, 0]
# with the robot base at x = -0.615 -> relative to the base the top spans x[-0.125, 1.084],
# y[-1.215, 1.203] (~symmetric) and its surface is the base plane (z = 0).
DEFAULT_TABLE_LENGTH = 1.209        # x extent of the table (m)
DEFAULT_TABLE_WIDTH = 2.418         # y extent of the table (m)
DEFAULT_TABLE_X_BEHIND_BASE = 0.125 # how far the table extends behind the robot base (m)
DEFAULT_TABLE_Y_OFFSET = 0.0        # table centre y relative to the base
LIBERO_TABLE_TOP_Z = 0.90           # standard LIBERO tabletop height (kept for reference)
TABLE_THICKNESS = 0.05
# LIBERO's OSC_POSE controller scaling (robosuite default_panda / LIBERO datasets)
OSC_POS_SCALE = 0.05
OSC_ROT_SCALE = 0.5
# [pd_ee] ManiSkill pd_ee_pose analogue (arm_controller="pd_ee"): absolute pen-tip target -> one damped least-squares
# IK step from the current joints (ManiSkill GPU IK: levenberg_marquardt, 1 iteration, lambda 1e-4) -> joint position
# goal tracked by robosuite's JOINT_POSITION controller. kp / kd are ManiSkill panda_stick's drive (stiffness 1e3,
# damping 1e2) per unit inertia; robosuite multiplies by the mass matrix and adds gravity compensation, and the
# Panda actuator torque limits (+-80 / +-12 Nm) still apply.
PD_EE_KP = 1000.0
PD_EE_DAMPING_RATIO = 100.0 / (2.0 * np.sqrt(PD_EE_KP))   # kd = 2*sqrt(kp)*ratio = 100
PD_EE_OUTPUT_MAX = 1.0                                     # rad: |joint goal - q| per control step (never reached in practice)
PD_EE_IK_LAMBDA = 1e-4
FRANKA_JOINT_VEL_LIMIT = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])   # rad/s, Franka Emika Panda datasheet
# down-facing end-effector orientation (robosuite eef frame: z = approach direction)
EEF_DOWN_QUAT_XYZW = np.array([1.0, 0.0, 0.0, 0.0])


def look_at_quat_wxyz(eye, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera quaternion (w, x, y, z) looking from ``eye`` at ``target``.

    Reproduces SAPIEN's ``look_at`` view (same forward direction and same roll), taking
    into account that a MuJoCo camera looks along its -z axis with +y up.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    rot = np.stack([right, cam_up, -forward], axis=1)
    q_xyzw = T.mat2quat(rot)
    return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


@dataclasses.dataclass
class SceneLayout:
    """World-frame layout of the LIBERO drawing scene (all lengths in metres)."""

    table_full_size: tuple
    workspace_offset: tuple          # LIBERO table-top centre (x, y, z_top)
    robot_base_xyz: tuple            # robot base (link0) position
    canvas_center_xyz: tuple         # canvas box centre
    canvas_half_size: tuple          # (hx, hy, hz)
    canvas_top_z: float
    camera_eye_xyz: tuple
    camera_target_xyz: tuple
    camera_quat_wxyz: tuple
    camera_fovy_deg: float
    camera_fovx_deg: float = float(np.degrees(MANISKILL_CAMERA_FOVX_RAD))
    camera_aspect: float = 1.0        # width / height of the stored agentview frames
    table_top_z: float = 0.912

    def fovy_deg_for_aspect(self, aspect: float) -> float:
        """Vertical FOV that keeps the horizontal FOV fixed for a given width/height aspect."""
        return float(np.degrees(2 * np.arctan(np.tan(np.radians(self.camera_fovx_deg) / 2) / aspect)))

    def painter_to_world(self, p):
        """ManiSkill/painter base-frame point(s) -> LIBERO world frame."""
        p = np.asarray(p, dtype=np.float64)
        return p + np.asarray(self.robot_base_xyz)

    def world_to_painter(self, p):
        p = np.asarray(p, dtype=np.float64)
        return p - np.asarray(self.robot_base_xyz)

    def to_dict(self):
        return dataclasses.asdict(self)


def compute_scene_layout(
    table_length: float = DEFAULT_TABLE_LENGTH,
    table_width: float = DEFAULT_TABLE_WIDTH,
    table_x_behind_base: float = DEFAULT_TABLE_X_BEHIND_BASE,
    table_y_offset: float = DEFAULT_TABLE_Y_OFFSET,
    table_top_z: float | None = None,
    robot_base_height: float = 0.912,
    canvas_half_size=(traj.CANVAS_HALF_SIZE[0], traj.CANVAS_HALF_SIZE[1]),
    canvas_top_above_base: float = traj.CANVAS_TOP_Z,
    camera_fovx_rad: float = MANISKILL_CAMERA_FOVX_RAD,
    camera_aspect: float = 1.0,
    camera_fovy_rad: float | None = None,
    camera_offset_xyz=(0.0, 0.0, 0.0),
    camera_eye_rel=None,
    camera_target_rel=None,
) -> SceneLayout:
    """Place table, canvas and camera so that everything *relative to the robot base* matches
    ManiSkill's ``DrawSVGParallel-v1``.

    LIBERO places a table-arena robot at ``x = -0.16 - table_length / 2`` (robosuite
    ``base_xpos_offset["table"]``) and its mounted Panda base at z = 0.912. By default the
    table top is put flush with the base plane (as in ManiSkill, where the Panda stands on
    the table) and the table has ManiSkill's footprint relative to the base (1.209 m deep,
    2.418 m wide, 0.125 m behind the base), so its far edge lands on the same image row and
    its side/near edges fall outside the frame like in the reference renders. The canvas top
    is ``canvas_top_above_base`` (0.02 m) above the base plane.

    Camera: position/orientation are ManiSkill's ``base_camera`` re-anchored to the base. The
    *horizontal* FOV is fixed to ManiSkill's (84.7 deg) and the vertical FOV follows the stored
    frame aspect (``camera_aspect`` = width / height): 4:3 frames reproduce ManiSkill's
    68.75 deg exactly, square 128x128 frames keep the same horizontal coverage (the whole
    canvas is visible) and simply show more table/background above and below — the ManiSkill
    view is their central 4:3 crop. ``camera_fovy_rad`` overrides this if given.
    """
    base_x = -0.16 - table_length / 2.0
    base = np.array([base_x, 0.0, robot_base_height])
    top_z = float(robot_base_height if table_top_z is None else table_top_z)
    table_center_x = base_x - table_x_behind_base + table_length / 2.0
    workspace_offset = (float(table_center_x), float(table_y_offset), top_z)
    canvas_top_z = base[2] + canvas_top_above_base
    canvas_hz = (canvas_top_z - top_z) / 2.0
    if canvas_hz <= 0:
        raise ValueError("canvas top must be above the table top")
    canvas_center = np.array([base_x + traj.CANVAS_CENTER_X, 0.0, top_z + canvas_hz])
    cam_off = np.asarray(camera_offset_xyz, dtype=np.float64)
    if camera_eye_rel is not None:
        # explicit camera (e.g. the lab's forward camera, burger.pipeline.appearance.CAMERA_PRESETS) given relative to the base
        eye = base + np.asarray(camera_eye_rel, dtype=np.float64) + cam_off
        target = base + np.asarray(camera_target_rel, dtype=np.float64) + cam_off
    else:
        eye = base + (MANISKILL_CAMERA_EYE - MANISKILL_ROBOT_BASE) + cam_off
        target = base + (MANISKILL_CAMERA_TARGET - MANISKILL_ROBOT_BASE) + cam_off
    fovx_deg = float(np.degrees(camera_fovx_rad))
    if camera_fovy_rad is not None:
        fovy_deg = float(np.degrees(camera_fovy_rad))
    else:
        fovy_deg = float(np.degrees(2 * np.arctan(np.tan(camera_fovx_rad / 2) / float(camera_aspect))))
    return SceneLayout(
        table_full_size=(float(table_length), float(table_width), TABLE_THICKNESS),
        workspace_offset=workspace_offset,
        robot_base_xyz=tuple(float(v) for v in base),
        canvas_center_xyz=tuple(float(v) for v in canvas_center),
        canvas_half_size=(float(canvas_half_size[0]), float(canvas_half_size[1]), float(canvas_hz)),
        canvas_top_z=float(canvas_top_z),
        camera_eye_xyz=tuple(float(v) for v in eye),
        camera_target_xyz=tuple(float(v) for v in target),
        camera_quat_wxyz=tuple(look_at_quat_wxyz(eye, target)),
        camera_fovy_deg=fovy_deg,
        camera_fovx_deg=fovx_deg,
        camera_aspect=float(camera_aspect),
        table_top_z=top_z,
    )


# ---------------------------------------------------------------------------------------
# Canvas fixture
# ---------------------------------------------------------------------------------------
def _canvas_xml(half_size, rgba, rim_width=0.0, rim_rgba=(0.15, 0.16, 0.16, 1.0)):
    hx, hy, hz = half_size
    rim = ""
    if rim_width > 0:
        # a slightly larger, 0.5 mm lower box: shows as a dark border around the white top (like the whiteboard's frame)
        rim = (f'\n        <geom name="canvas_rim_visual" type="box" size="{hx + rim_width:.5f} {hy + rim_width:.5f} {hz - 0.0005:.5f}" pos="0 0 -0.0005" '
               f'group="1" rgba="{array_to_string(rim_rgba)}" contype="0" conaffinity="0"/>')
    return f"""<mujoco model="drawing_canvas">
  <worldbody>
    <body>
      <body name="object">
        <geom name="canvas_collision" type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}" pos="0 0 0" group="0" rgba="1 1 1 0" friction="1 0.005 0.0001"/>
        <geom name="canvas_visual" type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}" pos="0 0 0" group="1" rgba="{array_to_string(rgba)}" contype="0" conaffinity="0"/>{rim}
        <site name="canvas_surface_region" pos="0 0 {hz:.5f}" quat="1 0 0 0" size="{hx:.5f} {hy:.5f} 0.001" type="box" rgba="0 0 1 0"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {-hz:.5f}" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {hz:.5f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{hx:.5f} {hy:.5f} 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""


@register_object
class DrawingCanvas(MujocoXMLObject):
    """Static white drawing canvas (LIBERO fixture, BDDL category ``drawing_canvas``)."""

    def __init__(self, name="drawing_canvas", joints=None, half_size=(0.4, 0.6, 0.016), rgba=(1.0, 1.0, 1.0, 1.0), rim_width=0.0, rim_rgba=(0.15, 0.16, 0.16, 1.0)):
        half_size = tuple(float(v) for v in half_size)
        rgba = tuple(float(v) for v in rgba)
        rim_rgba = tuple(float(v) for v in rim_rgba)
        gen_dir = ASSET_DIR / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        tag = "_".join(f"{v:.4f}" for v in half_size + rgba + ((float(rim_width),) + rim_rgba if rim_width > 0 else ())).replace(".", "p").replace("-", "m")
        xml_path = gen_dir / f"drawing_canvas_{tag}.xml"
        if not xml_path.exists():
            xml_path.write_text(_canvas_xml(half_size, rgba, rim_width, rim_rgba))
        super().__init__(str(xml_path), name=name, joints=joints, obj_type="all", duplicate_collision_geoms=False)
        self.category_name = "drawing_canvas"
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        self.half_size = np.array(half_size)


# ---------------------------------------------------------------------------------------
# Problem class
# ---------------------------------------------------------------------------------------
class Libero_Draw_Tabletop_Manipulation(Libero_Tabletop_Manipulation):
    """LIBERO tabletop scene with a drawing canvas and ManiSkill-style ink dots.

    Extra keyword arguments (all JSON serialisable so they can live in ``env_args``):
        max_dots: number of pre-allocated ink dots (ManiSkill reference: 1500).
        pen_length: length (m) of the visual pen held by the closed fingers (default 0.05;
            0 = draw with the bare fingertips).
        fingertip_offset: distance (m) from ``gripper0_grip_site`` to the closed fingertips.
        pen_tip_offset: override of the pen-tip distance below the grip site
            (default fingertip_offset + pen_length).
        ink_rgba / canvas_rgba: colours.
        robot_init_qpos: "maniskill_rest" (default, panda_stick rest keyframe),
            "libero_default", or an explicit list of 7 joint angles.
        table_length / table_width / table_x_behind_base / table_y_offset / table_top_z:
            table geometry (defaults reproduce ManiSkill's table footprint, top flush with the base).
        camera_offset_xyz: optional extra offset of the agentview camera.
        camera_fovx_deg: agentview horizontal FOV (default = ManiSkill's 84.7 deg); the vertical
            FOV follows the stored frame aspect. camera_fovy_deg overrides the vertical FOV.
        hide_walls / hide_mount_visuals: hide LIBERO-only room walls and the dark pedestal.
        success_coverage_threshold: fraction of pen-down target points that must be covered
            by ink for ``_check_success`` (only used when target waypoints are set).
    """

    def __init__(self, bddl_file_name, *args, **kwargs):
        self.max_dots = int(kwargs.pop("max_dots", 1500))
        # Pen geometry along the gripper approach axis (+z of the eef frame):
        #   grip site --(fingertip_offset)--> closed fingertips --(pen_length)--> pen tip.
        # The closed Panda fingertips end 9.3 mm below gripper0_grip_site (measured from the
        # finger meshes). A 0.05 m pen puts the hand 0.157 m above the canvas, close to the
        # 0.15 m stick of ManiSkill's panda_stick, so the arm configuration matches too.
        self.fingertip_offset = float(kwargs.pop("fingertip_offset", 0.0095))
        self.pen_length = float(kwargs.pop("pen_length", 0.05))
        pen_tip_offset = kwargs.pop("pen_tip_offset", None)
        self.pen_tip_offset = float(pen_tip_offset) if pen_tip_offset is not None else self.fingertip_offset + self.pen_length
        self.pen_radius = float(kwargs.pop("pen_radius", 0.004))
        self.pen_rgba = tuple(float(v) for v in kwargs.pop("pen_rgba", (0.15, 0.15, 0.15, 1.0)))
        self.ink_rgba = tuple(float(v) for v in kwargs.pop("ink_rgba", (0.8, 0.2, 0.2, 1.0)))
        self.canvas_rgba = tuple(float(v) for v in kwargs.pop("canvas_rgba", (1.0, 1.0, 1.0, 1.0)))
        self.robot_init_qpos_mode = kwargs.pop("robot_init_qpos", "maniskill_rest")
        self.success_coverage_threshold = float(kwargs.pop("success_coverage_threshold", 0.9))
        self.touch_tolerance_z = float(kwargs.pop("touch_tolerance_z", traj.TOUCH_TOLERANCE_Z))
        # [pd_ee] arm controller: "osc" (robosuite OSC_POSE deltas, default) or "pd_ee" (see PD_EE_* above)
        self.arm_controller = str(kwargs.pop("arm_controller", "osc"))
        self.pd_ee_kp = float(kwargs.pop("pd_ee_kp", PD_EE_KP))
        self.pd_ee_damping_ratio = float(kwargs.pop("pd_ee_damping_ratio", PD_EE_DAMPING_RATIO))
        self.pd_ee_ik_lambda = float(kwargs.pop("pd_ee_ik_lambda", PD_EE_IK_LAMBDA))
        # [pd_ee] cap the per-step IK joint step at this fraction of the Franka joint velocity limits (0 = off)
        self.pd_ee_max_joint_vel = float(kwargs.pop("pd_ee_max_joint_vel", 0.0))
        # [pd_ee] keep the IK joint goal this far (rad) inside the Panda joint ranges (0 = clip at the range itself)
        self.pd_ee_joint_margin = float(kwargs.pop("pd_ee_joint_margin", 0.0))
        if self.arm_controller == "pd_ee":
            kwargs["controller_configs"] = dict(
                type="JOINT_POSITION", input_max=1, input_min=-1, output_max=PD_EE_OUTPUT_MAX, output_min=-PD_EE_OUTPUT_MAX,
                kp=self.pd_ee_kp, damping_ratio=self.pd_ee_damping_ratio, impedance_mode="fixed", kp_limits=[0, 1e5],
                damping_ratio_limits=[0, 10], qpos_limits=None, interpolation=None, ramp_ratio=0.2)
        elif self.arm_controller != "osc":
            raise ValueError(f"unknown arm_controller {self.arm_controller!r} (osc | pd_ee)")
        # Appearance spec (burger.pipeline.appearance): preset name, JSON string or dict. When given it drives textures,
        # colours, lights, the table extent and the arm LEDs, and overrides the individual scene_* / light_* kwargs below.
        self.appearance = kwargs.pop("appearance", None)
        self.appearance_seed = int(kwargs.pop("appearance_seed", 0))
        # canvas (board) size (x depth, y width): ManiSkill's 0.8 x 1.2 m by default, the lab's 0.6 x 0.9 m with scene preset "real"
        _cs = kwargs.pop("canvas_size", None)
        self.canvas_size = tuple(float(v) for v in _cs) if _cs is not None else (2 * traj.CANVAS_HALF_SIZE[0], 2 * traj.CANVAS_HALF_SIZE[1])
        table_defaults = dict(table_length=DEFAULT_TABLE_LENGTH, table_width=DEFAULT_TABLE_WIDTH, table_x_behind_base=DEFAULT_TABLE_X_BEHIND_BASE, table_y_offset=DEFAULT_TABLE_Y_OFFSET)
        if self.appearance is not None:
            from burger.pipeline import appearance as _A
            if isinstance(self.appearance, str):
                self.appearance = _A.Appearance.from_dict(json.loads(self.appearance)) if self.appearance.strip().startswith("{") else _A.PRESETS[self.appearance]
            elif isinstance(self.appearance, dict):
                self.appearance = _A.Appearance.from_dict(self.appearance)
            self.ink_rgba = tuple(self.appearance.ink_color) + (1.0,)
            self.canvas_rgba = tuple(self.appearance.canvas_color) + (1.0,)
            if self.appearance.table_extent is not None:
                table_defaults.update(_A.table_layout_kwargs(self.appearance, self.canvas_size))
        # Run-level camera preset (burger.pipeline.appearance.CAMERA_PRESETS); None = ManiSkill's base_camera re-anchored to the base
        cam_preset = kwargs.pop("camera_preset", None)
        self.camera_spec = None
        if cam_preset not in (None, "", "none", "maniskill"):
            from burger.pipeline import appearance as _A
            self.camera_spec = _A.resolve_camera(dict(camera_preset=cam_preset, canvas_size=self.canvas_size))
        table_length = float(kwargs.pop("table_length", table_defaults["table_length"]))
        table_width = float(kwargs.pop("table_width", table_defaults["table_width"]))
        table_x_behind_base = float(kwargs.pop("table_x_behind_base", table_defaults["table_x_behind_base"]))
        table_y_offset = float(kwargs.pop("table_y_offset", table_defaults["table_y_offset"]))
        table_top_z = kwargs.pop("table_top_z", None)
        camera_offset_xyz = tuple(float(v) for v in kwargs.pop("camera_offset_xyz", (0.0, 0.0, 0.0)))
        camera_fovy_deg = kwargs.pop("camera_fovy_deg", None)
        camera_fovx_deg = kwargs.pop("camera_fovx_deg", None)
        # Visual tweaks that make the render match ManiSkill's layout (walls/pedestal do not exist there).
        self.hide_walls = bool(kwargs.pop("hide_walls", True))
        self.hide_mount_visuals = bool(kwargs.pop("hide_mount_visuals", True))
        # "maniskill": ManiSkill's wood table texture (1 repeat along the 1.209 m depth, ~3.94 along the 2.418 m width,
        # grain along x like table.glb) and its grey 4 m grid floor on a 100 m plane; "libero": stock LIBERO look
        self.scene_textures = str(kwargs.pop("scene_textures", "maniskill"))
        self.table_texrepeat = tuple(float(v) for v in kwargs.pop("table_texrepeat", (1.0, 3.94)))
        self.floor_texrepeat = tuple(float(v) for v in kwargs.pop("floor_texrepeat", (0.25, 0.25)))   # per metre -> 4 m tiles
        self.table_rgba = tuple(float(v) for v in kwargs.pop("table_rgba", (0.92, 0.92, 0.92, 1.0)))  # tint of the wood texture (matches ManiSkill's table brightness)
        self.floor_rgba = tuple(float(v) for v in kwargs.pop("floor_rgba", (1.0, 1.0, 1.0, 1.0)))
        self.floor_size = float(kwargs.pop("floor_size", 50.0))                                          # plane half size (m)
        # "maniskill": ManiSkill's lights (ambient + directional along (1,1,-1) + one straight down) instead of robosuite's
        # single light from the front-right; intensities tuned so canvas / table / floor brightness match the reference renders
        self.scene_lighting = str(kwargs.pop("scene_lighting", "maniskill"))
        self.light_ambient = float(kwargs.pop("light_ambient", 0.25))
        self.light_diffuse = float(kwargs.pop("light_diffuse", 0.45))
        self.fill_diffuse = float(kwargs.pop("fill_diffuse", 0.25))
        self.light_specular = float(kwargs.pop("light_specular", 0.1))
        self.headlight_diffuse = float(kwargs.pop("headlight_diffuse", 0.15))
        self.headlight_ambient = float(kwargs.pop("headlight_ambient", 0.0))
        # Aspect ratio of the stored agentview frames decides the vertical FOV (horizontal FOV fixed).
        cam_names = kwargs.get("camera_names", ["agentview", "robot0_eye_in_hand"])
        cam_names = [cam_names] if isinstance(cam_names, str) else list(cam_names)
        cam_idx = cam_names.index("agentview") if "agentview" in cam_names else 0
        ch, cw = kwargs.get("camera_heights", 128), kwargs.get("camera_widths", 128)
        ch = ch[cam_idx] if isinstance(ch, (list, tuple)) else ch
        cw = cw[cam_idx] if isinstance(cw, (list, tuple)) else cw
        self.agentview_hw = (int(ch), int(cw))
        self._agentview_crop_x0, self._agentview_crop_w = 0, None
        cam_extra = {}
        if self.camera_spec is not None:
            spec = self.camera_spec
            render_w, x0 = spec.render_geometry(int(cw))
            if render_w != int(cw):
                # MuJoCo cameras have no principal-point offset: render a wider centred frame and crop it (CameraSpec.render_geometry)
                self._agentview_crop_x0, self._agentview_crop_w = int(x0), int(cw)
                widths = kwargs.get("camera_widths", 128)
                widths = list(widths) if isinstance(widths, (list, tuple)) else [int(widths)] * len(cam_names)
                widths[cam_idx] = int(render_w)
                heights = kwargs.get("camera_heights", 128)
                kwargs["camera_widths"] = widths
                kwargs["camera_heights"] = list(heights) if isinstance(heights, (list, tuple)) else [int(heights)] * len(cam_names)
                kwargs["camera_names"] = cam_names
            cam_extra = dict(camera_eye_rel=tuple(np.asarray(spec.eye, dtype=np.float64) - MANISKILL_ROBOT_BASE),
                             camera_target_rel=tuple(np.asarray(spec.target, dtype=np.float64) - MANISKILL_ROBOT_BASE),
                             camera_fovy_rad=spec.fovy_rad(int(cw), int(ch)), camera_fovx_rad=spec.fovx_rad(int(cw)))
        self.layout = compute_scene_layout(
            table_length=table_length,
            table_width=table_width,
            table_x_behind_base=table_x_behind_base,
            table_y_offset=table_y_offset,
            table_top_z=None if table_top_z is None else float(table_top_z),
            camera_fovx_rad=cam_extra.get("camera_fovx_rad", np.radians(float(camera_fovx_deg)) if camera_fovx_deg is not None else MANISKILL_CAMERA_FOVX_RAD),
            camera_aspect=float(cw) / float(ch),
            camera_fovy_rad=cam_extra.get("camera_fovy_rad", np.radians(float(camera_fovy_deg)) if camera_fovy_deg is not None else None),
            camera_offset_xyz=camera_offset_xyz,
            camera_eye_rel=cam_extra.get("camera_eye_rel"),
            camera_target_rel=cam_extra.get("camera_target_rel"),
            canvas_half_size=(self.canvas_size[0] / 2, self.canvas_size[1] / 2),
        )

        # --- replicate Libero_Tabletop_Manipulation.__init__ with our table placement ---
        self.workspace_name = "main_table"
        self.visualization_sites_list = []
        self.table_full_size = tuple(self.layout.table_full_size)
        self.table_offset = tuple(self.layout.workspace_offset)
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update({"robots": [f"Mounted{robot_name}" if not robot_name.startswith("Mounted") else robot_name for robot_name in kwargs["robots"]]})
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})
        kwargs.update({"table_full_size": self.table_full_size})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/libero_tabletop_base_style.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update({"scene_properties": {"floor_style": "light-gray", "wall_style": "light-gray-plaster"}})

        # ink state
        self._ink_geom_ids = np.zeros(0, dtype=np.int64)
        self._ink_count = 0
        self._ink_positions = np.zeros((self.max_dots, 3), dtype=np.float64)
        self._ink_enabled = True
        self._episode_complete = False
        self._target_pen_down_xy = None
        self._covered = np.zeros(0, dtype=bool)
        self._coverage_dots_seen = 0
        self._coverage_radius = 2 * traj.BRUSH_RADIUS
        self._eef_site_id = None
        self._canvas_body_id = None
        # Orientation held during drawing (ManiSkill commands a constant [pi, 0, 0] hand rotation,
        # i.e. the rest orientation). It is captured from the grip-site frame at every reset:
        # robosuite's OSC controls ``gripper0_grip_site`` whose frame is yawed 90 deg w.r.t. the
        # hand frame reported by ``robot0_eef_quat``, so a hand-frame constant would make the
        # wrist rotate 90 deg at the start of each episode.
        self._R_target = T.quat2mat(EEF_DOWN_QUAT_XYZW)

        BDDLBaseDomain.__init__(self, bddl_file_name, *args, **kwargs)

    # ------------------------------------------------------------------ model building
    def _load_fixtures_in_arena(self, mujoco_arena):
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "table":
                continue
            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                if fixture_category == "drawing_canvas":
                    rim_w = float(self.appearance.canvas_rim_width) if self.appearance is not None else 0.0
                    rim_rgba = tuple(self.appearance.canvas_rim_color) + (1.0,) if self.appearance is not None else (0.15, 0.16, 0.16, 1.0)
                    self.fixtures_dict[fixture_instance] = DrawingCanvas(
                        name=fixture_instance, joints=None,
                        half_size=self.layout.canvas_half_size, rgba=self.canvas_rgba, rim_width=rim_w, rim_rgba=rim_rgba,
                    )
                else:
                    self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(name=fixture_instance, joints=None)
        # ink dot pool: geoms of ONE static body (parked below the floor), moved at runtime via
        # model.geom_pos. Using a single body instead of one body per dot keeps MuJoCo's per-body
        # kinematics/inertia loops small (1 vs 1500 extra bodies) without changing the physics.
        ink_body = new_body(name="ink_dots", pos=(0.0, 0.0, 0.0))
        for i in range(self.max_dots):
            ink_body.append(
                new_geom(
                    name=f"ink_dot_{i}_geom", type="cylinder",
                    size=[traj.BRUSH_RADIUS, traj.DOT_THICKNESS / 2.0], pos=(0.0, 0.0, -1.0),
                    group=1, rgba=array_to_string((*self.ink_rgba[:3], 0.0)), contype=0, conaffinity=0,
                )
            )
        mujoco_arena.worldbody.append(ink_body)

    def _setup_camera(self, mujoco_arena):
        super()._setup_camera(mujoco_arena)
        # Keep LIBERO's canonical viewpoint available under another name for reference.
        mujoco_arena.set_camera(
            camera_name="libero_agentview",
            pos=[0.6586131746834771, 0.0, 1.6103500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )
        # ManiSkill base_camera viewpoint, re-anchored to the LIBERO robot base.
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=list(self.layout.camera_eye_xyz),
            quat=list(self.layout.camera_quat_wxyz),
            camera_attribs={"fovy": f"{self.layout.camera_fovy_deg:.6f}"},
        )

    def _load_model(self):
        super()._load_model()
        # Robot base check: LIBERO computes the base from the table length + mount height.
        robot_model = self.robots[0].robot_model
        xpos = np.array(robot_model.base_xpos_offset["table"](self.table_full_size[0]), dtype=np.float64)
        base = xpos - np.asarray(robot_model.bottom_offset, dtype=np.float64)
        if not np.allclose(base, self.layout.robot_base_xyz, atol=1e-6):
            print(f"[burger.backends.libero] WARNING robot base mismatch: model {base} vs layout {self.layout.robot_base_xyz}")
        # Hide LIBERO-only visuals: room walls and the dark Rethink pedestal, whose top pokes
        # ~1 cm above the table around the robot base (the "black ring" in renders).
        if self.hide_walls or self.hide_mount_visuals:
            for geom in self.model.worldbody.iter("geom"):
                name = geom.get("name") or ""
                if (self.hide_walls and name.startswith("wall_")) or (
                    self.hide_mount_visuals and name.startswith("mount") and name.endswith("_vis")
                ):
                    geom.set("rgba", "0 0 0 0")
        if self.appearance is not None:
            self._apply_appearance(self.appearance)
        else:
            if self.scene_textures == "maniskill":
                self._apply_maniskill_textures()
            elif self.scene_textures != "libero":
                raise ValueError(f"unknown scene_textures {self.scene_textures!r} (maniskill | libero)")
            if self.scene_lighting == "maniskill":
                self._apply_maniskill_lighting()
            elif self.scene_lighting != "libero":
                raise ValueError(f"unknown scene_lighting {self.scene_lighting!r} (maniskill | libero)")
        # Visual pen held by the closed fingers (no collision; ink is placed kinematically).
        if self.pen_length > 0:
            prefix = self.robots[0].gripper.naming_prefix
            eef_body = self.model.worldbody.find(f".//body[@name='{prefix}eef']")
            if eef_body is None:
                raise RuntimeError(f"gripper eef body '{prefix}eef' not found for the pen attachment")
            z0 = max(self.fingertip_offset - 0.02, 0.0)  # start inside the finger pads
            z1 = self.pen_tip_offset
            eef_body.append(
                new_geom(
                    name=f"{prefix}pen_visual", type="cylinder",
                    size=[self.pen_radius, (z1 - z0) / 2.0], pos=(0.0, 0.0, (z0 + z1) / 2.0),
                    group=1, rgba=array_to_string(self.pen_rgba), contype=0, conaffinity=0, mass=0.0,
                )
            )
            if self.appearance is not None and getattr(self.appearance, "pen_style", None) == "marker":
                # real marker look (burger.pipeline.appearance.MARKER_PARTS, expressed here relative to the pen tip): a black
                # box around the finger pads + an 18 mm marker body; the thin pen_visual keeps the last mm of the tip
                from burger.pipeline import appearance as _A
                rgba = array_to_string(tuple(_A.MARKER_COLOR) + (1.0,))
                for j, (shape, centre, size) in enumerate(_A.MARKER_PARTS):
                    zc = z1 - (0.15 - float(centre[2]))       # hand-frame z (tip at 0.15) -> eef-frame z (tip at z1)
                    if shape == "box":
                        # the robosuite eef frame's x is the finger-opening direction (panda_hand's y): swap the half extents
                        eef_body.append(new_geom(name=f"{prefix}pen_marker_{j}", type="box", size=[float(size[1]), float(size[0]), float(size[2])], pos=(0.0, 0.0, zc), group=1, rgba=rgba, contype=0, conaffinity=0, mass=0.0))
                    else:
                        eef_body.append(new_geom(name=f"{prefix}pen_marker_{j}", type="cylinder", size=[float(size[0]), float(size[1])], pos=(0.0, 0.0, zc), group=1, rgba=rgba, contype=0, conaffinity=0, mass=0.0))
        # Initial arm configuration (ManiSkill panda_stick rest keyframe by default).
        if isinstance(self.robot_init_qpos_mode, str):
            if self.robot_init_qpos_mode == "maniskill_rest":
                self.robots[0].init_qpos = np.array(PANDA_STICK_REST_QPOS)
            elif self.robot_init_qpos_mode != "libero_default":
                raise ValueError(f"unknown robot_init_qpos mode {self.robot_init_qpos_mode!r}")
        else:
            self.robots[0].init_qpos = np.array(self.robot_init_qpos_mode, dtype=np.float64)

    def _apply_maniskill_textures(self):
        """Give the LIBERO table and floor the ManiSkill look: table.glb's wood texture on the table top (2d texture,
        u = grain along the table's x axis) and ManiSkill's grey grid on a large floor plane (4 m tiles)."""
        tex_dir = ASSET_DIR / "textures"
        wood = tex_dir / "maniskill_table_wood.png"   # MuJoCo 2.3 loads PNG only
        grid = tex_dir / "maniskill_grid.png"
        if not wood.exists() or not grid.exists():
            raise FileNotFoundError(f"ManiSkill textures missing under {tex_dir}")
        asset = self.model.asset
        for name in ("tex-ms-wood", "tex-ms-grid"):
            for el in asset.findall(f"./texture[@name='{name}']"):
                asset.remove(el)
        for name in ("mat-ms-wood", "mat-ms-grid"):
            for el in asset.findall(f"./material[@name='{name}']"):
                asset.remove(el)
        asset.append(new_element("texture", name="tex-ms-wood", type="2d", file=str(wood)))
        asset.append(new_element("material", name="mat-ms-wood", texture="tex-ms-wood", texrepeat=array_to_string(self.table_texrepeat),
                                 texuniform="false", reflectance="0", shininess="0", specular="0.1", rgba=array_to_string(self.table_rgba)))
        asset.append(new_element("texture", name="tex-ms-grid", type="2d", file=str(grid)))
        asset.append(new_element("material", name="mat-ms-grid", texture="tex-ms-grid", texrepeat=array_to_string(self.floor_texrepeat),
                                 texuniform="true", reflectance="0", shininess="0", specular="0", rgba=array_to_string(self.floor_rgba)))
        table_visual = self.model.worldbody.find(".//geom[@name='table_visual']")
        if table_visual is None:
            raise RuntimeError("table_visual geom not found")
        table_visual.set("material", "mat-ms-wood")
        floor = self.model.worldbody.find(".//geom[@name='floor']")
        if floor is not None:
            floor.set("material", "mat-ms-grid")
            floor.set("size", array_to_string([self.floor_size, self.floor_size, 0.125]))
        for leg in self.model.worldbody.iter("geom"):
            if (leg.get("name") or "").startswith("table_leg"):
                leg.set("material", "mat-ms-wood")

    # LIBERO light units relative to ManiSkill's (calibrated on the reference renders: ManiSkill ambient 0.3 + two
    # directional lights of 1.0 <-> MuJoCo ambient 0.25, key diffuse 0.45, fill diffuse 0.25, headlight 0.15)
    LIGHT_GAIN_AMBIENT, LIGHT_GAIN_KEY, LIGHT_GAIN_FILL, LIGHT_HEADLIGHT = 0.25 / 0.3, 0.45, 0.25, 0.15

    def _set_material_texture(self, tex_name, mat_name, png, texrepeat, texuniform, rgba, specular="0.1"):
        asset = self.model.asset
        for el in asset.findall(f"./texture[@name='{tex_name}']") + asset.findall(f"./material[@name='{mat_name}']"):
            asset.remove(el)
        if png is None:
            asset.append(new_element("texture", name=tex_name, type="2d", builtin="flat", rgb1="1 1 1", rgb2="1 1 1", width="8", height="8"))
        else:
            asset.append(new_element("texture", name=tex_name, type="2d", file=str(png)))
        asset.append(new_element("material", name=mat_name, texture=tex_name, texrepeat=array_to_string(texrepeat), texuniform="true" if texuniform else "false",
                                 reflectance="0", shininess="0", specular=specular, rgba=array_to_string(rgba)))

    def _apply_appearance(self, app):
        """Apply a ``burger.pipeline.appearance.Appearance``: table/floor textures & colours, background, lights."""
        from burger.pipeline import appearance as _A

        table_png = _A.table_texture_png(app, (self.layout.table_full_size[0], self.layout.table_full_size[1]), seed=self.appearance_seed)
        floor_png = _A.floor_texture_png(app, seed=self.appearance_seed)
        # textures are pre-coloured; a flat white texture + rgba colour is used when there is no pattern
        table_rgba = (1.0, 1.0, 1.0, 1.0) if table_png is not None else tuple(app.table_color) + (1.0,)
        floor_rgba = (1.0, 1.0, 1.0, 1.0) if floor_png is not None else tuple(app.floor_color) + (1.0,)
        self._set_material_texture("tex-app-table", "mat-app-table", table_png, (1.0, 1.0), False, table_rgba, specular=f"{0.3 * (1 - app.table_roughness):.3f}")
        self._set_material_texture("tex-app-floor", "mat-app-floor", floor_png, (app.floor_texture_scale, app.floor_texture_scale), True, floor_rgba, specular="0")
        wb = self.model.worldbody
        table_visual = wb.find(".//geom[@name='table_visual']")
        if table_visual is None:
            raise RuntimeError("table_visual geom not found")
        table_visual.set("material", "mat-app-table")
        apron_rgba = tuple(app.table_apron_color) + (1.0,) if getattr(app, "table_apron_color", None) else tuple(app.table_color) + (1.0,)
        for leg in wb.iter("geom"):
            if (leg.get("name") or "").startswith("table_leg"):
                leg.set("material", ""); leg.set("rgba", array_to_string(apron_rgba))
        # the real table shows a dark edge below a thin green top: a slightly larger dark box hides the lower 4 cm of the top's sides
        table_body = table_visual.getparent() if hasattr(table_visual, "getparent") else None
        if table_body is None:
            for body in wb.iter("body"):
                if table_visual in list(body):
                    table_body = body; break
        if table_body is not None and table_body.find("./geom[@name='app_table_apron']") is None:
            tv_pos = np.array([float(v) for v in (table_visual.get("pos") or "0 0 0").split()])
            tv_size = np.array([float(v) for v in table_visual.get("size").split()])
            table_body.append(new_geom(name="app_table_apron", type="box", size=[tv_size[0] + 0.001, tv_size[1] + 0.001, max(tv_size[2] - 0.005, 0.001)],
                                       pos=tuple(tv_pos + np.array([0.0, 0.0, -0.005])), group=1, rgba=apron_rgba, contype=0, conaffinity=0))
        floor = wb.find(".//geom[@name='floor']")
        if floor is not None:
            floor.set("material", "mat-app-floor")
            floor.set("size", array_to_string([self.floor_size, self.floor_size, 0.125]))
        # background: robosuite's gradient skybox -> flat colour
        for tex in self.model.asset.findall("./texture[@type='skybox']"):
            tex.set("builtin", "flat"); tex.set("rgb1", array_to_string(app.background_color)); tex.set("rgb2", array_to_string(app.background_color))
        # status LEDs on the arm: emissive spheres on the robot links (appearance.LED_MARKERS)
        if app.led_color:
            col = [min(1.0, float(c) * float(app.led_intensity)) for c in app.led_color]
            if self.model.asset.find("./material[@name='mat-app-led']") is None:
                self.model.asset.append(new_element("material", name="mat-app-led", rgba=array_to_string(col + [1.0]), emission="1", specular="0", shininess="0"))
            for j, (link_name, offset, shape, size) in enumerate(_A.LED_MARKERS):
                body = wb.find(f".//body[@name='robot0_{link_name.replace('panda_', '')}']")
                if body is None or body.find(f"./geom[@name='app_led_{j}']") is not None:
                    continue
                geom_size = [float(size)] if shape == "sphere" else [float(v) for v in size]   # box: half extents; capsule/cylinder: (r, half_length) along z
                body.append(new_geom(name=f"app_led_{j}", type=shape, size=geom_size, pos=tuple(float(v) for v in offset), group=1,
                                     material="mat-app-led", contype=0, conaffinity=0))
        # lights
        ex = float(app.exposure)
        key = [c * self.LIGHT_GAIN_KEY * app.key_intensity * ex for c in app.key_color]
        fill = [c * self.LIGHT_GAIN_FILL * app.fill_intensity * ex for c in app.fill_color]
        amb = [self.LIGHT_GAIN_AMBIENT * app.ambient * ex] * 3
        kd = np.asarray(app.key_dir, dtype=np.float64); kd /= np.linalg.norm(kd)
        fd = np.asarray(app.fill_dir, dtype=np.float64); fd /= np.linalg.norm(fd)
        lights = list(wb.iter("light"))
        sun = lights[0] if lights else new_element("light", name="app_key")
        if not lights:
            wb.append(sun)
        sun.set("name", sun.get("name") or "app_key"); sun.set("directional", "true"); sun.set("castshadow", "true" if app.shadows else "false")
        sun.set("dir", array_to_string(kd)); sun.set("pos", array_to_string(-2.0 * kd + np.array([0, 0, 0.9])))
        # MuJoCo attenuates a shadow-casting light's *ambient* term inside its shadow too, so the ambient light rides on the
        # (never shadowing) fill light instead of the key light
        sun.set("diffuse", array_to_string(key)); sun.set("ambient", "0 0 0"); sun.set("specular", array_to_string([0.1 * app.key_intensity] * 3))
        for extra in lights[1:]:
            extra.set("diffuse", "0 0 0"); extra.set("ambient", "0 0 0"); extra.set("specular", "0 0 0"); extra.set("castshadow", "false")
        key2_el = wb.find("./light[@name='app_key2']")
        if getattr(app, "key2_dir", None) is not None and float(getattr(app, "key2_intensity", 0.0)) > 0:
            # second lamp on the other side of the arm: two overlapping (hard) shadows lighten each other like the real soft shadow
            k2 = np.asarray(app.key2_dir, dtype=np.float64); k2 /= np.linalg.norm(k2)
            key2 = [c * self.LIGHT_GAIN_KEY * app.key2_intensity * ex for c in app.key2_color]
            if key2_el is None:
                key2_el = new_element("light", name="app_key2", directional="true"); wb.append(key2_el)
            key2_el.set("castshadow", "true" if app.shadows else "false"); key2_el.set("dir", array_to_string(k2)); key2_el.set("pos", array_to_string(-2.0 * k2 + np.array([0, 0, 0.9])))
            key2_el.set("diffuse", array_to_string(key2)); key2_el.set("ambient", "0 0 0"); key2_el.set("specular", array_to_string([0.1 * app.key2_intensity] * 3))
        elif key2_el is not None:
            key2_el.set("diffuse", "0 0 0"); key2_el.set("castshadow", "false")
        fill_el = wb.find("./light[@name='app_fill']")
        if fill_el is None:
            fill_el = new_element("light", name="app_fill", directional="true"); wb.append(fill_el)
        fill_el.set("castshadow", "false"); fill_el.set("pos", array_to_string(-2.0 * fd + np.array([0, 0, 0.9]))); fill_el.set("dir", array_to_string(fd))
        fill_el.set("diffuse", array_to_string(fill)); fill_el.set("ambient", array_to_string(amb)); fill_el.set("specular", "0 0 0")
        visual = self.model.root.find("visual")
        if visual is None:
            visual = ET.SubElement(self.model.root, "visual")
        headlight = visual.find("headlight")
        if headlight is None:
            headlight = ET.SubElement(visual, "headlight")
        headlight.set("diffuse", array_to_string([self.LIGHT_HEADLIGHT * ex] * 3)); headlight.set("ambient", "0 0 0"); headlight.set("specular", "0 0 0")
        if app.shadows:
            quality = visual.find("quality")
            if quality is None:
                quality = ET.SubElement(visual, "quality")
            quality.set("shadowsize", "4096")
            mp = visual.find("map")
            if mp is None:
                mp = ET.SubElement(visual, "map")
            mp.set("shadowclip", "3.0"); mp.set("shadowscale", "0.9")
            # MuJoCo sizes a directional light's shadow map from the model extent; with the 100 m floor that gives blocky
            # shadows, so pin the statistic to the work area (2 m around the canvas)
            stat = self.model.root.find("statistic")
            if stat is None:
                stat = ET.SubElement(self.model.root, "statistic")
            cx, cy, cz = self.layout.canvas_center_xyz
            stat.set("center", array_to_string([cx, cy, cz + 0.3])); stat.set("extent", "2.2")

    def _apply_maniskill_lighting(self):
        """ManiSkill's default lighting: ambient 0.3, a directional light along (1, 1, -1) and one straight down, no shadows.
        The robosuite arena light is re-pointed and a second light is added; the camera headlight is dimmed."""
        d = 1.0 / np.sqrt(3.0)
        lights = list(self.model.worldbody.iter("light"))
        a = array_to_string([self.light_ambient] * 3)
        sun = lights[0] if lights else None
        if sun is None:
            sun = new_element("light", name="ms_sun", directional="true", castshadow="false")
            self.model.worldbody.append(sun)
        sun.set("dir", array_to_string([d, d, -d])); sun.set("pos", "-1 -1 1.5"); sun.set("directional", "true"); sun.set("castshadow", "false")
        sun.set("diffuse", array_to_string([self.light_diffuse] * 3)); sun.set("ambient", a); sun.set("specular", array_to_string([self.light_specular] * 3))
        for extra in lights[1:]:
            extra.set("diffuse", "0 0 0"); extra.set("ambient", "0 0 0"); extra.set("specular", "0 0 0")
        if self.model.worldbody.find("./light[@name='ms_fill']") is None:
            self.model.worldbody.append(new_element("light", name="ms_fill", directional="true", castshadow="false", pos="0 0 2", dir="0 0 -1",
                                                    diffuse=array_to_string([self.fill_diffuse] * 3), ambient="0 0 0", specular="0 0 0"))
        visual = self.model.root.find("visual")
        if visual is None:
            visual = new_element("visual", name=None) if False else ET.SubElement(self.model.root, "visual")
        headlight = visual.find("headlight")
        if headlight is None:
            headlight = ET.SubElement(visual, "headlight")
        headlight.set("diffuse", array_to_string([self.headlight_diffuse] * 3))
        headlight.set("ambient", array_to_string([self.headlight_ambient] * 3))
        headlight.set("specular", "0 0 0")

    def _setup_references(self):
        super()._setup_references()
        self._eef_site_id = self.sim.model.site_name2id(f"{self.robots[0].gripper.naming_prefix}grip_site")
        self._ink_geom_ids = np.array([self.sim.model.geom_name2id(f"ink_dot_{i}_geom") for i in range(self.max_dots)], dtype=np.int64)
        canvas_names = [n for n in self.fixtures_dict if isinstance(self.fixtures_dict[n], DrawingCanvas)]
        if len(canvas_names) != 1:
            raise RuntimeError(f"expected exactly one drawing_canvas fixture, found {canvas_names}")
        self.canvas_name = canvas_names[0]
        self._canvas_body_id = self.sim.model.body_name2id(self.fixtures_dict[self.canvas_name].root_body)
        self._gripper_qpos_idx = self.robots[0]._ref_gripper_joint_pos_indexes

    # ------------------------------------------------------------------ resets
    def _reset_internal(self):
        super()._reset_internal()
        # Exact canvas placement (LIBERO's fixture sampler only sets xy on the table plane).
        self.sim.model.body_pos[self._canvas_body_id] = np.array(self.layout.canvas_center_xyz)
        self.sim.model.body_quat[self._canvas_body_id] = np.array([1.0, 0.0, 0.0, 0.0])
        # Start with a closed gripper (the "pen").
        self.sim.data.qpos[self._gripper_qpos_idx] = 0.0
        self.robots[0].gripper.current_action = np.array([-1.0, 1.0])
        # Hide all ink dots and reset the ink state.
        self.sim.model.geom_pos[self._ink_geom_ids] = (0.0, 0.0, -1.0)
        # Invisible pooled dots otherwise still incur GL draw/shadow passes.
        self.sim.model.geom_rgba[self._ink_geom_ids, 3] = 0.0
        self._ink_count = 0
        self._ink_positions[:] = 0.0
        self._episode_complete = False
        if self._target_pen_down_xy is not None:
            self._covered = np.zeros(len(self._target_pen_down_xy), dtype=bool)
            self._coverage_dots_seen = 0
        self.sim.forward()
        # Hold the reset orientation of the controlled (grip-site) frame for the whole episode.
        self._R_target = np.array(self.sim.data.site_xmat[self._eef_site_id]).reshape(3, 3).copy()
        self.robots[0].controller.reset_goal()
        self._enlarge_render_scene()

    def _enlarge_render_scene(self):
        """robosuite allocates ``MjvScene(maxgeom=1000)`` for offscreen rendering; with the ink
        dot pool the model has more geoms than that and the *last* ones (robot, canvas) would
        silently disappear from the rendered images. Grow the scene buffer to fit the model."""
        ctx = getattr(self.sim, "_render_context_offscreen", None)
        if ctx is None:
            return
        needed = int(self.sim.model.ngeom) + 200
        if ctx.scn.maxgeom < needed:
            ctx.scn = mujoco.MjvScene(self.sim.model._model, maxgeom=needed)

    # ------------------------------------------------------------------ faster control loop
    def step(self, action):
        """robosuite's ``MujocoEnv.step`` without the redundant per-substep ``sim.forward()``.

        robosuite calls ``mj_forward`` at the top of every physics substep *and* again inside the
        OSC controller's ``update()`` (its ``new_update`` flag is re-armed by ``run_controller``), so
        every substep recomputed identical kinematics/dynamics twice. The controller's forward is the
        one that feeds the torque computation, and observables are still sampled after ``mj_step``
        exactly as before, so dropping the loop-level call leaves the simulation bit-identical while
        saving 25 ``mj_forward`` calls (~6 ms) per control step. ``done`` follows LIBERO's
        ``BDDLBaseDomain.step`` semantics (success check)."""
        if self.action_dim == 4 and len(action) > 4:  # LIBERO OSC_POSITION compatibility
            action = np.array(action)
            action = np.concatenate((action[:3], action[-1:]), axis=-1)
        if self.done:
            raise ValueError("executing action in terminated episode")
        self.timestep += 1
        policy_step = True
        for _ in range(int(self.control_timestep / self.model_timestep)):
            self._pre_action(action, policy_step)
            self.sim.step()
            self._update_observables()
            policy_step = False
        self.cur_time += self.control_timestep
        reward, done, info = self._post_action(action)
        if self.viewer is not None and self.renderer != "mujoco":
            self.viewer.update()
        observations = self.viewer._get_observations() if self.viewer_get_obs else self._get_observations()
        done = self._check_success()
        return observations, reward, done, info

    # ------------------------------------------------------------------ ink mechanism
    def pen_tip_pos(self) -> np.ndarray:
        eef_pos = self.sim.data.site_xpos[self._eef_site_id]
        rot = self.sim.data.site_xmat[self._eef_site_id].reshape(3, 3)
        return eef_pos + rot @ np.array([0.0, 0.0, self.pen_tip_offset])

    def eef_pos_rot(self):
        return (
            np.array(self.sim.data.site_xpos[self._eef_site_id]),
            np.array(self.sim.data.site_xmat[self._eef_site_id]).reshape(3, 3),
        )

    @property
    def canvas_top_z(self) -> float:
        return self.layout.canvas_top_z

    @property
    def touch_z_threshold(self) -> float:
        """World z below which the pen tip counts as touching (ManiSkill: 0.02+0.003+0.012)."""
        return self.canvas_top_z + traj.DOT_THICKNESS + self.touch_tolerance_z

    def pen_touching(self) -> bool:
        return bool(self.pen_tip_pos()[2] < self.touch_z_threshold)

    def _place_ink_if_touching(self):
        if not self._ink_enabled:
            return False
        tip = self.pen_tip_pos()
        if tip[2] >= self.touch_z_threshold:
            return False
        if self._ink_count >= self.max_dots:
            raise RuntimeError(f"Ink capacity {self.max_dots} exceeded; increase backend_option max_dots instead of exporting a truncated drawing")
        pos = np.array([tip[0], tip[1], self.canvas_top_z + traj.DOT_THICKNESS / 2.0])
        self.sim.model.geom_pos[self._ink_geom_ids[self._ink_count]] = pos
        self.sim.model.geom_rgba[self._ink_geom_ids[self._ink_count], 3] = self.ink_rgba[3]
        self._ink_positions[self._ink_count] = pos
        self._ink_count += 1
        # make the new dot visible in this step's rendered observation
        mujoco.mj_kinematics(self.sim.model._model, self.sim.data._data)
        return True

    def _post_process(self):
        super()._post_process()
        self._place_ink_if_touching()

    @property
    def ink_count(self) -> int:
        return self._ink_count

    def ink_positions(self) -> np.ndarray:
        return self._ink_positions[: self._ink_count].copy()

    def set_ink_enabled(self, enabled: bool):
        self._ink_enabled = bool(enabled)

    # ------------------------------------------------------------------ targets / success
    def set_target_waypoints(self, waypoints_world: np.ndarray):
        """Register the intended drawing so ``_check_success`` can measure ink coverage."""
        wps = np.asarray(waypoints_world, dtype=np.float64)
        down = wps[:, 2] < self.touch_z_threshold
        self._target_pen_down_xy = wps[down, :2].copy()
        self._covered = np.zeros(len(self._target_pen_down_xy), dtype=bool)
        self._coverage_dots_seen = 0
        self._coverage_radius = 2 * traj.BRUSH_RADIUS

    def ink_coverage(self, radius: float = 2 * traj.BRUSH_RADIUS) -> float:
        """Fraction of pen-down target points that have an ink dot within ``radius``.

        Incremental: only dots placed since the previous call are tested against the targets
        (O(new dots x targets) per step instead of O(all dots x targets), which used to cost
        ~40 ms per control step late in an episode)."""
        if self._target_pen_down_xy is None or len(self._target_pen_down_xy) == 0:
            return 0.0
        if radius != self._coverage_radius:
            self._coverage_radius = radius
            self._covered[:] = False
            self._coverage_dots_seen = 0
        if self._coverage_dots_seen < self._ink_count:
            new_ink = self._ink_positions[self._coverage_dots_seen:self._ink_count, :2]
            d2 = ((self._target_pen_down_xy[:, None, :] - new_ink[None, :, :]) ** 2).sum(-1)
            self._covered |= d2.min(axis=1) <= radius**2
            self._coverage_dots_seen = self._ink_count
        return float(self._covered.mean())

    def mark_episode_complete(self, complete: bool = True):
        self._episode_complete = bool(complete)

    def _check_success(self):
        if self._episode_complete:
            return True
        if self._target_pen_down_xy is None:
            return False
        return self.ink_coverage() >= self.success_coverage_threshold

    # ------------------------------------------------------------------ action helpers
    def target_eef_from_pen_tip(self, tip_world: np.ndarray) -> np.ndarray:
        return np.asarray(tip_world, dtype=np.float64) - self._R_target @ np.array([0.0, 0.0, self.pen_tip_offset])

    def waypoint_action(self, tip_target_world, gain_pos: float = 1.0, gain_rot: float = 1.0, gripper: float = 1.0) -> np.ndarray:
        """LIBERO OSC_POSE delta action that drives the pen tip toward ``tip_target_world``
        while holding the reset (down-facing, camera-parallel) orientation, like ManiSkill's
        constant [pi, 0, 0] rotation target: the wrist does not rotate during the episode."""
        eef_pos, rot = self.eef_pos_rot()
        dpos = (self.target_eef_from_pen_tip(tip_target_world) - eef_pos) * gain_pos
        drot = T.quat2axisangle(T.mat2quat(self._R_target @ rot.T)) * gain_rot
        return np.concatenate([
            np.clip(dpos / OSC_POS_SCALE, -1.0, 1.0),
            np.clip(drot / OSC_ROT_SCALE, -1.0, 1.0),
            [float(gripper)],
        ])

    def pd_ee_action(self, tip_target_world, gripper: float = 1.0):
        """[pd_ee] JOINT_POSITION action for an absolute pen-tip target (ManiSkill pd_ee_pose analogue): one damped
        least-squares step q + (J^T J + lambda I)^-1 J^T e from the current joints, where e is the grip-site position
        error to ``target_eef_from_pen_tip(tip)`` and the axis-angle error to the held reset orientation. The goal is
        clipped to the Panda joint ranges. Returns (action, goal_clipped)."""
        robot = self.robots[0]
        self.sim.forward()          # refresh kinematics after the last substep (state unchanged; mj_step recomputes it anyway)
        m, d = self.sim.model._model, self.sim.data._data
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self._eef_site_id)
        vidx = robot._ref_joint_vel_indexes
        J = np.vstack([jacp[:, vidx], jacr[:, vidx]])
        eef_pos, rot = self.eef_pos_rot()
        err = np.concatenate([self.target_eef_from_pen_tip(tip_target_world) - eef_pos,
                              T.quat2axisangle(T.mat2quat(self._R_target @ rot.T))])
        dq = np.linalg.solve(J.T @ J + self.pd_ee_ik_lambda * np.eye(J.shape[1]), J.T @ err)
        if self.pd_ee_max_joint_vel > 0:
            # optional Franka joint-velocity cap: scale the whole step (keeps the Cartesian direction)
            lim = self.pd_ee_max_joint_vel * FRANKA_JOINT_VEL_LIMIT * self.control_timestep
            dq = dq * min(1.0, float(np.min(lim / np.maximum(np.abs(dq), 1e-12))))
        q = np.array(d.qpos[robot._ref_joint_pos_indexes])
        rng = m.jnt_range[robot._ref_joint_indexes]
        goal = np.clip(q + dq, rng[:, 0] + self.pd_ee_joint_margin, rng[:, 1] - self.pd_ee_joint_margin)  # [pd_ee] joint margin
        a = (goal - q) / PD_EE_OUTPUT_MAX
        clipped = bool(np.any(goal != q + dq) or np.any(np.abs(a) > 1.0))
        return np.concatenate([np.clip(a, -1.0, 1.0), [float(gripper)]]), clipped

    def teleport_pen_tip(self, tip_target_world, iters: int = 80, tol: float = 1e-3):
        """[pd_ee tcp_init] reset-time IK teleport (ManiSkill ``_maybe_randomize_tcp_init`` analogue): iterate damped
        least squares on the arm qpos (kinematics only, no dynamics) until the pen tip reaches ``tip_target_world``
        with the held reset orientation, then set the joints there at rest. Returns the final tip error (m); the
        caller decides whether to accept it (the joints are restored to the start pose when the error > ``tol``)."""
        robot = self.robots[0]
        m, d = self.sim.model._model, self.sim.data._data
        pidx, vidx = robot._ref_joint_pos_indexes, robot._ref_joint_vel_indexes
        rng = m.jnt_range[robot._ref_joint_indexes]
        q0 = np.array(d.qpos[pidx])
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        err_n = np.inf
        for _ in range(int(iters)):
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            eef_pos, rot = self.eef_pos_rot()
            err = np.concatenate([self.target_eef_from_pen_tip(tip_target_world) - eef_pos,
                                  T.quat2axisangle(T.mat2quat(self._R_target @ rot.T))])
            err_n = float(np.linalg.norm(self.pen_tip_pos() - np.asarray(tip_target_world)))
            if err_n < tol * 0.1:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self._eef_site_id)
            J = np.vstack([jacp[:, vidx], jacr[:, vidx]])
            dq = np.linalg.solve(J.T @ J + self.pd_ee_ik_lambda * np.eye(J.shape[1]), J.T @ err)
            d.qpos[pidx] = np.clip(d.qpos[pidx] + dq, rng[:, 0] + self.pd_ee_joint_margin, rng[:, 1] - self.pd_ee_joint_margin)
        mujoco.mj_kinematics(m, d)
        err_n = float(np.linalg.norm(self.pen_tip_pos() - np.asarray(tip_target_world)))
        if err_n > tol:
            d.qpos[pidx] = q0
        d.qvel[vidx] = 0.0
        self.sim.forward()
        robot.controller.update(force=True); robot.controller.reset_goal()
        return err_n

    # ------------------------------------------------------------------ rendering helpers
    def _get_observations(self, force_update=False):
        obs = super()._get_observations(force_update=force_update)
        if self._agentview_crop_w is not None and "agentview_image" in obs:
            x0 = self._agentview_crop_x0
            obs["agentview_image"] = np.ascontiguousarray(obs["agentview_image"][:, x0:x0 + self._agentview_crop_w])
        return obs

    def render_frame(self, camera_name: str = "agentview", width: int = 320, height: int = 240, match_fovx: bool = True) -> np.ndarray:
        """Render an extra frame (same image convention as the observations: OpenGL, i.e. the
        row order is flipped relative to a photo; use ``[::-1]`` to display).

        With ``match_fovx`` the agentview keeps ManiSkill's horizontal FOV for the requested
        aspect ratio, so a 320x240 render has exactly ManiSkill's intrinsics regardless of the
        stored frame size."""
        cam_id = self.sim.model.camera_name2id(camera_name)
        old_fovy = float(self.sim.model.cam_fovy[cam_id])
        render_w, x0 = int(width), 0
        if camera_name == "agentview" and self.camera_spec is not None:
            # camera preset: same relative intrinsics at any size; off-centre principal point via a wider render + crop
            render_w, x0 = self.camera_spec.render_geometry(int(width))
            self.sim.model.cam_fovy[cam_id] = float(np.degrees(self.camera_spec.fovy_rad(int(width), int(height))))
        elif match_fovx and camera_name == "agentview":
            self.sim.model.cam_fovy[cam_id] = self.layout.fovy_deg_for_aspect(width / height)
        try:
            img = self.sim.render(width=render_w, height=height, camera_name=camera_name)
            return img if render_w == int(width) else np.ascontiguousarray(img[:, x0:x0 + int(width)])
        finally:
            self.sim.model.cam_fovy[cam_id] = old_fovy

    def scene_info(self) -> dict:
        eef_pos, _ = self.eef_pos_rot()
        info = self.layout.to_dict()
        info.update(
            dict(
                robot_base_measured=self.sim.data.get_body_xpos("robot0_base").tolist(),
                canvas_body_pos=self.sim.model.body_pos[self._canvas_body_id].tolist(),
                pen_tip_offset=self.pen_tip_offset,
                pen_length=self.pen_length,
                fingertip_offset=self.fingertip_offset,
                touch_z_threshold=self.touch_z_threshold,
                render_scene_maxgeom=int(self.sim._render_context_offscreen.scn.maxgeom) if getattr(self.sim, "_render_context_offscreen", None) else None,
                agentview_hw=list(self.agentview_hw),
                hide_walls=self.hide_walls,
                hide_mount_visuals=self.hide_mount_visuals,
                model_ngeom=int(self.sim.model.ngeom),
                max_dots=self.max_dots,
                eef_pos=eef_pos.tolist(),
                pen_tip_pos=self.pen_tip_pos().tolist(),
                init_qpos=np.asarray(self.robots[0].init_qpos).tolist(),
            )
        )
        return info


# register with LIBERO (the decorator form would rebind the class name to None)
register_problem(Libero_Draw_Tabletop_Manipulation)
