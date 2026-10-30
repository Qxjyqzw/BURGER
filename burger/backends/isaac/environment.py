"""Isaac Sim (Isaac Lab 3.x, PhysX + RTX) port of ManiSkill's ``DrawSVGParallel-v1`` drawing scene.

What is reproduced from ``painter-main/.../maniskill_environments/draw_svg_parallel.py``:

* the same robot: ManiSkill's ``panda_stick`` URDF (Franka Panda arm + 10 cm stick, TCP 0.15 m below the
  hand frame) imported through Isaac Sim's URDF importer, base at ``(-0.615, 0, 0)``, rest pose
  ``[0, pi/8, 0, -5pi/8, 0, 3pi/4, pi/4]``, PhysX joint drives with ManiSkill's gains (1000 / 100 / 100 N m);
* the same controller: absolute ``pd_ee_pose`` targets in the root frame, converted to joint targets with one
  Levenberg-Marquardt step ``dq = (J^T J + 1e-4 I)^-1 J^T dx`` per control step (what ManiSkill's
  ``Kinematics.compute_ik`` does on the GPU), 5 physics sub-steps of 10 ms per 20 Hz control step;
* the same scene: ManiSkill's ``table.glb`` (scale 1.75, yawed 90 deg, top at z = 0) with its box collider,
  the 100 m checkered ground at z = -0.92, the white 0.8 x 1.2 x 0.02 m canvas centred 0.515 m in front of
  the base, ambient + two directional lights;
* the same cameras: ``base_camera`` at ``look_at([0.3, 0, 0.8], [0, 0, 0.1])`` with a fixed 84.7 deg
  horizontal FOV (the vertical FOV follows the frame aspect, like ManiSkill's fovy = 1.2 rad at 4:3) and the
  ``wrist_camera`` mounted on ``panda_hand`` at ``(-0.1, 0, -0.05)`` looking along the stick;
* the same ink mechanism: a pool of ``max_dots`` red cylinders (r = 4 mm, h = 3 mm) per env, one of which is
  moved from below the table to the pen-tip xy on the canvas top at every control step whose TCP is below
  ``z = 0.035``. The dots are one ``UsdGeom.PointInstancer`` (visual only, no physics).

Every environment uses ManiSkill's world frame as its local frame (env origin = ManiSkill world origin), so
poses reported by :meth:`IsaacDrawEnv.tcp_pose_env` are directly comparable to the ManiSkill data.
"""
from __future__ import annotations

import asyncio
import dataclasses
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from burger.backends.isaac import geometry as G
from burger.backends.isaac.tools.prepare_robot_assets import prepare as prepare_robot_urdf

ASSET_DIR = Path(__file__).resolve().parent / "assets"
# link prims of the imported URDF are nested under Robot/Geometry following the kinematic tree
HAND_PRIM_SUBPATH = "Geometry/panda_link0/panda_link1/panda_link2/panda_link3/panda_link4/panda_link5/panda_link6/panda_link7/panda_link8/panda_hand"
MANISKILL_ASSETS = ASSET_DIR / "maniskill"
GENERATED_DIR = ASSET_DIR / "generated"

DEFAULT_ENV_OPTIONS = dict(
    max_dots=1500, ik="maniskill_lm", ik_lambda=G.IK_LAMBDA, stiffness=G.ARM_STIFFNESS, damping=G.ARM_DAMPING, effort_limit=G.ARM_FORCE_LIMIT,
    sim_dt=G.SIM_DT, decimation=G.CONTROL_DECIMATION, disable_robot_gravity=True, solver_position_iterations=4, solver_velocity_iterations=0, contact_offset=0.01, rest_offset=0.0,
    env_spacing=12.0, render_mode="RealTimePathTracing", light_shadows=False, light_angle_deg=0.0, antialiasing="FXAA", enable_shadows=False, sampled_lighting=True, samples_per_pixel=1, tonemap_op=1, ambient_intensity=1.0, dome_intensity=0.0, sun_intensity=900.0, fill_intensity=0.0,
    table_visual=True, ground_texture=True, canvas_roughness=0.9, extra_kit_args="",
    appearance_list=None,   # list of burger.pipeline.appearance.Appearance dicts (one per env); None = ManiSkill look
    appearance_seed=0,
    camera_preset=None,     # burger.pipeline.appearance.CAMERA_PRESETS name / dict; None = ManiSkill's base_camera
    canvas_size=None,       # (x depth, y width) of the canvas; None = ManiSkill's 0.8 x 1.2 m (the lab's board is 0.6 x 0.9 m)
    # [isaac v6] optional joint-space limits on the per-step IK goal (LIBERO pd_ee_max_joint_vel / pd_ee_joint_margin analogue):
    # max_joint_vel > 0 scales the whole dq so |dq| <= max_joint_vel * Franka datasheet velocity * control_dt (keeps the
    # Cartesian direction); joint_margin (rad) keeps the goal inside the URDF joint range. 0 = off (ManiSkill behaviour)
    max_joint_vel=0.0, joint_margin=0.0,
)


# ------------------------------------------------------------------------------------------
# USD helpers (only usable once the Kit app is running)
# ------------------------------------------------------------------------------------------
def convert_table_asset(out_dir: Path = GENERATED_DIR, log=print) -> Path:
    """Convert ManiSkill's ``table.glb`` to USD with Kit's asset converter (cached in ``out_dir``)."""
    out = Path(out_dir) / "table.usd"
    if out.exists():
        return out
    from isaaclab.sim.utils import enable_extension

    enable_extension("omni.kit.asset_converter")
    import omni.kit.asset_converter as converter

    out.parent.mkdir(parents=True, exist_ok=True)

    async def _convert():
        ctx = converter.AssetConverterContext()
        ctx.ignore_materials = False
        ctx.ignore_animations = True
        ctx.ignore_camera = True
        ctx.ignore_light = True
        ctx.use_meter_as_world_unit = True
        ctx.baking_scales = True
        task = converter.get_instance().create_converter_task(str(MANISKILL_ASSETS / "table.glb"), str(out), None, ctx)
        ok = await task.wait_until_finished()
        if not ok:
            raise RuntimeError(f"table.glb conversion failed: {task.get_status()} {task.get_error_message()}")

    t0 = time.time()
    asyncio.get_event_loop().run_until_complete(_convert())
    log(f"[isaac env] converted table.glb -> {out} in {time.time() - t0:.1f}s")
    return out


def create_textured_ground(stage, path: str, size: float, altitude: float, texture: Path, square_len: float):
    """ManiSkill ``build_ground``: a ``size`` x ``size`` plane at ``altitude`` tiled with ``texture`` every ``square_len`` m."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    h = size / 2.0
    mesh.CreatePointsAttr([Gf.Vec3f(-h, -h, altitude), Gf.Vec3f(h, -h, altitude), Gf.Vec3f(h, h, altitude), Gf.Vec3f(-h, h, altitude)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.CreateExtentAttr([Gf.Vec3f(-h, -h, altitude), Gf.Vec3f(h, h, altitude)])
    reps = size / square_len
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(reps, 0), Gf.Vec2f(reps, reps), Gf.Vec2f(0, reps)]))
    mat = UsdShade.Material.Define(stage, path + "/material")
    shader = UsdShade.Shader.Define(stage, path + "/material/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, path + "/material/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, path + "/material/texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return mesh


def bind_preview_material(stage, prim_path: str, mat_path: str, color, texture: Path | None = None, roughness: float = 0.8, uv_scale=None):
    """Bind a UsdPreviewSurface material (flat colour or PNG texture, optional st scale) to ``prim_path``."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if texture is None:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in color]))
    else:
        reader = UsdShade.Shader.Define(stage, mat_path + "/stReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_out = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        if uv_scale is not None:
            xf = UsdShade.Shader.Define(stage, mat_path + "/stScale")
            xf.CreateIdAttr("UsdTransform2d")
            xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
            xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(float(uv_scale[0]), float(uv_scale[1])))
            st_out = xf.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(stage, mat_path + "/texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    prim = stage.GetPrimAtPath(prim_path)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    return mat


def create_ink_instancer(stage, path: str, positions: np.ndarray, radius: float, height: float, color):
    """One PointInstancer holding every ink dot of every env (cylinder prototype, axis Z)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    inst = UsdGeom.PointInstancer.Define(stage, path)
    colors = [tuple(color)] if not (isinstance(color, (list, tuple)) and len(color) and isinstance(color[0], (list, tuple))) else [tuple(c) for c in color]
    protos = []
    for k, col in enumerate(colors):
        proto = UsdGeom.Cylinder.Define(stage, path + f"/proto/dot_{k}")
        proto.CreateRadiusAttr(float(radius))
        proto.CreateHeightAttr(float(height))
        proto.CreateAxisAttr("Z")
        proto.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2), Gf.Vec3f(radius, radius, height / 2)])
        mat = UsdShade.Material.Define(stage, path + f"/proto/material_{k}")
        shader = UsdShade.Shader.Define(stage, path + f"/proto/material_{k}/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in col[:3]]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(proto.GetPrim()).Bind(mat)
        protos.append(proto.GetPath())
    inst.CreatePrototypesRel().SetTargets(protos)
    n = int(positions.shape[0])
    per_env = n // max(len(colors), 1)
    inst.CreateProtoIndicesAttr().Set(Vt.IntArray([min(i // max(per_env, 1), len(colors) - 1) for i in range(n)]))
    inst.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(positions, dtype=np.float32)))
    return inst


def _skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[:, 0])
    return torch.stack([torch.stack([z, -v[:, 2], v[:, 1]], 1), torch.stack([v[:, 2], z, -v[:, 0]], 1), torch.stack([-v[:, 1], v[:, 0], z], 1)], 1)


# ------------------------------------------------------------------------------------------
# the environment
# ------------------------------------------------------------------------------------------
class IsaacDrawEnv:
    """Vectorised drawing scene. Construct after ``AppLauncher`` created the Kit app.

    Args:
        num_envs: number of parallel environments (cloned on a grid).
        camera_width/height: size of ``base_camera`` and ``wrist_camera`` frames.
        wrist_camera: also render the hand-mounted camera.
        compare_wh: optional (w, h) of an extra ``compare_camera`` with the base-camera pose (e.g. (320, 240)
            to compare pixel-for-pixel with the ManiSkill reference renders).
        options: see ``DEFAULT_ENV_OPTIONS``.
        device: Isaac Lab device string (``cuda:N``); N must be the GPU passed to ``AppLauncher``.
    """

    def __init__(self, num_envs: int, camera_width: int = 128, camera_height: int = 128, wrist_camera: bool = True, compare_wh=None,
                 options: dict | None = None, device: str = "cuda:0", log=print):
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.sim.converters import UrdfConverterCfg
        from isaaclab.utils import configclass
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
        from isaaclab_physx.sim.schemas import PhysxArticulationCfg
        import omni.usd

        self.opt = opt = dict(DEFAULT_ENV_OPTIONS, **(options or {}))
        self.num_envs = int(num_envs)
        self.device = device
        self.log = log
        self.camera_hw = (int(camera_height), int(camera_width))
        self.wrist_camera = bool(wrist_camera)
        self.compare_wh = tuple(int(v) for v in compare_wh) if compare_wh else None
        self.max_dots = int(opt["max_dots"])
        self.decimation = int(opt["decimation"])
        self.sim_dt = float(opt["sim_dt"])
        self.control_dt = self.sim_dt * self.decimation
        self.canvas_size = tuple(float(v) for v in (opt.get("canvas_size") or G.CANVAS_SIZE[:2]))
        canvas_size3 = (self.canvas_size[0], self.canvas_size[1], float(G.CANVAS_SIZE[2]))
        self.spec = dataclasses.replace(G.SceneSpec.for_camera(camera_width, camera_height), canvas_size=canvas_size3)

        t0 = time.time()
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.sim_dt, render_interval=self.decimation, device=device,
            physics=PhysxCfg(solver_type=1, min_position_iteration_count=int(opt["solver_position_iterations"]),
                             max_position_iteration_count=int(opt["solver_position_iterations"]),
                             min_velocity_iteration_count=int(opt["solver_velocity_iterations"]), max_velocity_iteration_count=int(opt["solver_velocity_iterations"])),
        ))
        # RTPT compatibility mode honors the GI and reflection controls
        # used by this scene. Without it, disabling GI left bright stochastic speckles.
        import carb

        self._render_carb_settings = {
            "/rtx/shadows/enabled": bool(opt["enable_shadows"]), "/rtx/directLighting/sampledLighting/enabled": bool(opt["sampled_lighting"]),
            "/rtx/directLighting/sampledLighting/samplesPerPixel": int(opt["samples_per_pixel"]), "/rtx/ambientOcclusion/enabled": False,
            "/rtx/indirectDiffuse/enabled": False, "/rtx/reflections/enabled": False, "/rtx/translucency/enabled": False,
            "/rtx/post/aa/op": {"Off": 0, "TAA": 1, "FXAA": 2, "DLSS": 3, "DLAA": 4}[str(opt["antialiasing"])],
            "/rtx/post/tonemap/op": int(opt["tonemap_op"]), "/rtx/post/histogram/enabled": False, "/rtx/sceneDb/ambientLightIntensity": float(opt["ambient_intensity"]),
            "/rtx/rtpt/rtCompatibility": True,
            "/rtx/rtpt/cached/enabled": False,
            "/rtx/rtpt/lightcache/cached/enabled": False,
            "/rtx/rendermode": str(opt["render_mode"]),
            "/isaaclab/render/deterministic": False,
        }
        settings = carb.settings.get_settings()
        for k, v in self._render_carb_settings.items():
            settings.set(k, v)
        # appearance specs (one per env); when given they replace the ManiSkill table/ground/canvas look and the lights
        self.appearances = None
        if opt.get("appearance_list"):
            from burger.pipeline import appearance as _A
            apps = [a if isinstance(a, _A.Appearance) else (_A.Appearance.from_dict(a) if isinstance(a, dict) else _A.PRESETS[str(a)]) for a in opt["appearance_list"]]
            self.appearances = [apps[i % len(apps)] for i in range(self.num_envs)]
        use_app = self.appearances is not None
        self.camera_spec = None
        if opt.get("camera_preset") not in (None, "", "none", "maniskill"):
            from burger.pipeline import appearance as _A
            self.camera_spec = _A.resolve_camera(dict(camera_preset=opt["camera_preset"], canvas_size=self.canvas_size))
        table_usd = convert_table_asset(log=log) if (opt["table_visual"] and not use_app) else None
        robot_urdf = prepare_robot_urdf(log=log)   # glb visuals -> OBJ (the URDF importer drops glb meshes)
        # Keep the same settings before camera creation and after renderer initialization.
        # FXAA uses the current frame; RTPT color/light caches are also disabled.
        render_settings = IsaacRtxRendererGlobalSettingsCfg(
            antialiasing_mode=opt["antialiasing"], enable_shadows=bool(opt["enable_shadows"]), enable_ambient_occlusion=False, enable_global_illumination=False,
            enable_reflections=False, enable_translucency=False, enable_direct_lighting=True,
            carb_settings=dict(self._render_carb_settings),
        )
        renderer_cfg = IsaacRtxRendererCfg(global_settings=render_settings)
        aperture_ms = G.horizontal_aperture_for(G.CAMERA_FOVX)

        def pinhole(aperture=aperture_ms):
            return sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=aperture, clipping_range=(0.01, 100.0))

        H, W = self.camera_hw
        # camera preset: same pose / relative intrinsics as the other backends. Omniverse ignores principal-point offsets, so an
        # off-centre principal point is produced by rendering a wider centred frame and cropping it (CameraSpec.render_geometry)
        self._crops = {}   # camera name -> (x0, width)
        base_w, base_aperture = W, aperture_ms
        compare_w, compare_aperture = (self.compare_wh[0] if self.compare_wh else None), aperture_ms
        if self.camera_spec is not None:
            spec = self.camera_spec
            cam_eye, cam_target = tuple(float(v) for v in spec.eye), tuple(float(v) for v in spec.target)
            base_w, x0 = spec.render_geometry(W)
            base_aperture = 24.0 * base_w / (spec.f_rel * W)          # fx = 24 * render_w / aperture = f_rel * W
            if base_w != W:
                self._crops["base_camera"] = (x0, W)
            if self.compare_wh:
                compare_w, x0c = spec.render_geometry(self.compare_wh[0])
                compare_aperture = 24.0 * compare_w / (spec.f_rel * self.compare_wh[0])
                if compare_w != self.compare_wh[0]:
                    self._crops["compare_camera"] = (x0c, self.compare_wh[0])
        else:
            cam_eye, cam_target = G.CAMERA_EYE, G.CAMERA_TARGET
        cam_q = G.look_at_quat_xyzw(cam_eye, cam_target)
        wrist_q = G.euler_sxyz_quat_xyzw(*G.WRIST_CAMERA_EULER_SXYZ)
        rest = {f"panda_joint{i + 1}": float(q) for i, q in enumerate(G.REST_QPOS)}
        collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=float(opt["contact_offset"]), rest_offset=float(opt["rest_offset"]))

        @configclass
        class DrawSceneCfg(InteractiveSceneCfg):
            table_collision = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/TableCollision",
                                           spawn=sim_utils.CuboidCfg(size=G.TABLE_COLLISION_SIZE, visible=False, collision_props=collision_props),
                                           init_state=AssetBaseCfg.InitialStateCfg(pos=G.TABLE_COLLISION_CENTER))
            canvas = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Canvas",
                                  spawn=sim_utils.CuboidCfg(size=canvas_size3, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=float(opt["canvas_roughness"])),
                                                            collision_props=collision_props),
                                  init_state=AssetBaseCfg.InitialStateCfg(pos=G.CANVAS_CENTER))
            robot = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=str(robot_urdf), usd_dir=str(GENERATED_DIR), usd_file_name="panda_stick_isaac.usd",
                    fix_base=True, fix_root_link=True, merge_fixed_joints=False, run_multi_physics_conversion=False,
                    joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=float(opt["stiffness"]), damping=float(opt["damping"])), target_type="position"),
                    # The fragment writer normalizes every root-bearing API, including the
                    # imported Newton schema, without removing referenced USD properties.
                    articulation_props={"(/.*)?": [PhysxArticulationCfg(
                        enabled_self_collisions=False,
                        solver_position_iteration_count=int(opt["solver_position_iterations"]),
                        solver_velocity_iteration_count=int(opt["solver_velocity_iterations"]),
                    )]},
                    collision_props=collision_props,
                    # ManiSkill disables gravity on every robot link when a controller is attached (BaseAgent: "balancing
                    # passive force is not supported in PhysX"), which is why its PD tracks within a few mm.
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=bool(opt["disable_robot_gravity"])),
                ),
                init_state=ArticulationCfg.InitialStateCfg(pos=G.ROBOT_BASE, joint_pos=rest),
                actuators={"arm": ImplicitActuatorCfg(joint_names_expr=["panda_joint.*"], stiffness=float(opt["stiffness"]), damping=float(opt["damping"]), effort_limit_sim=float(opt["effort_limit"]))},
            )
            base_camera = CameraCfg(prim_path="{ENV_REGEX_NS}/base_camera", offset=CameraCfg.OffsetCfg(pos=cam_eye, rot=cam_q, convention="world"),
                                    data_types=["rgb"], spawn=pinhole(base_aperture), width=base_w, height=H, renderer_cfg=renderer_cfg)

        scene_cfg_cls = DrawSceneCfg
        extra_fields = {}
        if table_usd is not None:
            extra_fields["table"] = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table", spawn=sim_utils.UsdFileCfg(usd_path=str(table_usd), scale=(G.TABLE_SCALE,) * 3),
                                                 init_state=AssetBaseCfg.InitialStateCfg(pos=G.TABLE_POS, rot=G.yaw_quat_xyzw(G.ISAAC_TABLE_YAW)))
        elif use_app:
            # 5 cm table-top slab with the appearance's table extent (default: ManiSkill's footprint); material set per env after cloning
            ext0 = _A.table_extent_of(self.appearances[0], self.canvas_size)
            extra_fields["table"] = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table", spawn=sim_utils.CuboidCfg(size=(ext0[1] - ext0[0], ext0[3] - ext0[2], 0.01), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5))),
                                                 init_state=AssetBaseCfg.InitialStateCfg(pos=((ext0[0] + ext0[1]) / 2, (ext0[2] + ext0[3]) / 2, -0.005)))
            # dark table edge below the 1 cm top (the real table's apron)
            extra_fields["table_apron"] = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/TableApron", spawn=sim_utils.CuboidCfg(size=(ext0[1] - ext0[0] + 0.002, ext0[3] - ext0[2] + 0.002, 0.04), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.02, 0.02))),
                                                       init_state=AssetBaseCfg.InitialStateCfg(pos=((ext0[0] + ext0[1]) / 2, (ext0[2] + ext0[3]) / 2, -0.03)))
            # per-env ground planes (the appearance floor may differ per env) and canvas rims
            extra_fields["ground_env"] = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Ground", spawn=sim_utils.CuboidCfg(size=(float(opt["env_spacing"]) * 0.98, float(opt["env_spacing"]) * 0.98, 0.01), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2))),
                                                      init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, G.GROUND_Z - 0.005)))
            rim_w = max(float(a.canvas_rim_width) for a in self.appearances)
            if rim_w > 0:
                extra_fields["canvas_rim"] = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/CanvasRim", spawn=sim_utils.CuboidCfg(size=(canvas_size3[0] + 2 * rim_w, canvas_size3[1] + 2 * rim_w, canvas_size3[2] - 0.001), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.16, 0.16))),
                                                          init_state=AssetBaseCfg.InitialStateCfg(pos=(G.CANVAS_CENTER[0], G.CANVAS_CENTER[1], G.CANVAS_CENTER[2] - 0.0005)))
        else:
            extra_fields["table"] = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table", spawn=sim_utils.CuboidCfg(size=G.TABLE_COLLISION_SIZE, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.28))),
                                                 init_state=AssetBaseCfg.InitialStateCfg(pos=G.TABLE_COLLISION_CENTER))
        # lighting: flat RTX ambient (carb setting above) + ManiSkill's two directional lights; a dome light is optional.
        # NOTE: light attributes are baked at scene creation (runtime USD edits are not picked up by the renderer).
        if use_app:
            # scene-wide lights from the first appearance (RTX lights are global): Isaac units relative to the spec are
            # calibrated on the "real" preset (flat ambient 2.7 x, distant light 1000 x key, 380 x fill)
            a0 = self.appearances[0]
            self._render_carb_settings["/rtx/sceneDb/ambientLightIntensity"] = float(2.35 * a0.ambient * a0.exposure)
            settings.set("/rtx/sceneDb/ambientLightIntensity", self._render_carb_settings["/rtx/sceneDb/ambientLightIntensity"])
            opt = dict(opt, dome_intensity=0.0, sun_intensity=float(950.0 * a0.key_intensity * a0.exposure), fill_intensity=float(360.0 * a0.fill_intensity * a0.exposure),
                       light_shadows=bool(a0.shadows))
            self.opt = opt
            sun_dir, fill_dir, sun_color, fill_color = tuple(a0.key_dir), tuple(a0.fill_dir), tuple(a0.key_color), tuple(a0.fill_color)
            light_angle = max(float(getattr(a0, "light_angle_deg", 0.0)), 0.53)    # angular size -> soft shadows (RTX area light)
            if getattr(a0, "key2_dir", None) is not None and float(getattr(a0, "key2_intensity", 0.0)) > 0:
                extra_fields["sun2"] = AssetBaseCfg(prim_path="/World/sun2_light", spawn=sim_utils.DistantLightCfg(intensity=float(950.0 * a0.key2_intensity * a0.exposure), color=tuple(float(c) for c in a0.key2_color), angle=light_angle),
                                                    init_state=AssetBaseCfg.InitialStateCfg(rot=G.distant_light_quat_xyzw(tuple(a0.key2_dir))))
        else:
            sun_dir, fill_dir, sun_color, fill_color = G.SUN_DIRECTION, G.FILL_DIRECTION, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)
            light_angle = 0.53
        if float(opt["dome_intensity"]) > 0:
            extra_fields["dome"] = AssetBaseCfg(prim_path="/World/dome_light", spawn=sim_utils.DomeLightCfg(intensity=float(opt["dome_intensity"]), color=(1.0, 1.0, 1.0)))
        if float(opt["sun_intensity"]) > 0:
            extra_fields["sun"] = AssetBaseCfg(prim_path="/World/sun_light", spawn=sim_utils.DistantLightCfg(intensity=float(opt["sun_intensity"]), color=tuple(float(c) for c in sun_color), angle=light_angle),
                                               init_state=AssetBaseCfg.InitialStateCfg(rot=G.distant_light_quat_xyzw(sun_dir)))
        if float(opt["fill_intensity"]) > 0:
            extra_fields["fill"] = AssetBaseCfg(prim_path="/World/fill_light", spawn=sim_utils.DistantLightCfg(intensity=float(opt["fill_intensity"]), color=tuple(float(c) for c in fill_color)),
                                                init_state=AssetBaseCfg.InitialStateCfg(rot=G.distant_light_quat_xyzw(fill_dir)))
        if not opt["ground_texture"] and not use_app:
            extra_fields["ground"] = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(), init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, G.GROUND_Z)))
        if self.wrist_camera:
            extra_fields["wrist_camera"] = CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/" + HAND_PRIM_SUBPATH + "/wrist_camera", offset=CameraCfg.OffsetCfg(pos=G.WRIST_CAMERA_POS, rot=wrist_q, convention="world"),
                                                     data_types=["rgb"], spawn=pinhole(), width=W, height=H, renderer_cfg=renderer_cfg)
        if self.compare_wh:
            extra_fields["compare_camera"] = CameraCfg(prim_path="{ENV_REGEX_NS}/compare_camera", offset=CameraCfg.OffsetCfg(pos=cam_eye, rot=cam_q, convention="world"),
                                                       data_types=["rgb"], spawn=pinhole(compare_aperture), width=compare_w, height=self.compare_wh[1], renderer_cfg=renderer_cfg)
        if extra_fields:
            scene_cfg_cls = configclass(type("DrawSceneCfgExt", (DrawSceneCfg,), {"__annotations__": {k: type(v) for k, v in extra_fields.items()}, **extra_fields}))
        self.scene_cfg = scene_cfg_cls(num_envs=self.num_envs, env_spacing=float(opt["env_spacing"]))
        self.scene = InteractiveScene(self.scene_cfg)
        stage = omni.usd.get_context().get_stage()
        # fix_root_link delegates fixed-joint/root normalization to the PhysX
        # backend, which migrates every root-bearing schema (including Newton's).
        # Removing only PhysicsArticulationRootAPI leaves a second implicit root.
        # directional lights: sharp disc (no penumbra sampling noise); ManiSkill renders without cast shadows, which the
        # RTX path tracer only honours per *geometry* (primvars:doNotCastShadows), so flag every gprim in the scene
        from pxr import Sdf, UsdGeom, UsdLux
        for name in ("sun_light", "fill_light"):
            prim = stage.GetPrimAtPath(f"/World/{name}")
            if prim.IsValid():
                UsdLux.ShadowAPI.Apply(prim).CreateShadowEnableAttr(bool(opt["light_shadows"]))
                UsdLux.DistantLight(prim).CreateAngleAttr(float(opt["light_angle_deg"]))
        if not opt["light_shadows"]:
            self._disable_cast_shadows(stage, "/World")
        # ground (ManiSkill grid texture) and ink dots are global, non-physics prims
        n_cols = math.ceil(math.sqrt(self.num_envs))
        ground_size = max(G.GROUND_SIZE, float(opt["env_spacing"]) * (n_cols + 2))
        if opt["ground_texture"] and not use_app:
            create_textured_ground(stage, "/World/ground", ground_size, G.GROUND_Z, MANISKILL_ASSETS / "grid_texture.png", G.GROUND_TEXTURE_SQUARE_M)
        if use_app:
            self._apply_appearance_materials(stage)
        self.env_origins_np = None  # filled after reset
        self._ink_positions = None
        self._ink = None
        self._stage = stage
        self.log(f"[isaac env] scene with {self.num_envs} envs built in {time.time() - t0:.1f}s")
        t1 = time.time()
        self.sim.reset()
        self.robot = self.scene["robot"]
        if not self.robot.is_fixed_base:
            raise RuntimeError("Isaac robot articulation must be fixed-base; refusing inconsistent robot-base coordinates")
        self.env_origins = self.scene.env_origins.to(device)
        self.env_origins_np = self.env_origins.cpu().numpy()
        # ink pool: hidden inside the table at z = -0.003 (per env, in world coordinates)
        hidden = np.repeat(self.env_origins_np[:, None, :], self.max_dots, axis=1).reshape(-1, 3).astype(np.float32)
        hidden[:, 2] = G.DOT_HIDDEN_Z
        self._ink_hidden = hidden
        self._ink_positions = hidden.copy()
        ink_colors = [tuple(a.ink_color) for a in self.appearances] if use_app else G.DOT_COLOR
        self._ink = create_ink_instancer(stage, "/World/Ink", self._ink_positions, G.DOT_RADIUS, G.DOT_THICKNESS, ink_colors)
        if not opt["light_shadows"]:
            self._disable_cast_shadows(stage, "/World/Ink")
        self._ink_dirty = True
        # robot indices / buffers
        self.arm_ids, _ = self.robot.find_joints(["panda_joint.*"])
        self.hand_idx = self.robot.body_names.index("panda_hand")
        self._jac_body = self.hand_idx - 1
        self._jac_cols = list(self.arm_ids)
        self._tcp_offset = torch.tensor([0.0, 0.0, 0.15], device=device).expand(self.num_envs, 3)   # panda_hand -> panda_hand_tcp
        self._rest_q = torch.tensor(G.REST_QPOS, device=device, dtype=torch.float32).expand(self.num_envs, 7).clone()
        self._eye7 = torch.eye(7, device=device)
        self._lambda = float(opt["ik_lambda"])
        # [isaac v6] optional joint-velocity cap / joint margin (off by default) + numpy kinematics for the TCP-init teleport
        from burger.backends.isaac.kinematics import FRANKA_JOINT_VEL_LIMIT, PandaStickKinematics
        self.kin = PandaStickKinematics()
        mjv, margin = float(opt.get("max_joint_vel", 0.0)), float(opt.get("joint_margin", 0.0))
        self._max_dq = torch.tensor(mjv * FRANKA_JOINT_VEL_LIMIT * self.control_dt, device=device, dtype=torch.float32) if mjv > 0 else None
        self._q_lo = torch.tensor(self.kin.lower + margin, device=device, dtype=torch.float32) if margin > 0 else None
        self._q_hi = torch.tensor(self.kin.upper - margin, device=device, dtype=torch.float32) if margin > 0 else None
        q_target = G.euler_sxyz_quat_xyzw(*G.TCP_TARGET_EULER_XYZ)  # Rx(pi): stick pointing down, hand x along world x
        self._target_quat = torch.tensor(q_target, device=device, dtype=torch.float32).expand(self.num_envs, 4).clone()
        self.draw_count = np.zeros(self.num_envs, dtype=np.int64)
        self._all_env_ids = torch.arange(self.num_envs, device=device)
        self.reset_envs(None)
        # the cameras' render products are created on first use and Isaac Lab re-applies its own RTX defaults then;
        # render once, force our settings again and warm the renderer up so the first recorded frame is final
        self.render()
        for k, v in self._render_carb_settings.items():
            settings.set(k, v)
        for _ in range(2):
            self.sim.render()
        effective = {k: settings.get(k) for k in (
            "/rtx/rendermode", "/rtx/rtpt/rtCompatibility", "/rtx/rtpt/cached/enabled",
            "/rtx/rtpt/lightcache/cached/enabled", "/rtx/shadows/enabled",
            "/rtx/directLighting/sampledLighting/enabled", "/rtx/directLighting/sampledLighting/samplesPerPixel",
            "/rtx/post/aa/op", "/rtx/post/tonemap/op", "/rtx/sceneDb/ambientLightIntensity",
        )}
        self.log(f"[isaac env] sim.reset + initialise in {time.time() - t1:.1f}s; bodies={self.robot.body_names}; fixed_base={self.robot.is_fixed_base}; render settings {effective}")
        self.scene_build_s = time.time() - t0

    @staticmethod
    def _make_quad(stage, path, size_xy, z, uv=(1.0, 1.0), center=(0.0, 0.0)):
        """Horizontal textured quad (Isaac Lab's cuboid meshes carry no UVs, so textures need an authored mesh)."""
        from pxr import Gf, Sdf, UsdGeom, Vt

        mesh = UsdGeom.Mesh.Define(stage, path)
        hx, hy = size_xy[0] / 2.0, size_xy[1] / 2.0
        cx, cy = center
        mesh.CreatePointsAttr([Gf.Vec3f(cx - hx, cy - hy, z), Gf.Vec3f(cx + hx, cy - hy, z), Gf.Vec3f(cx + hx, cy + hy, z), Gf.Vec3f(cx - hx, cy + hy, z)])
        mesh.CreateFaceVertexCountsAttr([4]); mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
        mesh.CreateExtentAttr([Gf.Vec3f(cx - hx, cy - hy, z), Gf.Vec3f(cx + hx, cy + hy, z)])
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        u, v = float(uv[0]), float(uv[1])   # u along x (table depth), v along y (table width) like the baked textures
        st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(u, 0), Gf.Vec2f(u, v), Gf.Vec2f(0, v)]))
        return mesh

    def _apply_appearance_materials(self, stage):
        """Per-env table / ground / canvas / rim materials from ``self.appearances`` (textures baked like the other backends)."""
        from burger.pipeline import appearance as _A

        env_ns = self.scene.env_ns
        side = float(self.opt["env_spacing"]) * 0.98
        for i, a in enumerate(self.appearances):
            seed = int(self.opt.get("appearance_seed", 0)) + i
            base = f"{env_ns}/env_{i}"
            # slab sides / base plane get the flat colours, textured quads sit 0.6 mm above them
            bind_preview_material(stage, f"{base}/Table", f"{base}/Looks/table_mat", a.table_color, None, roughness=float(a.table_roughness))
            bind_preview_material(stage, f"{base}/Ground", f"{base}/Looks/ground_mat", a.floor_color, None, roughness=0.9)
            if stage.GetPrimAtPath(f"{base}/TableApron").IsValid():
                bind_preview_material(stage, f"{base}/TableApron", f"{base}/Looks/apron_mat", a.table_apron_color or a.table_color, None, roughness=0.9)
            ext = _A.table_extent_of(a, self.canvas_size)
            tsize, tcenter = (ext[1] - ext[0], ext[3] - ext[2]), ((ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2)
            tex = _A.table_texture_png(a, tsize, seed=seed)
            if tex is not None:
                self._make_quad(stage, f"{base}/TableTop", tsize, 0.0006, (1.0, 1.0), tcenter)
                bind_preview_material(stage, f"{base}/TableTop", f"{base}/Looks/table_top_mat", a.table_color, tex, roughness=float(a.table_roughness))
            ftex = _A.floor_texture_png(a, seed=seed)
            if ftex is not None:
                reps = side * float(a.floor_texture_scale)
                self._make_quad(stage, f"{base}/GroundTop", (side, side), G.GROUND_Z + 0.0006, (reps, reps))
                bind_preview_material(stage, f"{base}/GroundTop", f"{base}/Looks/ground_top_mat", a.floor_color, ftex, roughness=0.9)
            bind_preview_material(stage, f"{base}/Canvas", f"{base}/Looks/canvas_mat", a.canvas_color, None, roughness=0.9)
            if stage.GetPrimAtPath(f"{base}/CanvasRim").IsValid():
                bind_preview_material(stage, f"{base}/CanvasRim", f"{base}/Looks/rim_mat", a.canvas_rim_color, None, roughness=0.8)
            if a.led_color or getattr(a, "pen_style", None) == "marker":
                self._add_led_markers(stage, f"{base}/Robot", a)
        if not self.opt["light_shadows"]:
            self._disable_cast_shadows(stage, env_ns)

    def _add_led_markers(self, stage, robot_path: str, a):
        """Emissive status-LED spheres on the arm links (burger.pipeline.appearance.LED_MARKERS)."""
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        from burger.pipeline import appearance as _A

        from pxr import Usd

        k = float(a.led_intensity)
        col = [float(c) for c in (a.led_color or (0.0, 0.0, 0.0))]
        robot_prim = stage.GetPrimAtPath(robot_path)

        def find_link(name):
            # the URDF converter nests the links (Robot/Geometry/panda_link0/.../panda_link5); match by prim name
            hits = [pr for pr in Usd.PrimRange(robot_prim) if pr.GetName() == name]
            geo = [pr for pr in hits if "/Geometry/" in str(pr.GetPath())]
            return (geo or hits or [None])[0]

        def add_part(link_prim, name, shape, offset, size, color, emissive=None, roughness=0.5, no_shadow=True):
            path = f"{link_prim.GetPath()}/{name}"
            if shape == "sphere":
                geom = UsdGeom.Sphere.Define(stage, path); geom.CreateRadiusAttr(float(size))
                xf = UsdGeom.Xformable(geom); xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in offset]))
            elif shape == "box":
                geom = UsdGeom.Cube.Define(stage, path); geom.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(geom); xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in offset])); xf.AddScaleOp().Set(Gf.Vec3f(*[2.0 * float(v) for v in size]))
            elif shape == "capsule":   # along the link's z axis: (radius, half_length)
                geom = UsdGeom.Capsule.Define(stage, path); geom.CreateRadiusAttr(float(size[0])); geom.CreateHeightAttr(2.0 * float(size[1])); geom.CreateAxisAttr("Z")
                xf = UsdGeom.Xformable(geom); xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in offset]))
            else:   # cylinder along the link's z axis: (radius, half_length)
                geom = UsdGeom.Cylinder.Define(stage, path); geom.CreateRadiusAttr(float(size[0])); geom.CreateHeightAttr(2.0 * float(size[1])); geom.CreateAxisAttr("Z")
                xf = UsdGeom.Xformable(geom); xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in offset]))
            mat = UsdShade.Material.Define(stage, path + "_mat")
            sh = UsdShade.Shader.Define(stage, path + "_mat/Shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in color]))
            if emissive is not None:
                sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in emissive]))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat)
            if no_shadow:
                UsdGeom.PrimvarsAPI(geom.GetPrim()).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)

        for j, (link, offset, shape, size) in enumerate(_A.LED_MARKERS if a.led_color else ()):
            link_prim = find_link(link) if robot_prim.IsValid() else None
            if link_prim is None:
                self.log(f"[isaac] LED marker: link prim {link} not found under {robot_path}")
                continue
            # emissive x1.6 keeps the LED colour visible (x4 washes it out to white under RTX)
            add_part(link_prim, f"app_led_{j}", shape, offset, size, col, emissive=[c * k * 1.6 for c in col])
        if getattr(a, "pen_style", None) == "marker":
            hand = find_link("panda_hand") if robot_prim.IsValid() else None
            if hand is None:
                self.log(f"[isaac] marker: panda_hand prim not found under {robot_path}")
            else:
                for j, (shape, offset, size) in enumerate(_A.MARKER_PARTS):
                    add_part(hand, f"app_marker_{j}", shape, offset, size, _A.MARKER_COLOR, roughness=0.6, no_shadow=not bool(self.opt.get("light_shadows", True)))

    @staticmethod
    def _disable_cast_shadows(stage, root: str):
        """Author the RTX ``primvars:doNotCastShadows`` primvar (inherited by descendants) on ``root`` and on every
        editable gprim below it; instance proxies (instanceable robot meshes) inherit it from their ancestors."""
        from pxr import Sdf, Usd, UsdGeom

        n = 0
        root_prim = stage.GetPrimAtPath(root)
        if not root_prim.IsValid():
            return 0
        for prim in Usd.PrimRange(root_prim):
            if prim.IsInstanceProxy():
                continue
            if prim == root_prim or prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.PointInstancer) or prim.IsInstance():
                UsdGeom.PrimvarsAPI(prim).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
                n += 1
        return n

    # ---------------------------------------------------------------- state access
    def tcp_pose_w(self):
        """(N, 7) TCP pose in the stage world frame: hand pose shifted 0.15 m along the hand z axis. Returns (pose, offset_w)."""
        from isaaclab.utils.math import quat_apply

        hand = self.robot.data.body_pose_w.torch[:, self.hand_idx]
        r = quat_apply(hand[:, 3:7], self._tcp_offset)
        return torch.cat([hand[:, :3] + r, hand[:, 3:7]], dim=1), r

    def tcp_pose_env(self) -> np.ndarray:
        """(N, 7) TCP pose in the ManiSkill world frame (pos xyz, quat wxyz) - what ManiSkill records as ``tcp_pose``."""
        pose, _ = self.tcp_pose_w()
        pose = pose.detach().cpu().numpy().astype(np.float64)
        pose[:, :3] -= self.env_origins_np
        return np.concatenate([pose[:, :3], G.quat_xyzw_to_wxyz(pose[:, 3:7])], axis=1)

    def tcp_pos_root(self) -> torch.Tensor:
        """(N, 3) TCP position in the robot root frame (the pd_ee_pose action frame)."""
        from isaaclab.utils.math import subtract_frame_transforms
        pose, _ = self.tcp_pose_w()
        root = self.robot.data.root_pose_w.torch
        position, _ = subtract_frame_transforms(root[:, :3], root[:, 3:7], pose[:, :3], pose[:, 3:7])
        return position

    def joint_state(self):
        return self.robot.data.joint_pos.torch[:, self.arm_ids].detach().cpu().numpy(), self.robot.data.joint_vel.torch[:, self.arm_ids].detach().cpu().numpy()

    # ---------------------------------------------------------------- control
    def control_step(self, target_pos_root: torch.Tensor, target_quat_xyzw: torch.Tensor | None = None, hold_env_ids=None):
        """One 20 Hz control step: LM-IK to the absolute root-frame target, PD drive, ``decimation`` physics steps."""
        from isaaclab.utils.math import axis_angle_from_quat, quat_inv, quat_mul, subtract_frame_transforms

        root = self.robot.data.root_pose_w.torch
        ee, r = self.tcp_pose_w()
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], ee[:, :3], ee[:, 3:7])
        tq = self._target_quat if target_quat_xyzw is None else target_quat_xyzw
        dpos = target_pos_root - ee_pos_b
        drot = axis_angle_from_quat(quat_mul(tq, quat_inv(ee_quat_b)))
        dx = torch.cat([dpos, drot], dim=1).unsqueeze(-1)
        jh = self.robot.data.body_link_jacobian_w.torch[:, self._jac_body][:, :, self._jac_cols]
        jac = torch.cat([jh[:, :3] - _skew(r) @ jh[:, 3:], jh[:, 3:]], dim=1)   # hand-frame Jacobian moved to the TCP
        q = self.robot.data.joint_pos.torch[:, self.arm_ids]
        if self.opt["ik"] == "dls":
            jt = jac.transpose(1, 2)
            dq = (jt @ torch.linalg.inv(jac @ jt + self._lambda * torch.eye(6, device=self.device)) @ dx).squeeze(-1)
        else:  # ManiSkill Kinematics.compute_ik (levenberg_marquardt, alpha = 1)
            jt = jac.transpose(1, 2)
            dq = torch.linalg.solve(jt @ jac + self._lambda * self._eye7, jt @ dx).squeeze(-1)
        if self._max_dq is not None:   # [isaac v6] Franka joint-velocity cap (uniform scaling of the step)
            dq = dq * torch.clamp(torch.min(self._max_dq / dq.abs().clamp_min(1e-12), dim=1, keepdim=True).values, max=1.0)
        goal = q + dq
        if self._q_lo is not None:     # [isaac v6] joint margin
            goal = torch.maximum(torch.minimum(goal, self._q_hi), self._q_lo)
            dq = goal - q
        if hold_env_ids is not None and len(hold_env_ids):
            # Reset settling must match the initial joint hold, including orientation.
            goal[hold_env_ids] = q[hold_env_ids]
            dq[hold_env_ids] = 0
        self.robot.set_joint_position_target_index(target=goal, joint_ids=self.arm_ids)
        self.scene.write_data_to_sim()
        for _ in range(self.decimation):
            self.sim.step(render=False)
        self.scene.update(self.control_dt)
        return dq

    def settle(self):
        """Hold the current joint positions for one control period (``decimation`` physics steps).

        Kinematic state writes (``reset_envs``) only reach the renderer after a physics step, so this is used before
        recording the first observation of an episode; with gravity-free links and PD targets at the current pose
        the state does not change measurably."""
        q = self.robot.data.joint_pos.torch[:, self.arm_ids]
        self.robot.set_joint_position_target_index(target=q, joint_ids=self.arm_ids)
        self.scene.write_data_to_sim()
        for _ in range(self.decimation):
            self.sim.step(render=False)
        self.scene.update(self.control_dt)

    def render(self) -> dict:
        """Render all cameras once; returns uint8 numpy frames ``{"base": (N,H,W,3), "wrist": ..., "compare": ...}``."""
        if self._ink_dirty:
            from pxr import Vt

            self._ink.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(self._ink_positions))
            self._ink_dirty = False
        self.sim.render()
        out = {"base": self._fetch("base_camera")}
        if self.wrist_camera:
            out["wrist"] = self._fetch("wrist_camera")
        if self.compare_wh:
            out["compare"] = self._fetch("compare_camera")
        return out

    def _fetch(self, name: str) -> np.ndarray:
        rgb = self.scene[name].data.output["rgb"]
        out = rgb[..., :3].detach().cpu().numpy().astype(np.uint8, copy=False)
        crop = self._crops.get(name)
        if crop is not None:
            out = np.ascontiguousarray(out[:, :, crop[0]:crop[0] + crop[1]])
        return out

    # ---------------------------------------------------------------- ink
    def place_ink(self, active: np.ndarray | None = None) -> np.ndarray:
        """ManiSkill ``_after_control_step``: envs whose TCP is below the touch threshold leave one dot at the TCP xy."""
        tcp = self.tcp_pose_env()
        touching = tcp[:, 2] < G.TOUCH_Z_THRESHOLD
        if active is not None:
            touching &= active
        if np.any(touching & (self.draw_count >= self.max_dots)):
            raise RuntimeError(f"Ink capacity {self.max_dots} exceeded; increase backend_option max_dots instead of exporting a truncated drawing")
        idx = np.nonzero(touching)[0]
        for i in idx:
            k = i * self.max_dots + self.draw_count[i]
            self._ink_positions[k] = (tcp[i, 0] + self.env_origins_np[i, 0], tcp[i, 1] + self.env_origins_np[i, 1], G.DOT_PLACED_Z)
        if len(idx):
            self.draw_count[idx] += 1
            self._ink_dirty = True
        return touching

    def ink_positions_env(self, env_id: int) -> np.ndarray:
        n = int(self.draw_count[env_id])
        p = self._ink_positions[env_id * self.max_dots: env_id * self.max_dots + n].astype(np.float64)
        return p - self.env_origins_np[env_id]

    # ---------------------------------------------------------------- reset
    def tcp_init_qpos(self, tcp_target_root, max_trials_q0=None):
        """[isaac v6] reset-time TCP init (ManiSkill ``_maybe_randomize_tcp_init`` / LIBERO ``teleport_pen_tip`` analogue):
        joint positions (from the rest keyframe, iterated LM, kinematics only) that put the TCP at ``tcp_target_root``
        (robot root frame) with the pd_ee orientation target. Returns (q (7,), pos_err_m)."""
        from burger.backends.isaac.kinematics import rot_x_pi
        q, pe, _ = self.kin.ik(np.asarray(tcp_target_root, dtype=np.float64), rot_x_pi(), np.asarray(G.REST_QPOS), margin=float(self.opt.get("joint_margin", 0.0)))
        return q, pe

    def reset_envs(self, env_ids, qpos=None):
        """ManiSkill ``_initialize_episode``: robot to the rest keyframe (drive targets = rest), all dots hidden.
        [isaac v6] ``qpos`` (len(env_ids), 7): per-env initial joint positions instead of the rest keyframe (TCP init)."""
        ids = self._all_env_ids if env_ids is None else torch.as_tensor(np.asarray(env_ids, dtype=np.int64), device=self.device)
        n = len(ids)
        root = self.robot.data.default_root_pose.torch[ids].clone()
        root[:, :3] = self.env_origins[ids] + torch.tensor(G.ROBOT_BASE, device=self.device)
        self.robot.write_root_pose_to_sim_index(root_pose=root, env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((n, 6), device=self.device), env_ids=ids)
        q = self._rest_q[ids] if qpos is None else torch.as_tensor(np.asarray(qpos), device=self.device, dtype=torch.float32).reshape(n, 7)
        self.robot.write_joint_position_to_sim_index(position=q, joint_ids=self.arm_ids, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=self.arm_ids, env_ids=ids)
        self.robot.set_joint_position_target_index(target=q, joint_ids=self.arm_ids, env_ids=ids)
        self.robot.reset(ids)
        self.scene.write_data_to_sim()
        ids_np = ids.detach().cpu().numpy()
        for i in ids_np:
            self._ink_positions[i * self.max_dots: (i + 1) * self.max_dots] = self._ink_hidden[i * self.max_dots: (i + 1) * self.max_dots]
        self.draw_count[ids_np] = 0
        self._ink_dirty = True
        # refresh link poses for the new joint state without integrating (kinematic update)
        self.sim.forward()
        self.scene.update(0.0)

    def close(self):
        pass
