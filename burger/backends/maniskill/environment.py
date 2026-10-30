"""GPU-parallel SVG drawing environment for dataset generation.

Waypoints are produced by burger.pipeline.svg_trajectory and expressed in the
robot root frame. Rendering and robot dynamics follow the original ManiSkill
panda_stick environment; policy evaluation and real-robot tools are excluded.
"""

from __future__ import annotations

import numpy as np
import sapien
import torch
from mani_skill.agents.robots.panda.panda_stick import PandaStick
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from transforms3d.euler import euler2quat


@register_env("DrawSVGParallel-v1", max_episode_steps=None)
class DrawSVGParallelEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda_stick"]
    SUPPORTED_REWARD_MODES = ["sparse"]
    agent: PandaStick
    DOT_THICKNESS = 0.003
    CANVAS_THICKNESS = 0.02
    BRUSH_RADIUS = 0.004
    BRUSH_COLORS = [[0.8, 0.2, 0.2, 1.0]]
    TOUCH_TOLERANCE_Z = 0.012

    def __init__(
        self,
        *args,
        waypoints_list,
        robot_uids="panda_stick",
        generation_seed=0,
        max_dots=1500,
        wrist_camera=True,
        camera_width=128,
        camera_height=128,
        camera_fov=1.4790102243423462,
        camera_view="front",
        camera_preset=None,
        canvas_size=(0.8, 1.2),
        appearance_list=None,
        appearance_seed=0,
        randomize_tcp_init=False,
        robot_init_qpos_noise=0.0,
        robot_init_qpos=None,
        tcp_init_x_min=-0.05,
        tcp_init_x_max=0.05,
        tcp_init_y_min=-0.05,
        tcp_init_y_max=0.05,
        tcp_init_z=0.12,
        tcp_init_max_trials=10,
        **kwargs,
    ):
        self.waypoints_list = [np.asarray(p, dtype=np.float32) for p in waypoints_list]
        if not self.waypoints_list:
            raise ValueError("At least one SVG trajectory is required")
        if any(
            p.ndim != 2 or p.shape[1] != 3 or len(p) == 0 or not np.isfinite(p).all()
            for p in self.waypoints_list
        ):
            raise ValueError("Each trajectory must be a nonempty finite (T, 3) array")
        self.generation_seed = int(generation_seed)
        self.MAX_DOTS = int(max_dots)
        if self.MAX_DOTS < 1:
            raise ValueError("max_dots must be positive")
        self.num_envs = int(kwargs.get("num_envs", 1))
        self.next_svg_index = 0
        self.env_svg_indices = torch.full((self.num_envs,), -1, dtype=torch.long)
        self.env_point_indices = torch.zeros(self.num_envs, dtype=torch.long)
        self.env_completed = torch.zeros(self.num_envs, dtype=torch.bool)
        self.env_need_reset = torch.zeros(self.num_envs, dtype=torch.bool)
        self.env_svg_points = {}
        self.wrist_camera = bool(wrist_camera)
        self.camera_width, self.camera_height = int(camera_width), int(camera_height)
        self.camera_fov = float(camera_fov)
        self.camera_view = camera_view
        self.camera_preset = camera_preset
        self.canvas_size = tuple(canvas_size)
        self.appearance_list = appearance_list
        if appearance_list is not None and len(appearance_list) != self.num_envs:
            raise ValueError(
                "appearance_list must contain one appearance per environment"
            )
        self.appearance_seed = int(appearance_seed)
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.robot_init_qpos = robot_init_qpos
        self.randomize_tcp_init = bool(randomize_tcp_init)
        self.tcp_init_xy_noise = 0.0
        self.tcp_init_xy_margin = 0.0
        self.tcp_init_x_min, self.tcp_init_x_max = (
            float(tcp_init_x_min),
            float(tcp_init_x_max),
        )
        self.tcp_init_y_min, self.tcp_init_y_max = (
            float(tcp_init_y_min),
            float(tcp_init_y_max),
        )
        self.tcp_init_z = float(tcp_init_z)
        self.tcp_init_max_trials = int(tcp_init_max_trials)
        self.tcp_init_ik_iters = 80
        self._tcp_init_fallback_count = 0
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        for name in (
            "env_svg_indices",
            "env_point_indices",
            "env_completed",
            "env_need_reset",
        ):
            setattr(self, name, getattr(self, name).to(self.device))

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=100,
            control_freq=20,
            scene_config=SceneConfig(
                contact_offset=0.01,
                solver_position_iterations=4,
                solver_velocity_iterations=0,
            ),
        )

    @property
    def _default_sensor_configs(self):
        eye = [0.3, 0.0, 0.8] if self.camera_view == "front" else [0.3, -0.3, 0.6]
        pose = sapien_utils.look_at(eye=eye, target=[0, 0, 0.1])
        intrinsics = dict(fov=self.camera_fov)
        if self.camera_preset not in (None, "", "none", "maniskill"):
            from burger.pipeline.appearance import resolve_camera

            spec = resolve_camera(
                dict(camera_preset=self.camera_preset, canvas_size=self.canvas_size)
            )
            pose = sapien_utils.look_at(eye=list(spec.eye), target=list(spec.target))
            fx, fy, cx, cy = spec.intrinsics(self.camera_width, self.camera_height)
            intrinsics = dict(
                fov=None,
                intrinsic=np.array(
                    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
                ),
            )
        configs = [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=self.camera_width,
                height=self.camera_height,
                near=0.01,
                far=100,
                **intrinsics,
            )
        ]
        if self.wrist_camera:
            configs.append(
                CameraConfig(
                    "wrist_camera",
                    pose=sapien.Pose(
                        p=[-0.1, 0.0, -0.05], q=euler2quat(0, -np.pi / 2, 0)
                    ),
                    width=self.camera_width,
                    height=self.camera_height,
                    fov=self.camera_fov,
                    near=0.01,
                    far=100,
                    mount=self.agent.robot.links_map["panda_hand"],
                )
            )
        return configs

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            pose=sapien_utils.look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1]),
            width=640,
            height=480,
            fov=1.2,
            near=0.01,
            far=100,
        )

    def _load_scene(self, options):
        # Noise is added below from an SVG-index seed, independent of scheduling.
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.0)
        self.table_scene.build()
        if self.appearance_list is None:
            b = self.scene.create_actor_builder()
            size = (
                self.canvas_size[0] / 2,
                self.canvas_size[1] / 2,
                self.CANVAS_THICKNESS / 2,
            )
            b.add_box_visual(
                half_size=size,
                material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
            )
            b.add_box_collision(half_size=size)
            b.initial_pose = sapien.Pose(p=[-0.1, 0, self.CANVAS_THICKNESS / 2])
            self.canvas = b.build_static(name="canvas")
            colors = [self.BRUSH_COLORS[0]] * self.num_envs
        else:
            self._build_appearance_scene()
            colors = [list(a.ink_color) + [1.0] for a in self.appearance_list]
            self._add_tool_markers()
        self.dots = []
        for env_idx in range(self.num_envs):
            dots = []
            for i in range(self.MAX_DOTS):
                b = self.scene.create_actor_builder()
                b.add_cylinder_visual(
                    radius=self.BRUSH_RADIUS,
                    half_length=self.DOT_THICKNESS / 2,
                    material=sapien.render.RenderMaterial(base_color=colors[env_idx]),
                )
                b.set_scene_idxs([env_idx])
                b.initial_pose = sapien.Pose(p=[0, 0, -self.DOT_THICKNESS])
                dots.append(b.build_kinematic(name=f"dot_{i}_{env_idx}"))
            self.dots.append(dots)
        self.env_draw_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _build_appearance_scene(self):
        from burger.pipeline import appearance as A

        for actor in (self.table_scene.table, self.table_scene.ground):
            for obj in actor._objs:
                comp = obj.find_component_by_type(sapien.render.RenderBodyComponent)
                if comp is not None:
                    comp.disable()
        self.canvas_actors, self.appearance_actors = [], []
        for env_idx, app in enumerate(self.appearance_list):
            seed = self.appearance_seed + env_idx
            ext = A.table_extent_of(app, self.canvas_size)
            length, width = ext[1] - ext[0], ext[3] - ext[2]
            tex = A.table_texture_png(app, (length, width), seed=seed)
            mat = sapien.render.RenderMaterial(
                base_color=[1, 1, 1, 1] if tex else list(app.table_color) + [1],
                roughness=app.table_roughness,
            )
            if tex:
                mat.base_color_texture = sapien.render.RenderTexture2D(
                    filename=str(tex), srgb=True
                )
            b = self.scene.create_actor_builder()
            b.add_box_visual(half_size=(length / 2, width / 2, 0.005), material=mat)
            apron = app.table_apron_color or app.table_color
            b.add_box_visual(
                pose=sapien.Pose(p=[0, 0, -0.025]),
                half_size=(length / 2 + 0.001, width / 2 + 0.001, 0.02),
                material=sapien.render.RenderMaterial(
                    base_color=list(apron) + [1], roughness=0.9
                ),
            )
            b.set_scene_idxs([env_idx])
            b.initial_pose = sapien.Pose(
                p=[(ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2, -0.005]
            )
            self.appearance_actors.append(b.build_static(name=f"table_top_{env_idx}"))
            ftex = A.floor_texture_png(app, seed=seed) or A.texture_path(
                "plain", app.floor_color, 1.0, seed
            )
            self.appearance_actors.append(
                self._build_textured_ground(
                    env_idx,
                    str(ftex),
                    1.0 / max(app.floor_texture_scale, 1e-3),
                    -self.table_scene.table_height,
                )
            )
            hx, hy, hz = (
                self.canvas_size[0] / 2,
                self.canvas_size[1] / 2,
                self.CANVAS_THICKNESS / 2,
            )
            b = self.scene.create_actor_builder()
            b.add_box_visual(
                half_size=(hx, hy, hz),
                material=sapien.render.RenderMaterial(
                    base_color=list(app.canvas_color) + [1], roughness=0.9
                ),
            )
            b.add_box_collision(half_size=(hx, hy, hz))
            if app.canvas_rim_width > 0:
                rim = app.canvas_rim_width
                b.add_box_visual(
                    pose=sapien.Pose(p=[0, 0, -0.0005]),
                    half_size=(hx + rim, hy + rim, hz - 0.0005),
                    material=sapien.render.RenderMaterial(
                        base_color=list(app.canvas_rim_color) + [1]
                    ),
                )
            b.set_scene_idxs([env_idx])
            b.initial_pose = sapien.Pose(p=[-0.1, 0, hz])
            self.canvas_actors.append(b.build_static(name=f"canvas_{env_idx}"))
        # One batched view covers every environment; never index env>0 into only canvas_0.
        self.canvas = Actor.merge(self.canvas_actors, name="canvas")

    def _load_lighting(self, options):
        if self.appearance_list is None:
            return super()._load_lighting(options)
        # Retain the ManiSkill renderer conversion factors from the source backend.
        ambient_gain, key_gain, fill_gain = 0.65, 1.5, 0.4
        ambient = float(
            np.mean(
                [ambient_gain * a.ambient * a.exposure for a in self.appearance_list]
            )
        )
        self.scene.set_ambient_light([ambient] * 3)
        for i, a in enumerate(self.appearance_list):
            self.scene.add_directional_light(
                list(a.key_dir),
                [c * key_gain * a.key_intensity * a.exposure for c in a.key_color],
                shadow=a.shadows,
                shadow_scale=3,
                shadow_map_size=2048,
                scene_idxs=[i],
            )
            if a.fill_intensity:
                self.scene.add_directional_light(
                    list(a.fill_dir),
                    [
                        c * fill_gain * a.fill_intensity * a.exposure
                        for c in a.fill_color
                    ],
                    shadow=False,
                    scene_idxs=[i],
                )
            if a.key2_dir is not None and a.key2_intensity:
                self.scene.add_directional_light(
                    list(a.key2_dir),
                    [
                        c * key_gain * a.key2_intensity * a.exposure
                        for c in a.key2_color
                    ],
                    shadow=False,
                    scene_idxs=[i],
                )

    def _add_tool_markers(self):
        from burger.pipeline import appearance as A

        self.led_actors = []
        by_link = {}
        for link, offset, shape, size in A.LED_MARKERS:
            by_link.setdefault(link, []).append((offset, shape, size, True))
        for shape, offset, size in A.MARKER_PARTS:
            by_link.setdefault("panda_hand", []).append((offset, shape, size, False))
        for link_name, parts in by_link.items():
            link = self.agent.robot.links_map[link_name]
            for i, app in enumerate(self.appearance_list):
                b = self.scene.create_actor_builder()
                count = 0
                for offset, shape, size, led in parts:
                    if (led and app.led_color is None) or (
                        not led and app.pen_style != "marker"
                    ):
                        continue
                    color = (
                        [c * scale for c, scale in zip(app.led_color, (0.6, 1.1, 1.35))]
                        if led
                        else A.MARKER_COLOR
                    )
                    base_color = [c * 0.3 for c in color] if led else list(color)
                    mat = sapien.render.RenderMaterial(
                        base_color=base_color + [1], roughness=0.6
                    )
                    if led:
                        mat.emission = [c * app.led_intensity * 0.65 for c in color] + [
                            1
                        ]
                    pose = sapien.Pose(
                        p=offset,
                        q=euler2quat(0, np.pi / 2, 0)
                        if shape in ("cylinder", "capsule")
                        else [1, 0, 0, 0],
                    )
                    if shape == "box":
                        b.add_box_visual(pose=pose, half_size=size, material=mat)
                    elif shape == "sphere":
                        b.add_sphere_visual(pose=pose, radius=size, material=mat)
                    elif shape == "capsule":
                        b.add_capsule_visual(
                            pose=pose, radius=size[0], half_length=size[1], material=mat
                        )
                    else:
                        b.add_cylinder_visual(
                            pose=pose, radius=size[0], half_length=size[1], material=mat
                        )
                    count += 1
                if count:
                    b.set_scene_idxs([i])
                    b.initial_pose = sapien.Pose(p=[0, 0, -1])
                    self.led_actors.append(
                        (link, i, b.build_kinematic(name=f"tool_{link_name}_{i}"))
                    )

    def _sync_led_markers(self):
        for link, i, actor in getattr(self, "led_actors", []):
            actor.set_pose(link.pose[i])

    def _initialize_episode(self, env_idx, options):
        # BaseEnv invokes this during construction, before the first assignment.
        for name in (
            "env_svg_indices",
            "env_point_indices",
            "env_completed",
            "env_need_reset",
        ):
            setattr(self, name, getattr(self, name).to(self.device))
        if options.get("assign_svg", False):
            for env_id in env_idx.tolist():
                if self.next_svg_index >= len(self.waypoints_list):
                    raise RuntimeError(
                        "Collector attempted to assign an exhausted SVG queue"
                    )
                self.env_svg_indices[env_id] = self.next_svg_index
                points = torch.as_tensor(
                    self.waypoints_list[self.next_svg_index], device=self.device
                )
                self.env_svg_points[env_id] = points
                self._store_waypoints(env_id, points)
                self.next_svg_index += 1
                self.env_point_indices[env_id] = 0
                self.env_need_reset[env_id] = False
        self.table_scene.initialize(env_idx)
        if self.robot_init_qpos is not None or self.robot_init_qpos_noise:
            qpos = self.agent.robot.get_qpos()[env_idx].clone()
            if self.robot_init_qpos is not None:
                qpos[:, :7] = torch.as_tensor(self.robot_init_qpos, device=self.device)
            if self.robot_init_qpos_noise:
                for row, env_id in enumerate(env_idx.tolist()):
                    g = torch.Generator(device=self.device)
                    g.manual_seed(
                        self.generation_seed + max(0, int(self.env_svg_indices[env_id]))
                    )
                    qpos[row, :7] += (
                        torch.randn(7, generator=g, device=self.device)
                        * self.robot_init_qpos_noise
                    )
            self.agent.reset(init_qpos=qpos)
        # Apply robot reset before fetching: a fetch-first sequence restores the old qpos.
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        if options.get("assign_svg", False):
            self._maybe_randomize_tcp_init(env_idx)
        self.env_draw_steps[env_idx] = 0
        for env_id in env_idx.tolist():
            for dot in self.dots[env_id]:
                dot.set_pose(
                    sapien.Pose(
                        p=[0, 0, -self.DOT_THICKNESS], q=euler2quat(0, np.pi / 2, 0)
                    )
                )
        self._sync_led_markers()
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()

    def _after_control_step(self):
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        threshold = self.CANVAS_THICKNESS + self.DOT_THICKNESS + self.TOUCH_TOLERANCE_Z
        touching = self.agent.tcp.pose.p[:, 2] < threshold
        active = ~self.env_completed & (self.env_svg_indices >= 0)
        if bool((touching & active & (self.env_draw_steps >= self.MAX_DOTS)).any()):
            raise RuntimeError(
                "Ink capacity exceeded; increase backend_options.max_dots and regenerate"
            )
        can_draw = touching & active
        for i in torch.where(can_draw)[0].tolist():
            pos = self.agent.tcp.pose.p[i].clone()
            pos[2] = self.CANVAS_THICKNESS + self.DOT_THICKNESS / 2
            self.dots[i][int(self.env_draw_steps[i])].set_pose(
                Pose.create_from_pq(pos, euler2quat(0, np.pi / 2, 0))
            )
            self.env_draw_steps[i] += 1
        self._sync_led_markers()
        if self.gpu_sim_enabled:
            # Only ink and LED actors moved. Reapplying articulation root poses
            # here feeds fetched float32 poses back into PhysX every control step
            # and accumulates drift even for the fixed robot base.
            self.scene.px.gpu_apply_rigid_dynamic_data()

    def get_hardcoded_action(self):
        # Inactive and just-finished environments hold their current root-frame TCP.
        action_pos = (
            self.agent.robot.get_root_pose().inv() * self.agent.tcp.pose
        ).p.clone()
        if hasattr(self, "_wp_len"):
            indices = self.env_point_indices.long()
            active = ~self.env_completed & (self.env_svg_indices >= 0)
            take = active & (indices < self._wp_len)
            rows = torch.where(take)[0]
            action_pos[rows] = self._wp_buffer[rows, indices[rows]]
            self.env_point_indices[take] += 1
            self.env_need_reset[active & ~take] = True
        rotation = torch.tensor([np.pi, 0, 0], device=self.device).repeat(
            self.num_envs, 1
        )
        return torch.cat([action_pos, rotation], dim=1)

    def _get_obs_extra(self, info):
        return dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            episode_svg_index=self.env_svg_indices[:, None],
            ink_count=self.env_draw_steps[:, None],
        )

    def evaluate(self):
        if not hasattr(self, "_wp_len"):
            return {
                "success": torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            }
        return {
            "success": (self.env_svg_indices >= 0)
            & (self.env_point_indices >= self._wp_len)
        }

    def _build_textured_ground(
        self, env_idx, texture_file, square_len, altitude, floor_width=40
    ):
        """Per-env variant of ``mani_skill.utils.building.ground.build_ground`` (visual only)."""
        import numpy as _np

        builder = self.scene.create_actor_builder()
        builder.set_scene_idxs([env_idx])
        builder.initial_pose = sapien.Pose()
        actor = builder.build_static(name=f"ground_{env_idx}")
        num = (floor_width + 1) ** 2
        vertices = _np.zeros((num, 3))
        half = floor_width / 2
        rng_ = _np.arange(-half, half + 1)
        xx, yy = _np.meshgrid(rng_, rng_)
        xys = _np.stack((yy, xx), axis=2).reshape(-1, 2)
        vertices[:, 0] = xys[:, 0]
        vertices[:, 1] = xys[:, 1]
        vertices[:, 2] = altitude
        normals = _np.zeros_like(vertices)
        normals[:, 2] = 1
        mat = sapien.render.RenderMaterial(roughness=0.9)
        mat.base_color_texture = sapien.render.RenderTexture2D(
            filename=texture_file, mipmap_levels=4, srgb=True
        )
        uv_scale = floor_width / square_len
        uvs = _np.zeros((num, 2))
        uvs[:, 0] = (xys[:, 0] * uv_scale + half) / floor_width
        uvs[:, 1] = (xys[:, 1] * uv_scale + half) / floor_width
        tris = []
        for i in range(floor_width):
            base = _np.arange(floor_width) + i * (floor_width + 1)
            tris.append(_np.stack([base, base + floor_width + 1, base + 1], axis=1))
            tris.append(
                _np.stack(
                    [base + floor_width + 1, base + floor_width + 2, base + 1], axis=1
                )
            )
        shape = sapien.render.RenderShapeTriangleMesh(
            vertices=vertices,
            triangles=_np.concatenate(tris),
            normals=normals,
            uvs=uvs,
            material=mat,
        )
        for obj in actor._objs:
            comp = sapien.render.RenderBodyComponent()
            comp.attach(shape)
            obj.add_component(comp)
        return actor

    def _store_waypoints(self, env_idx: int, points: torch.Tensor):
        """Keep all per-env trajectories in one padded GPU buffer so that action generation and
        success evaluation are single batched tensor ops (no per-env ``.item()`` GPU syncs)."""
        n = int(points.shape[0])
        buf = getattr(self, "_wp_buffer", None)
        if buf is None or buf.shape[1] < n:
            new_len = max(n, buf.shape[1] if buf is not None else 0)
            new_buf = torch.zeros(
                (self.num_envs, new_len, 3), device=self.device, dtype=torch.float32
            )
            if buf is not None:
                new_buf[:, : buf.shape[1]] = buf
            self._wp_buffer = new_buf
        if not hasattr(self, "_wp_len"):
            self._wp_len = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
        self._wp_buffer[env_idx, :n] = points
        self._wp_len[env_idx] = n

    def _maybe_randomize_tcp_init(self, env_idx: torch.Tensor):
        """Reset directly to a downward TCP pose within a safe visible XY workspace.

        Samples use generation_seed + SVG index, independently of vector slot.
        Bounds and target height are in ManiSkill world coordinates. Each IK
        target is converted to the robot root frame before the iterative solve.
        """
        if not self.randomize_tcp_init:
            return

        arm_controller = self.agent.controller.controllers.get("arm")
        if arm_controller is None or not hasattr(arm_controller, "kinematics"):
            raise RuntimeError("Random TCP initialization requires an IK controller")

        def _as_batched(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor

        def _get_reset_batch_tcp_pos() -> torch.Tensor:
            tcp_pos = _as_batched(self.agent.tcp.pose.p)
            if tcp_pos.shape[0] == reset_batch_size:
                return tcp_pos.clone()
            if tcp_pos.shape[0] == self.num_envs:
                return tcp_pos[env_idx].clone()
            raise RuntimeError(
                f"[tcp init] Unexpected tcp pose batch shape {tuple(tcp_pos.shape)} "
                f"for reset batch size {reset_batch_size}."
            )

        current_qpos = _as_batched(self.agent.robot.get_qpos().clone())
        reset_batch_size = len(env_idx)
        if current_qpos.shape[0] == self.num_envs:
            current_qpos = current_qpos[env_idx].clone()
        elif current_qpos.shape[0] != reset_batch_size:
            raise RuntimeError(
                f"[tcp init] Unexpected qpos batch shape {tuple(current_qpos.shape)} "
                f"for reset batch size {reset_batch_size}."
            )
        init_qpos = current_qpos.clone()
        base_tcp_pos = _get_reset_batch_tcp_pos()

        batch_size = reset_batch_size
        target_pos = base_tcp_pos.clone()
        if self.tcp_init_z is not None:
            target_pos[:, 2] = self.tcp_init_z

        workspace_center_xy = base_tcp_pos[:, :2].clone()
        user_min_xy = (
            torch.tensor(
                [self.tcp_init_x_min, self.tcp_init_y_min],
                device=self.device,
                dtype=base_tcp_pos.dtype,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        user_max_xy = (
            torch.tensor(
                [self.tcp_init_x_max, self.tcp_init_y_max],
                device=self.device,
                dtype=base_tcp_pos.dtype,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        if self.tcp_init_xy_margin > 0:
            user_min_xy = user_min_xy + self.tcp_init_xy_margin
            user_max_xy = user_max_xy - self.tcp_init_xy_margin
        if self.tcp_init_xy_noise > 0:
            sample_half_range_xy = torch.full(
                (batch_size, 2),
                self.tcp_init_xy_noise,
                device=self.device,
                dtype=base_tcp_pos.dtype,
            )
            sample_min_xy = torch.max(
                workspace_center_xy - sample_half_range_xy, user_min_xy
            )
            sample_max_xy = torch.min(
                workspace_center_xy + sample_half_range_xy, user_max_xy
            )
        else:
            sample_min_xy = user_min_xy
            sample_max_xy = user_max_xy
        down_quat = torch.tensor(
            euler2quat(np.pi, 0, 0),
            device=self.device,
            dtype=base_tcp_pos.dtype,
        ).unsqueeze(0)

        invalid_range_mask = torch.any(sample_max_xy <= sample_min_xy, dim=1)
        if invalid_range_mask.any():
            raise ValueError("TCP initialization sampling bounds are empty")
        unresolved_mask = ~invalid_range_mask

        # Reuse each SVG's generator across retries.
        row_generators = {}
        for local_row, env_id in enumerate(env_idx.tolist()):
            gen = torch.Generator(device=self.device)
            gen.manual_seed(self.generation_seed + int(self.env_svg_indices[env_id]))
            row_generators[local_row] = gen

        for _ in range(self.tcp_init_max_trials):
            active_rows = torch.where(unresolved_mask)[0]
            if len(active_rows) == 0:
                break
            trial_init_qpos = init_qpos.clone()
            trial_goals = target_pos.clone()
            proposed_mask = torch.zeros(
                batch_size, dtype=torch.bool, device=self.device
            )

            for local_row in active_rows.tolist():
                env_id = int(env_idx[local_row])
                trial_target_pos = target_pos[local_row : local_row + 1].clone()
                trial_target_pos[:, :2] = (
                    torch.empty(
                        (1, 2), device=self.device, dtype=base_tcp_pos.dtype
                    ).uniform_(0.0, 1.0, generator=row_generators.get(local_row))
                    * (
                        sample_max_xy[local_row : local_row + 1]
                        - sample_min_xy[local_row : local_row + 1]
                    )
                    + sample_min_xy[local_row : local_row + 1]
                )
                trial_target_pos[:, :2] = torch.max(
                    torch.min(
                        trial_target_pos[:, :2], user_max_xy[local_row : local_row + 1]
                    ),
                    user_min_xy[local_row : local_row + 1],
                )

                target_pose = Pose.create_from_pq(
                    p=trial_target_pos,
                    q=down_quat,
                )
                # compute_ik interprets the target in the robot ROOT frame, but
                # target_pose is in world coords; convert it (the base may be
                # offset from the world origin, e.g. panda_stick at x=-0.615).
                target_pose_ik = (
                    self.agent.robot.get_root_pose()[env_id].inv() * target_pose
                )
                # GPU compute_ik returns a single LM step; iterate (feeding the
                # solution back as q0) so it actually converges to target_pose.
                q_full = init_qpos[local_row : local_row + 1].clone()
                arm_target_qpos = None
                for _ in range(self.tcp_init_ik_iters):
                    sol = arm_controller.kinematics.compute_ik(
                        pose=target_pose_ik,
                        q0=q_full,
                    )
                    if sol is None:
                        break
                    if sol.ndim == 1:
                        sol = sol.unsqueeze(0)
                    if sol.shape[1] == q_full.shape[1]:
                        q_full = sol.clone()
                    else:
                        q_full = q_full.clone()
                        q_full[:, arm_controller.active_joint_indices] = sol
                    arm_target_qpos = q_full
                if arm_target_qpos is None:
                    continue

                trial_goals[local_row] = trial_target_pos[0]
                proposed_mask[local_row] = True
                trial_init_qpos[local_row : local_row + 1] = arm_target_qpos

            if not proposed_mask.any():
                continue

            self.agent.reset(init_qpos=trial_init_qpos)
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene._gpu_fetch_all()

            actual_tcp_pos = _get_reset_batch_tcp_pos()
            actual_xy = actual_tcp_pos[:, :2]
            in_bounds = (
                torch.all(actual_xy >= user_min_xy, dim=1)
                & torch.all(actual_xy <= user_max_xy, dim=1)
                & proposed_mask
                & ((actual_tcp_pos - trial_goals).norm(dim=1) <= 0.001)
            )
            if in_bounds.any():
                accepted_local_rows = torch.where(in_bounds)[0]
                init_qpos[accepted_local_rows] = trial_init_qpos[accepted_local_rows]
                unresolved_mask[accepted_local_rows] = False

        failed_rows = torch.where(unresolved_mask)[0]
        if len(failed_rows):
            raise RuntimeError(
                f"TCP initialization failed after {self.tcp_init_max_trials} trials"
            )

        self.agent.reset(init_qpos=init_qpos)
        self.agent.controller.reset()
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
