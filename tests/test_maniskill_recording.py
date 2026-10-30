"""CPU regression checks for vector recording; no simulator installation needed."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from burger.backends.maniskill.collector import (
    record_episodes,
)
from burger.data.canonical import CanonicalWriter, DatasetInfo
from burger.backends.maniskill.backend import (
    ManiSkillBackend,
    validate_options,
)
from burger.backends.base import GenerationConfig


class Pose:
    def __init__(self, rows):
        self.raw_pose = rows

    @property
    def p(self):
        return self.raw_pose[:, :3]


class FakeEnv:
    """A mutable two-slot simulator double with deliberately uneven episode lengths."""

    def __init__(self, lengths=(2, 6, 1), truncate=False):
        self.unwrapped = self
        self.num_envs = 2
        self.lengths = lengths
        self.next_svg_index = 0
        self.env_completed = np.zeros(2, dtype=bool)
        self.env_need_reset = np.zeros(2, dtype=bool)
        self.env_svg_indices = np.full(2, -1)
        self.env_draw_steps = np.zeros(2, dtype=int)
        self.point_indices = np.zeros(2, dtype=int)
        self.root = Pose(np.tile([-0.615, 0, 0, 1, 0, 0, 0], (2, 1)))
        self.tcp = Pose(np.tile([0.0, 0, 0.12, 0, 1, 0, 0], (2, 1)))
        self.rgb = np.zeros((2, 2, 2, 3), dtype=np.uint8)
        self.agent = SimpleNamespace(
            tcp=SimpleNamespace(pose=self.tcp),
            robot=SimpleNamespace(
                get_qpos=lambda: np.zeros((2, 7)), get_root_pose=lambda: self.root
            ),
        )
        self.reset_log = []
        self.truncate = truncate

    def obs(self):
        return {"sensor_data": {"base_camera": {"rgb": self.rgb}}}

    def reset(self, seed=None, options=None):
        indices = np.asarray(options.get("env_idx", [0, 1]))
        self.reset_log.append(indices.tolist())
        for i in indices:
            assert self.next_svg_index < len(self.lengths), (
                "reset must never allocate an idle episode"
            )
            self.env_svg_indices[i] = self.next_svg_index
            self.next_svg_index += 1
            self.point_indices[i] = 0
            self.env_draw_steps[i] = 0
            self.env_need_reset[i] = False
            self.tcp.p[i] = [self.env_svg_indices[i] / 100, 0, 0.12]
            self.rgb[i] = [self.env_svg_indices[i], 0, 0]
        return self.obs(), {}

    def get_hardcoded_action(self):
        actions = np.column_stack(
            [self.tcp.p - self.root.p, np.tile([np.pi, 0, 0], (2, 1))]
        )
        for i in np.flatnonzero(~self.env_completed):
            idx = self.env_svg_indices[i]
            if self.point_indices[i] < self.lengths[idx]:
                actions[i, :3] = [0.515 + self.point_indices[i] / 100, idx / 100, 0.02]
                self.point_indices[i] += 1
            else:
                self.env_need_reset[i] = True
        return actions

    def step(self, actions):
        self.tcp.p[:] = actions[:, :3] + self.root.p
        self.env_draw_steps += ~self.env_completed
        self.rgb[:, :, :, 1] += 1  # Mutate the same observation buffer in place.
        return (
            self.obs(),
            None,
            self.env_need_reset.copy(),
            np.full(2, self.truncate),
            {},
        )


def info():
    return DatasetInfo(
        "maniskill",
        "Draw the supplied shape",
        20,
        "panda_stick",
        ["agentview_rgb"],
        {"dim": 6},
    )


def record(env, path):
    waypoints = [np.zeros((n, 3)) for n in env.lengths]
    with CanonicalWriter(path, info()) as writer:
        record_episodes(
            env,
            writer,
            [f"{i}.svg" for i in range(len(env.lengths))],
            ["M0 0L1 1"] * len(env.lengths),
            waypoints,
            np.zeros((len(env.lengths), 2)),
            log=lambda _: None,
        )


def test_uneven_lengths_record_exact_svg_count_without_idle_tail(tmp_path):
    env = FakeEnv()
    path = tmp_path / "recorded.h5"
    record(env, path)
    assert env.reset_log == [[0, 1], [0]]
    with h5py.File(path) as f:
        assert set(f["episodes"]) == {"ep_0", "ep_1", "ep_2"}
        assert f.attrs["num_episodes"] == 3
        for i, n in enumerate(env.lengths):
            ep = f[f"episodes/ep_{i}"]
            assert ep["actions"].shape == (n + 1, 6)
            assert ep.attrs["svg_index"] == i
            assert ep.attrs["trajectory_complete"]
            assert ep.attrs["final_ink_count"] == n + 1
            assert ep["obs/agentview_rgb"][0, 0, 0].tolist() == [i, 0, 0]
            np.testing.assert_allclose(ep["obs/eef_pos"][0], [i / 100, 0, 0.12])
            np.testing.assert_allclose(
                ep["arrays/targets_world"][:], ep["actions"][:, :3] + [-0.615, 0, 0]
            )
            np.testing.assert_allclose(
                ep["obs/pen_tip_painter"][:], ep["obs/eef_pos"][:] - [-0.615, 0, 0]
            )
            np.testing.assert_allclose(
                ep["obs/eef_pos"][1:], ep["arrays/targets_world"][:-1]
            )
        # Environment 1 retains every pre-action frame despite environment 0 resetting.
        assert f["episodes/ep_1/obs/agentview_rgb"][:, 0, 0, 1].tolist() == list(
            range(7)
        )


def test_truncation_is_failure_not_successful_short_episode(tmp_path):
    with pytest.raises(RuntimeError, match="truncated"):
        record(FakeEnv(truncate=True), tmp_path / "truncated.h5")


def test_world_frame_terminal_hold_is_rejected_before_execution(tmp_path):
    class HistoricalHoldEnv(FakeEnv):
        def get_hardcoded_action(self):
            action = super().get_hardcoded_action()
            # Reproduce the original collector's erroneous world-coordinate hold.
            action[self.env_need_reset, :3] = self.tcp.p[self.env_need_reset]
            return action

        def step(self, action):
            assert not self.env_need_reset.any(), "The faulty hold reached physics"
            return super().step(action)

    with pytest.raises(ValueError, match="Terminal hold action must use robot-base coordinates"):
        record(HistoricalHoldEnv(), tmp_path / "bad-hold.h5")


@pytest.mark.parametrize("quaternion", [[0.0, 1.0, 0.0, 0.0], [np.nan] * 4])
def test_invalid_root_pose_reports_simulator_values(tmp_path, quaternion):
    env = FakeEnv()
    env.root.raw_pose[1, 3:] = quaternion
    with pytest.raises(
        ValueError,
        match=r"translation-only.*'env_index': 1.*'svg_index': 1.*'recorded_rows': 0.*root_pose_xyz_wxyz",
    ):
        record(env, tmp_path / "invalid.h5")


@pytest.mark.parametrize(
    "options",
    [
        {"max_dot": 10},
        {"randomize_tcp_init": "false"},
        {"max_dots": True},
        {"max_dots": 0},
        {"camera_view": "unknown"},
        {"appearance_random": "mild"},
        {"tcp_init_x_min": 0.5, "tcp_init_x_max": 0.0},
        {"robot_init_qpos": [0, 0]},
    ],
)
def test_invalid_backend_options_fail_before_simulator_start(options):
    with pytest.raises(ValueError):
        validate_options(options)


def test_ambiguous_gpu_mask_is_preserved_and_rejected_before_simulator_import(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    cfg = GenerationConfig(svg_files=["example.svg"], language="Draw a shape")
    with pytest.raises(RuntimeError, match="mask is preserved"):
        ManiSkillBackend().generate(cfg, tmp_path / "out.h5", tmp_path)
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
