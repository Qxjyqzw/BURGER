"""CPU regressions for worker integrity and the actual Isaac collection loop."""
import importlib.metadata
import json
from pathlib import Path
import importlib.util
import subprocess
import sys
import types

import h5py
import numpy as np
import pytest

from burger.data import canonical as C
from burger.backends._worker_utils import validate_options, validate_parts, validate_gpu_selection
from burger.backends import _worker_utils
from burger.backends.isaac import backend as isaac_backend
from burger.backends.libero import backend as libero_backend
from burger.backends.base import GenerationConfig


@pytest.fixture(autouse=True)
def isolate_cpu_tests_from_scheduler_masks(monkeypatch):
    # Tests below use fake workers and CPU fixtures, not the host's CUDA devices.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)


@pytest.mark.parametrize("mask", ["0", "GPU-verified-single-device"])
def test_single_gpu_mask_is_preserved_for_isaac_and_libero(monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setattr(_worker_utils.subprocess, "run", lambda *args, **kwargs:
                        types.SimpleNamespace(stdout="0, GPU-verified-single-device\n"))
    assert validate_gpu_selection(None, backend="Isaac") == 0
    assert libero_backend.set_egl_device(0) == 0
    assert _worker_utils.os.environ["CUDA_VISIBLE_DEVICES"] == mask
    assert _worker_utils.os.environ["MUJOCO_EGL_DEVICE_ID"] == "0"


@pytest.mark.parametrize("mask,inventory,device", [
    ("", "0, GPU-single\n", 0),
    ("-1", "0, GPU-single\n", 0),
    ("1", "0, GPU-single\n", 0),
    ("0,1", "0, GPU-a\n1, GPU-b\n", 0),
    ("0", "0, GPU-a\n1, GPU-b\n", 0),
    ("0", "0, GPU-single\n", 1),
    ("GPU-other", "0, GPU-single\n", 0),
    ("MIG-unverified", "0, GPU-single\n", 0),
    ("0", "unreadable", 0),
])
def test_ambiguous_render_mapping_fails_without_changing_mask(monkeypatch, mask, inventory, device):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "4")
    monkeypatch.setattr(_worker_utils.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(stdout=inventory))
    with pytest.raises(RuntimeError, match="mask is preserved"):
        validate_gpu_selection(device, backend="Isaac")
    with pytest.raises(RuntimeError, match="mask is preserved"):
        libero_backend.set_egl_device(device)
    assert _worker_utils.os.environ["CUDA_VISIBLE_DEVICES"] == mask
    assert _worker_utils.os.environ["MUJOCO_EGL_DEVICE_ID"] == "4"


def test_masked_generation_requires_inventory_verification(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    def missing_inventory(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(_worker_utils.subprocess, "run", missing_inventory)
    with pytest.raises(RuntimeError, match="Could not verify"):
        validate_gpu_selection(0, backend="Isaac")
    assert _worker_utils.os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def episode(source, index, steps=2):
    positions = np.tile([0.0, 0.0, 0.1], (steps, 1))
    return C.Episode(
        obs=dict(agentview_rgb=np.zeros((steps, 4, 4, 3), np.uint8), eef_pos=positions,
                 eef_quat_wxyz=np.tile([1., 0., 0., 0.], (steps, 1))),
        actions=np.zeros((steps, 6)), arrays={},
        meta=dict(svg_file=str(source), svg_index=index),
    )


def part(path, items):
    info = C.DatasetInfo("isaac", "draw the supplied geometric sketch", 20, "panda_stick", ["agentview_rgb"], {"dim": 6})
    with C.CanonicalWriter(path, info) as output:
        for position, (source, index) in enumerate(items):
            output.add_episode(episode(source, index), position)
    return path


def test_worker_outputs_require_exact_sources_and_unique_ids(tmp_path):
    sources = [tmp_path / "first.svg", tmp_path / "second.svg"]
    first = part(tmp_path / "a.h5", [(sources[0], 0)])
    second = part(tmp_path / "b.h5", [(sources[1], 1)])
    assert validate_parts([second, first], sources) == {"num_episodes": 2, "total_steps": 4}
    with pytest.raises(RuntimeError, match="missing"):
        validate_parts([first], sources)
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_parts([first, first], sources)
    with pytest.raises(RuntimeError, match="Unexpected"):
        validate_parts([first], [tmp_path / "different.svg"])
    with h5py.File(second, "r+") as output:
        output["episodes/ep_0"].attrs["truncated"] = True
    with pytest.raises(RuntimeError, match="Truncated"):
        validate_parts([first, second], sources)


def test_failed_zero_exit_worker_cannot_reuse_old_files(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (work / "parts").mkdir()
    source = tmp_path / "input.svg"
    part(work / "parts/part_0.h5", [(source, 0)])
    (work / "isaac_worker_summary.json").write_text('{"num_episodes": 1, "status": "complete"}')

    class EmptyWorker:
        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(isaac_backend.subprocess, "Popen", EmptyWorker)
    monkeypatch.setattr(isaac_backend.IsaacBackend, "venv_python", classmethod(lambda cls: Path(sys.executable)))
    cfg = GenerationConfig([str(source)], "draw the supplied geometric sketch")
    output = tmp_path / "published.h5"
    with pytest.raises(RuntimeError, match="failed"):
        isaac_backend.IsaacBackend().generate(cfg, output, work, log=lambda *_: None)
    assert not output.exists()


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), subprocess.TimeoutExpired("worker", 1)])
def test_isaac_worker_cleanup_runs_on_interrupt_or_timeout(tmp_path, monkeypatch, failure):
    cleaned = []

    class InterruptedWorker:
        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout):
            raise failure

    monkeypatch.setattr(isaac_backend.subprocess, "Popen", InterruptedWorker)
    monkeypatch.setattr(isaac_backend, "stop_process_group", lambda process: cleaned.append(process))
    monkeypatch.setattr(isaac_backend.IsaacBackend, "venv_python", classmethod(lambda cls: Path(sys.executable)))
    cfg = GenerationConfig([str(tmp_path / "input.svg")], "draw the supplied geometric sketch")
    with pytest.raises((KeyboardInterrupt, RuntimeError)):
        isaac_backend.IsaacBackend().generate(cfg, tmp_path / "output.h5", tmp_path / "work", log=lambda *_: None)
    assert len(cleaned) == 1


def test_isaac_kit_shutdown_gets_failure_code_and_summary(tmp_path, monkeypatch):
    from burger.backends.isaac import worker as collector
    close_codes = []
    launch_options = []
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"options": {}}))
    summary = tmp_path / "summary.json"

    def launch(**kwargs):
        launch_options.append(kwargs)
        return types.SimpleNamespace(app=types.SimpleNamespace(close=lambda **kw: close_codes.append(kw)))

    def fail(*args):
        raise RuntimeError("deliberate worker failure")

    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.app", types.SimpleNamespace(AppLauncher=launch))
    monkeypatch.setattr(collector, "run", fail)
    assert collector.main(["--job_json", str(job), "--summary_json", str(summary), "--device", "2"]) == 1
    assert close_codes == [{"exit_code": 1}]
    assert json.loads(summary.read_text())["status"] == "failed"
    assert "--/renderer/activeGpu=2" in launch_options[0]["kit_args"]


@pytest.mark.parametrize("options", [{"max_dots": "10"}, {"keep_work": "false"}, {"hold_steps": -1},
                                     {"misspelled_option": 1}, {"timeout_s": 0}, {"antialiasing": "DLSS"},
                                     {"render_mode": "PathTracing"}, {"sampled_lighting": False}, {"samples_per_pixel": 0},
                                     {"max_dots": True}, {"tcp_init_random": -1},
                                     {"tcp_init_x_min": 2., "tcp_init_x_max": 1.}])
def test_invalid_backend_options_fail_early(options):
    with pytest.raises(ValueError):
        validate_options(options, isaac_backend.DEFAULT_OPTIONS, isaac_backend.EXTRA_OPTIONS)


class Tensor(np.ndarray):
    """Only the tensor transport used by the collector, with physics implemented below."""
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def tensor(value, **kwargs):
    return np.asarray(value).view(Tensor)


def test_isaac_ink_overflow_raises_instead_of_silently_dropping_ink(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(Tensor=Tensor))
    path = Path(__file__).parents[1] / "burger/backends/isaac/environment.py"
    spec = importlib.util.spec_from_file_location("_test_draw_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = types.SimpleNamespace(tcp_pose_env=lambda: np.array([[0., 0., .01, 1., 0., 0., 0.]]),
                                draw_count=np.array([2]), max_dots=2)
    with pytest.raises(RuntimeError, match="Ink capacity"):
        module.IsaacDrawEnv.place_ink(env, np.array([True]))


@pytest.mark.parametrize("parallel", [1, 2])
def test_isaac_executes_final_action_and_resets_only_reused_slots(tmp_path, monkeypatch, parallel):
    from burger.backends.isaac import worker as collector
    from burger.backends.isaac import geometry as G
    from burger.pipeline import svg_trajectory

    executed = []

    class FakeEnv:
        def __init__(self, num_envs, **kwargs):
            self.num_envs = num_envs
            self.current = np.tile([0.1, 0.0, 0.2], (num_envs, 1))
            self.draw_count = np.zeros(num_envs, dtype=int)
            self.control_dt = 0.05
            self.scene_build_s = 0.001
            self.spec = types.SimpleNamespace(to_dict=lambda: {})
            self.opt = {}
            self.next = self.current.copy()
            self.reset_slots = []

        def settle(self):
            pass

        def render(self):
            return {"base": np.zeros((self.num_envs, 4, 4, 3), dtype=np.uint8)}

        def tcp_pose_env(self):
            return np.column_stack((self.current + G.ROBOT_BASE, np.tile([0., 1., 0., 0.], (self.num_envs, 1))))

        def joint_state(self):
            return np.zeros((self.num_envs, 7)), np.zeros((self.num_envs, 7))

        def tcp_pos_root(self):
            return tensor(self.current.copy())

        def control_step(self, targets, hold_env_ids=None):
            assert set(self.reset_slots) <= set(hold_env_ids or ()), "Reused slots must settle with a joint hold"
            self.reset_slots = []
            goal = np.asarray(targets).copy()
            if hold_env_ids:
                goal[hold_env_ids] = self.current[hold_env_ids]
            self.current = goal
            self.next = self.current.copy()

        def place_ink(self, active):
            for i in np.flatnonzero(active):
                executed.append(tuple(np.round(self.next[i], 4)))
            self.draw_count += active

        def reset_envs(self, indices, **kwargs):
            self.current[indices] = [0.1, 0.0, 0.2]
            self.draw_count[indices] = 0
            self.reset_slots = list(indices)

        def ink_positions_env(self, index):
            return np.tile(self.current[index] + G.ROBOT_BASE, (self.draw_count[index], 1))

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        zeros=lambda shape, **kwargs: tensor(np.zeros(shape)), as_tensor=tensor, float32=np.float32,
    ))
    monkeypatch.setitem(sys.modules, "burger.backends.isaac.environment", types.SimpleNamespace(
        IsaacDrawEnv=FakeEnv, DEFAULT_ENV_OPTIONS={},
    ))
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "test")
    sources = []
    for index in range(3):
        source = tmp_path / f"{index}.svg"
        source.write_text(f'<svg><path d="M {index} 0 L {index+1} 0"/></svg>')
        sources.append(str(source))
    sequences = {
        svg_trajectory.read_svg_path_string(sources[i]): np.array([[0.3 + i * .1, j * .01, .02] for j in range(2 + i)])
        for i in range(3)
    }
    monkeypatch.setattr(svg_trajectory, "build_ee_waypoints", lambda data, *args, **kwargs: sequences[data].copy())
    cfg = GenerationConfig(sources, "draw the supplied geometric sketch", parallel=parallel, wrist_camera=False,
                           camera_height=4, camera_width=4)
    output = tmp_path / "part.h5"
    job = tmp_path / "job.json"
    options = dict(isaac_backend.DEFAULT_OPTIONS, hold_steps=0, raise_steps=0)
    job.write_text(json.dumps(dict(run_id="test", cfg=cfg.to_json(), options=options, svg_files=sources, part=str(output))))
    args = types.SimpleNamespace(job_json=str(job), device=0, summary_json=str(tmp_path/"summary.json"), max_steps=None, debug_dir=None)
    assert collector.run(args, 0.001) == 0
    assert validate_parts([output], sources) == {"num_episodes": 3, "total_steps": 9}
    expected = [tuple(np.round(row, 4)) for sequence in sequences.values() for row in sequence]
    assert sorted(executed) == sorted(expected)
    for index, (_, recorded) in enumerate(C.iter_episodes(output)):
        np.testing.assert_allclose(recorded.actions[:, :3], list(sequences.values())[index])
        assert recorded.obs["ink_count"][0] == 0
        assert recorded.meta["ink_count"] == len(recorded.actions)
        assert recorded.meta["trajectory_complete"]


def test_libero_guard_applies_to_osc_and_noise_is_episode_seeded(tmp_path, monkeypatch):
    from burger.pipeline import svg_trajectory
    source = tmp_path / "guard.svg"
    source.write_text('<svg><path d="M 0 0 L 1 0"/></svg>')
    offsets = []
    commands = []

    def build(data, *args, **kwargs):
        offsets.append(kwargs["xy_offset"])
        return np.array([[.6, .1, .02]])

    monkeypatch.setattr(svg_trajectory, "build_ee_waypoints", build)
    monkeypatch.setitem(sys.modules, "robosuite", types.ModuleType("robosuite"))
    monkeypatch.setitem(sys.modules, "robosuite.utils", types.ModuleType("robosuite.utils"))
    monkeypatch.setitem(sys.modules, "robosuite.utils.transform_utils", types.ModuleType("robosuite.utils.transform_utils"))
    obs = dict(agentview_image=np.zeros((4, 4, 3), np.uint8), robot0_eef_pos=np.array([0., 0., .1]),
               robot0_eef_quat=np.array([0., 0., 0., 1.]), robot0_joint_pos=np.zeros(7), robot0_gripper_qpos=np.zeros(2))
    penv = types.SimpleNamespace(
        pen_tip_pos=lambda: np.array([0., 0., .1]), set_target_waypoints=lambda _: None,
        ink_count=0, ink_positions=lambda: np.zeros((0, 3)), touch_z_threshold=.035,
        mark_episode_complete=lambda _: None, ink_coverage=lambda: 0.,
        waypoint_action=lambda target, **kwargs: commands.append(target.copy()) or np.zeros(7),
    )
    env = types.SimpleNamespace(env=penv, reset=lambda: obs, step=lambda _: (obs, 0, False, {}),
                                sim=types.SimpleNamespace(model=types.SimpleNamespace(get_xml=lambda: "<mujoco/>")))
    layout = types.SimpleNamespace(painter_to_world=lambda values: values.copy(), world_to_painter=lambda values: values.copy())
    cfg = GenerationConfig([str(source)], "draw the supplied geometric sketch", wrist_camera=False, camera_height=4, camera_width=4, svg_xy_noise=.1)
    options = dict(libero_backend.DEFAULT_OPTIONS, approach_guard_xy=.01, approach_guard_z=.2, store_states=False)
    for rank in (0, 9):
        recorded = libero_backend.run_episode(env, layout, source, cfg, options, np.random.default_rng(rank), 3)
        np.testing.assert_allclose(recorded.arrays["targets_world"][0], [.6, .1, .02])
        np.testing.assert_allclose(recorded.arrays["commanded_targets_world"][0], [.6, .1, .2])
    np.testing.assert_array_equal(offsets[0], offsets[1])
    np.testing.assert_array_equal(commands[0], [.6, .1, .2])
