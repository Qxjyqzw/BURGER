# LIBERO / MuJoCo backend

Generate drawing demonstrations with LIBERO, robosuite 1.4.1, and MuJoCo 2.3.7. Use the [root setup guide](../../README.md) for the complete three-simulator workflow and common LeRobot export.

## Environment and smoke run

Run on Linux with a working NVIDIA EGL driver for headless rendering. This backend has its own Python 3.10 environment; do not install it into the Isaac or ManiSkill environment. From the repository root:

```bash
bash scripts/setup_libero.sh
bash scripts/generate.sh \
  --backend libero --format canonical \
  --svg_dir examples/svg --output_dir outputs/libero-smoke \
  --object_name geometry --source_desc 'draw the supplied geometric sketch' \
  --parallel 2 --device 0 --camera_height 128 --camera_width 128 \
  --no_wrist_camera
```

The setup script pins LIBERO to `8f1084e3132a39270c3a13ebe37270a43ece2a01`, creates `.venv-libero`, and installs [pinned direct dependencies](../../scripts/requirements-libero.txt). It is safe to rerun against the same clean checkout. Configuration is stored in `.venv-libero/libero-config`, without an interactive prompt or changes to `~/.libero`. The installed dependency list is saved as `.venv-libero/installed-requirements.txt`.

`--parallel` is the number of independent worker processes. Each creates its own MuJoCo environment. The supported verification target is a dedicated single-GPU host with `--device 0`. An inherited `CUDA_VISIBLE_DEVICES` mask is preserved. Masked generation proceeds only when `nvidia-smi` reports one GPU at index 0 and the mask is `0` or that GPU's full UUID. Masked multi-GPU hosts, remapped ordinals, disabled-device masks, and an unverifiable inventory fail before importing the renderer. CUDA-to-EGL ordinal mapping on multi-GPU hosts remains unverified.

Worker logs and BDDL are retained under a unique task `work/` directory. A timeout, failed worker, missing episode, duplicate identity, or ink overflow fails the run; partially generated data is not promoted to a finished dataset.

## Recording and options

Samples contain an observation **before** the associated native action. The default controller is robosuite `OSC_POSE`; its seven native values are normalized translation/rotation deltas and gripper command. `arm_controller=pd_ee` instead uses absolute pen-tip targets to calculate joint-position commands. These native controllers have different tracking behavior even when the common export has the same state/action shape.

Use `--backend_option key=value`. Booleans use JSON `true`/`false`.

| Option | Default | Meaning |
|---|---:|---|
| `arm_controller` | `osc` | `osc` or `pd_ee` |
| `max_dots` | `1500` | Ink capacity; overflow fails rather than silently truncating |
| `gain_pos` | `4.0` | Proportional gain for OSC translation |
| `end_hold_steps` | `0` | Recorded steps holding the current pen-tip position |
| `end_lift_steps` | `0` | Extra recorded lifting steps after the SVG |
| `end_lift_height` | `0.35` | Lift target Z in robot-base coordinates, metres |
| `end_lift_xy` | `last` | Lift at `last` waypoint XY or robot `base` XY |
| `end_lift_ramp` | `0` | `1` ramps lifting Z |
| `tcp_init_random` | `0` | `1` enables seeded reset-time IK initialization |
| `approach_guard_xy` | `0.0` | Positive value enables an initial pen-up approach guard |
| `approach_guard_z` | `0.04` | Guarded target minimum Z in robot-base coordinates |
| `timeout_s` | `14400` | Wall-clock timeout for all workers, seconds |

An enabled guard is applied to both controllers. Requested targets remain in `arrays/targets_world`; actual guarded targets are stored in `arrays/commanded_targets_world`, and native actions are the commands sent to MuJoCo. SVG noise and reset seeds depend on the run seed and SVG index, independently of worker scheduling.

The optional wrist view is robosuite's native `robot0_eye_in_hand` camera. Its mounting and field of view differ from ManiSkill/Isaac. Use `--no_wrist_camera` when you need to avoid presenting those viewpoints as an identical modality. Main-camera fields also describe backend observations, not a claim of pixel-identical rendering.

`--source_desc` is required and labels every SVG in the selected folder. BDDL language cannot contain parentheses, semicolons, or control characters. Use separate runs for differently labelled drawing classes. A completed trajectory does not certify visual success; coverage/error metrics are recorded for inspection.
