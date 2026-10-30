# Isaac Sim backend

Generate drawing demonstrations with Isaac Sim 6.1, Isaac Lab's PhysX backend, and RTX cameras. Use the [root setup guide](../../README.md) for the complete three-simulator workflow and LeRobot export.

## Environment and smoke run

Use Linux x86_64 with glibc 2.35 or newer (Ubuntu 22.04+), an RTX-capable NVIDIA GPU, a compatible NVIDIA driver, and at least 50 GB free disk. Headless mode still requires Vulkan/RTX rendering; a CUDA-only GPU is insufficient. Run from the repository root:

```bash
bash scripts/setup_isaac.sh
```

Before generation, review the [NVIDIA Omniverse license agreement](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html). **Only if you accept those terms**, explicitly set this variable in the shell that will launch generation:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

The worker has no interactive input, so it cannot answer NVIDIA's acceptance prompt. A prior acceptance recorded by NVIDIA's interactive launcher in this same installation is also recognized. The setup and generation scripts do not accept terms automatically. Then run:

```bash
bash scripts/generate.sh \
  --backend isaac --format canonical \
  --svg_dir examples/svg --output_dir outputs/isaac-smoke \
  --object_name geometry --source_desc 'draw the supplied geometric sketch' \
  --parallel 2 --device 0 --camera_height 128 --camera_width 128 \
  --no_wrist_camera
```

The setup script creates `.venv-isaac61` with Python 3.12 and checks out Isaac Lab at `eae3f3cfcdc61342c6fd5b12d5ed9316e97a92e7`. It verifies an existing checkout before reuse, registers the repository for imports, and writes `.venv-isaac61/installed-requirements.txt`. Run the command again with a different output directory and `--parallel 1` to exercise sequential reuse of an environment.

The supported verification target is a dedicated single-GPU host with `--device 0`. An inherited `CUDA_VISIBLE_DEVICES` mask is preserved. Masked generation proceeds only when `nvidia-smi` reports one GPU at index 0 and the mask is `0` or that GPU's full UUID. Masked multi-GPU hosts, remapped ordinals, disabled-device masks, and an unverifiable inventory fail before Kit launches. CUDA-to-Vulkan ordinal mapping on multi-GPU hosts remains unverified.

`--device` is passed to physics and RTX rendering; small auxiliary driver contexts may still appear on other GPUs on an unmasked host. Logs and partial files live in a unique directory under the task's `work/`; incomplete or mismatched worker output is rejected. The launcher terminates worker processes on timeout, Ctrl-C, or SIGTERM.

The first launch can spend several minutes compiling RTX shaders. The worker log prints the detailed Kit log path near startup; that log reports compilation progress while the worker's standard output is quiet.

The documented serial and parallel smoke runs passed on Ubuntu 24.04.5 with an L40S and driver 580.178.04, including canonical/LeRobot readback and both camera streams. The three fixtures produced 33, 42, and 96 samples; inspected initial and final frames showed clean reset canvases and visible strokes. Direct measurements also verified the fixed robot-base pose through slot reuse. See the [verification record](../VERIFICATION.md) for versions and test scope.

## Recording and options

Every sample pairs an observation **before** its action with that action. Native actions are six values: absolute pen-tip position in the robot-base frame, then XYZ Euler angles. The fixed target orientation is `[pi, 0, 0]`. The robot base is fixed at `[-0.615, 0, 0]` in the local world frame; initialization fails if PhysX does not create a fixed-base articulation.

Pass backend controls using `--backend_option key=value`, with JSON `true`/`false` for booleans. Unknown names and invalid scalar types fail early. Common controls:

| Option | Default | Meaning |
|---|---:|---|
| `max_dots` | `1500` | Ink capacity per episode; overflow fails the run instead of dropping ink |
| `hold_steps` | `1` | Recorded steps holding the current pen tip after the SVG |
| `raise_steps` | `0` | Additional recorded raising steps |
| `raise_height` | `0.35` | Raising target Z, metres in robot-base coordinates |
| `raise_xy` | `base` | `base` or `last` waypoint XY |
| `raise_ramp` | `0` | `1` ramps raising Z instead of applying a fixed target |
| `tcp_init_random` | `0` | `1` enables seeded reset-time IK initialization |
| `approach_guard_xy` | `0.0` | Positive distance enables an initial pen-up approach guard |
| `approach_guard_z` | `0.04` | Guarded target minimum Z in the robot-base frame |
| `antialiasing` | `FXAA` | `FXAA` or `Off`; temporal modes are excluded to avoid previous-episode ghosting |
| `render_mode` | `RealTimePathTracing` | Supported renderer, using RTX compatibility controls with GI and rendering caches disabled |
| `sampled_lighting` | `true` | Required by this RTX mode; `false` is rejected because the renderer re-enables it |
| `samples_per_pixel` | `1` | Direct-light samples per pixel: `1`, `2`, `4`, or `8`; this does not accumulate whole frames |
| `timeout_s` | `14400` | Worker wall-clock timeout, seconds |

With an approach guard, `arrays/targets_world` retains the requested trajectory for the common LeRobot export. Native `actions` and `arrays/commanded_targets_world` record the guarded commands sent to the controller. These are explicitly different quantities.

SVG translation noise and random TCP initialization are seeded by the run seed and SVG index. Appearance randomization currently draws once per vectorized environment, with global lighting taken from the first appearance; changing `--parallel` can therefore change appearance.

Trajectory completion means every requested action was executed. A research-specific drawing-quality threshold and common camera calibration remain open choices. Bundled scene assets have separate terms in [third-party notices](../../THIRD_PARTY_NOTICES.md).
