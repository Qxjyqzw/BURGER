# BURGER usage guide

This guide covers installation, verification and data generation in detail.
For a project overview, see the [README](../README.md). Run every command from
the repository root:

```bash
git clone https://github.com/Qxjyqzw/BURGER.git
cd BURGER
```

## 1. Prepare the host

Run the commands below from the repository
root. Each environment is isolated; no activation is needed.

Simulation needs **Linux x86_64 with an NVIDIA GPU**. Isaac also needs an
RTX-capable GPU, glibc ≥ 2.35 (Ubuntu 22.04+), and a driver compatible with
Isaac Sim 6.1. ManiSkill needs Vulkan; headless LIBERO uses EGL.
Allow 50 GB free for Isaac alone or 150 GB for full verification.

On Ubuntu:

In a root shell (common in containers), omit `sudo` from these two commands.

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential pkg-config ffmpeg \
  libgl1 libegl1 libglu1-mesa libvulkan1 libglib2.0-0 libatomic1 libxt6
nvidia-smi
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed.
For the version used by the verification recipe:

```bash
curl -LsSf https://astral.sh/uv/0.9.6/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

The scripts use uv to create the required Python environments. Preprocessing
and export can also run on a CPU-only machine.

## 2. Set up shared tools

```bash
bash scripts/setup_cpu.sh
bash scripts/setup_export.sh
.venv-cpu/bin/python -m pytest -q -m 'not export'
.venv-lerobot-export/bin/python -m pytest -q \
  tests/test_data_contract.py tests/test_svg_trajectory.py tests/test_export.py
```

The first environment provides preprocessing and canonical checks. The second
uses **LeRobot 0.3.3 / dataset format v2.1** and runs a real video/Parquet
roundtrip test. Newer LeRobot versions may require dataset migration.

## 3. Install one simulator

| Backend | Setup command | Python/environment |
|---|---|---|
| ManiSkill | `bash scripts/setup_maniskill.sh` | 3.10, `.venv-maniskill` |
| LIBERO | `bash scripts/setup_libero.sh` | 3.10, `.venv-libero` |
| Isaac | `bash scripts/setup_isaac.sh` | 3.12, `.venv-isaac61` |

ManiSkill is pinned to 3.0.1 and SAPIEN to 3.0.3. LIBERO uses robosuite 1.4.1,
MuJoCo 2.3.7 and a pinned LIBERO revision. Isaac uses Sim 6.1.0 and a pinned
Isaac Lab revision. Details: [LIBERO](backends/LIBERO.md),
[Isaac](backends/ISAAC.md).

Isaac's first launch requires acceptance of the
[NVIDIA Omniverse EULA](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html).
After reviewing and accepting those terms, set this in the shell used for
generation:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

For a remote or scheduled run, pass this environment variable to the process
that starts generation. The worker is noninteractive and cannot answer a
license prompt. The scripts do not set acceptance automatically.

## 4. Verify generation and readback

Run the command for the backend you installed:

```bash
bash scripts/verify_backend.sh maniskill
# Alternatives:
bash scripts/verify_backend.sh libero
bash scripts/verify_backend.sh isaac
```

Each generates three unequal-length drawings with two environments/workers,
checks episode counts and numeric structure, exports LeRobot, and decodes the
exported dataset. This exercises partial resets and a partially filled final
batch. For a serial comparison, supply a separate output folder and `1`:

```bash
bash scripts/verify_backend.sh maniskill validation/maniskill-serial 1
```

Outputs and environments are ignored by Git. Structural checks do not establish
drawing quality or cross-simulator visual equivalence.

## 5. Generate your own data

Place SVGs for **one language instruction** in a directory. This example uses
the supplied infrastructure-test drawings:

```bash
bash scripts/generate.sh \
  --backend maniskill \
  --svg_dir examples/svg --output_dir outputs \
  --object_name geometric_shapes \
  --source_desc "draw the supplied geometric sketch" \
  --parallel 2 --seed 7 --format canonical,lerobot \
  --lerobot_repo_name shapes-v1
```

Replace the backend with `libero` or `isaac`. The wrapper selects its environment
and delegates export to `.venv-lerobot-export`.

```text
outputs/geometric_shapes/maniskill/
  geometric_shapes.canonical.h5
  generation_summary.json
  shapes-v1/
    meta/   data/   videos/
    painter_export.json
  work/
```

Canonical data retains native actions and provenance for re-export. LeRobot
uses the adopted common convention:

```text
state  (8): pen-tip XYZ in ManiSkill world, quaternion WXYZ, previous end flag
action (8): absolute target XYZ in robot-base frame, π, 0, 0, pen/gripper=1, end flag
```

Units are metres/radians. The default end flag covers the last five recorded
steps; it is not a measured success label. Read [DATA_FORMAT.md](DATA_FORMAT.md).
The existing contract identifier `painter-absolute-target-v1` and
`painter_export.json` filename are preserved for data compatibility.
Old exports may have the same vector length with different meanings.
Published legacy ManiSkill datasets use seven-value actions and one camera;
the common release export uses eight-value actions and defaults to two cameras.
Do not mix those schemas without an explicitly reviewed transform.

Every input receives exactly `--source_desc`; use separate runs for different
instructions. Seeds, input order, options and source frequency are recorded.
Appearance may be fixed per environment, so changing `--parallel` can change it.
Randomized appearance requires an explicit `--appearance` preset.

Supported SVG geometry is ordered line paths (`M/L/H/V/Z`, including relative
forms). Curves, transforms and unsupported geometry fail explicitly. The
historical fixed scale depends on coordinates; fixtures use a 512×512 viewBox.
ViewBox dimensions alone do not normalize a drawing's scale.

Existing output requires `--overwrite`. Failed generation does not replace a
previous canonical file. Check exit status and `generation_summary.json`.
Keep the source SVGs, generation command, code commit, canonical file, and the
environment's `installed-requirements.txt` with your dataset provenance.
A LeRobot-only upload does not retain the complete generation inputs.

## Use a published reference SVG

The [original SVG sample](https://huggingface.co/datasets/Zhaosying66/BURGER/tree/06baeab1a192f6d3fce43dfaca555b5ba53c9953/svg_samples)
contains one drawing per single-object category, with original bytes and
checksums. Download the verified baseball input:

```bash
mkdir -p outputs/reference-inputs/baseball
curl --fail --location \
  https://huggingface.co/datasets/Zhaosying66/BURGER/resolve/06baeab1a192f6d3fce43dfaca555b5ba53c9953/svg_samples/single/baseball/baseball_001.svg \
  --output outputs/reference-inputs/baseball/baseball_001.svg
.venv-cpu/bin/python - <<'PY'
import hashlib
from pathlib import Path
p = Path("outputs/reference-inputs/baseball/baseball_001.svg")
assert hashlib.sha256(p.read_bytes()).hexdigest() == \
    "e7b8c02cb70a247d5ef415439a15c994aa69131c0ae9eb410356e986532c6a98"
print("Reference SVG checksum verified")
PY
.venv-cpu/bin/python -m burger.pipeline.svg_trajectory \
  --svg outputs/reference-inputs/baseball/baseball_001.svg
```

This produces 1,493 planned waypoints. With LIBERO installed, these explicit
ending options reproduce the **1,513 target/action rows** in the inspected
published baseball episode:

```bash
bash scripts/generate.sh \
  --backend libero --svg_dir outputs/reference-inputs/baseball \
  --output_dir outputs/reference-run --object_name baseball_reference \
  --source_desc "draw a baseball" --parallel 1 --seed 7 \
  --format canonical,lerobot --lerobot_repo_name baseball-reference-export \
  --backend_option end_hold_steps=0 --backend_option end_lift_steps=20 \
  --backend_option end_lift_xy=last --backend_option end_lift_height=0.25 \
  --backend_option end_lift_ramp=1
```

The 20-step lift is specific to this observed action sequence. Physics,
initialization and camera settings remain release defaults; this is not a
claim to reproduce the published states or images. To test this SVG with
ManiSkill or Isaac, select that backend and omit the LIBERO ending options.

All 132 reference SVGs pass parsing and finite-waypoint checks. Eleven have
more than 1,500 planned pen-down targets. For these larger inputs at the default
spacing, use `--backend_option max_dots=3000`; finer spacing or longer tails
require recalculating capacity. Actual ink use depends on simulated pen motion.
Keep each category's inputs and language instruction in a separate run.

## Optional: PNG → SVG

Use dark strokes on a light background. Alpha is composited onto white; add
`--invert` for light strokes on a dark background.

```bash
.venv-cpu/bin/python -m burger.preprocess.skeletonize \
  --input_folder my_sketches --output_folder outputs/skeletons
.venv-cpu/bin/python -m burger.preprocess.svg_convert \
  --input_folder outputs/skeletons --output_folder outputs/svgs --spacing 5
```

Inspect the results before simulation. Cleanup, ordering, short-path filtering
and circular smoothing are heuristics; paper-specific settings await confirmation.
Disable circular smoothing with `--no_smooth_circular_paths`. Empty results fail.

## Optional: re-export existing data

```bash
.venv-lerobot-export/bin/python -m burger export \
  --canonical outputs/geometric_shapes/maniskill/geometric_shapes.canonical.h5 \
  --format lerobot --out_dir outputs/reexport --lerobot_repo_name shapes-v1
```

`--format libero` is a compatibility export, not a replayable LIBERO task or
cross-backend action normalization. Its limitations are in the data contract.
For the supported legacy ManiSkill importer:

```bash
.venv-cpu/bin/python -m burger import-maniskill --help
```

## Run the complete verification

On a Linux GPU workstation or remote GPU server, install the system packages
and uv from step 1, then run from the repository root. The verification script
sets up the Python environments, runs CPU/export tests, and generates and
reads back data with all three simulators, both serially and in parallel.
Isaac requires an RTX-capable GPU and the explicit license acceptance described
above.

```bash
bash scripts/verify_all.sh
```

To check selected simulators:

```bash
BURGER_VERIFY_BACKENDS="maniskill libero" bash scripts/verify_all.sh
```

Each run records host information, dependency versions, per-step logs and a
PASS/FAIL summary under `validation/verification/run-*`. A failed step returns
a nonzero exit status; the runner still attempts the other selected backends.
The [verification report](VERIFICATION.md) lists the hardware, versions
and checks already completed for this release.

## Troubleshooting

- Missing environment: run the matching setup script.
- Multi-GPU host: select the GPU with `--device N` and leave
  `CUDA_VISIBLE_DEVICES` unset. Generation refuses an inherited mask on
  multi-GPU hosts because it cannot map it safely to the rendering GPU.
- No CUDA/Vulkan/EGL: check NVIDIA driver/runtime compatibility and host GPU
  configuration. A CPU-only install cannot validate rendering.
- Video decoding: use the pinned export environment; verification uses PyAV.
- First Isaac launch: RTX shader compilation can take several minutes. Check
  the worker log for compilation progress before treating a cold start as stuck.
- Unsupported SVG or full ink pool: the run fails rather than dropping geometry.
- Failed run: inspect its full log and exit status before using any output.

See the [verification report](VERIFICATION.md) and
[third-party notices](../THIRD_PARTY_NOTICES.md). Bundled assets have separate
license terms from the source code.
