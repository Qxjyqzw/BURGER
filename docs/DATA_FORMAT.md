# Data format

Use the canonical HDF5 file as the source recording and **LeRobot v2.1 with
`painter-absolute-target-v1`** for the common export across ManiSkill, LIBERO and
Isaac. This release adopts the absolute-target convention from
`isaac_svg_pipeline`; older `libero_data` LeRobot exports may have different action
dimensions or different meanings for the same eight columns.

## LeRobot contract

Every LeRobot export contains `painter_export.json`. Its `data_contract` records the
convention version, column names, units, coordinate frames, fixed target
orientation and end-signal rule. `meta/info.json` also contains the individual
state and action column names.

| Feature | Columns | Meaning |
| --- | --- | --- |
| `observation.state` | `pen_tip_x, pen_tip_y, pen_tip_z, eef_qw, eef_qx, eef_qy, eef_qz, previous_end` | Eight float32 values. Position is the pen tip in ManiSkill world coordinates. Quaternion is the recorded end-effector orientation in wxyz order. |
| `action` | `target_x, target_y, target_z, target_rx, target_ry, target_rz, pen_gripper, end` | Eight float32 values. XYZ is an absolute pen-tip target in the robot-base frame; Euler XYZ is fixed at `(π, 0, 0)` and `pen_gripper` is fixed at `1`. |
| `observation.image` | RGB video | Third-person camera, display/top-down row order. |
| `observation.wrist_image` | RGB video, optional | Present only when every episode contains a wrist camera. |
| `task` | Language instruction | Preserved from canonical dataset metadata by LeRobot. |

Distances are metres and angles are radians. The ManiSkill reference robot base is
`(-0.615, 0, 0)`. For each recorded episode:

```text
backend_base_world = pen_tip_world - pen_tip_painter
state_xyz = pen_tip_painter + (-0.615, 0, 0)
action_xyz = targets_world - backend_base_world
```

The constant `pen_gripper=1` is part of this export convention; it is not a
measured pen-contact signal.

The transform is a **constant translation**, checked across all rows of each
episode. Quaternion values are preserved and checked for unit norm; the exporter
does not rotate them, change their sign or reinterpret their physical reference.
This convention supports the fixed-base, aligned-axis drawing scenes included in
the release. It is not a general conversion for rotated or moving robot bases.

The native simulator action is retained in canonical data. It is not copied to the
LeRobot `action` column. LIBERO OSC deltas and ManiSkill/Isaac control commands
therefore remain available for inspection without being mixed into one training
action field.

### Why historical actions have different widths

LeRobot permits different action dimensions; the difference comes from the
producer/exporter conventions.

| Recording | Action layout | Evidence |
|---|---|---|
| Historical ManiSkill | `[x, y, z, rx, ry, rz, end]` | The old converter copies the six-value absolute `pd_ee_pose` action and appends `end`. The `panda_stick` action has no independent gripper command. |
| Inspected real-robot recording | Seven values with the same observed pose-plus-end pattern | The sample establishes the width and values. Its producer's coordinate, tool/contact and quaternion conventions remain unconfirmed. |
| Inspected published LIBERO baseball and the new common export | `[x, y, z, rx, ry, rz, pen_gripper, end]` | The common exporter constructs absolute pen-tip targets explicitly and inserts constant `pen_gripper=1`. The baseball sample matches this action layout and sequence. |

Use the new eight-value absolute-target contract for newly generated data from
all three simulators. The extra gripper field is a compatibility convention,
not evidence of pen contact or an additional measured degree of freedom.
Preserve simulator-native commands separately in canonical recordings.

For historical data, inserting `1` immediately before `end` aligns the action
width only after target coordinates, tool reference, orientation and units
have been verified. Keep `end` last and preserve the per-episode previous-end
rule. Review or regenerate affected ManiSkill terminal tails separately:
changing the action width does not repair erroneous commands or the resulting
observations. For real data, record the unresolved transforms rather than
guessing them. Recompute training normalization statistics after an approved
conversion.

The state contract also matters: the new state position is a pen tip in the
documented reference world, while historical metadata names its columns only
`state`. The exporter preserves recorded quaternion values; it does not
establish a shared physical orientation reference between robots.

### Episode ending and time base

`end=1` for the final `--end_signal_length` frames, default 5; shorter episodes have
`end=1` throughout. `previous_end` starts at zero separately for every episode and
then contains the preceding action's end value. This is an episode-ending
convention, not a measurement of drawing success. No explicit success filter is
applied by the exporter.

The recorded control frequency becomes LeRobot FPS. Export does not resample
trajectories or harmonize control frequencies across datasets. All episodes in one
export must use the declared cameras and matching image dimensions. LeRobot video
compression is lossy; keep canonical images for exact pixel comparisons.

## Canonical recording

Canonical files use `painter_canonical_version="1.0"` and one group per episode:

```text
episodes/ep_<index>/
  obs/agentview_rgb       uint8, (T, H, W, 3), display/top-down order
  obs/eye_in_hand_rgb     optional uint8, (T, h, w, 3)
  obs/eef_pos             (T, 3), backend-native world coordinates
  obs/eef_quat_wxyz       (T, 4), recorded end-effector quaternion
  obs/pen_tip_world       (T, 3), pen tip in backend world coordinates
  obs/pen_tip_painter     (T, 3), pen tip in robot-base coordinates
  actions                (T, A), native simulator actions
  arrays/targets_world   (T, 3), pen-tip target in backend world coordinates
```

Other optional observations include joints, gripper position and ink count;
episode arrays may include waypoints, ink points and simulator state. The root
metadata records `backend`, `robot`, `language`, `control_freq`, `cameras`,
`action_space`, `backend_config`, `generation`, episode/step totals and image row
order. Episode metadata contains its source SVG and available collection metrics.

New files are marked `painter_write_complete=true` only after successful writing.
Before export, validation checks schema/version, completion, nonempty episodes,
counts, numeric finiteness, temporal lengths, quaternion norms, camera consistency,
and the extra position fields required by LeRobot. Existing schema-1.0 files
without a completion flag are accepted only when those other checks pass.

Exports are built in temporary sibling locations. A failed conversion keeps an
existing complete output intact; the output is replaced only after writing succeeds.
`--overwrite` is still required to replace an existing output.

## Export an existing canonical file

From the repository root, create the CPU export environment once:

```bash
bash scripts/setup_export.sh
source .venv-lerobot-export/bin/activate

python -m burger export \
  --canonical outputs/square/maniskill/square.canonical.h5 \
  --format lerobot \
  --out_dir outputs/square/maniskill \
  --lerobot_repo_name square-maniskill
```

Replace the canonical path with the path printed by your generation run. The
dataset is written to `outputs/square/maniskill/square-maniskill/`.
Paths are resolved relative to the caller's directory, including when another
simulator environment delegates export to `.venv-lerobot-export`. For a custom
environment, set `PAINTER_LEROBOT_PYTHON` to its Python executable.

To verify the format writer independently of a simulator:

```bash
python -m pytest -q \
  tests/test_data_contract.py tests/test_svg_trajectory.py tests/test_export.py
```

`test_lerobot_roundtrip` writes two episodes, encodes third-person and wrist
videos, reads them through the official LeRobot dataset loader, and verifies
state, actions, end-signal reset, image orientation and the contract metadata.
That test skips when LeRobot is absent; a skip does not verify LeRobot export.

## Legacy LIBERO-style HDF5

`--format libero` remains available for compatibility with the `data/demo_i/...`
training layout. It is **not a supported simulator replay format**. This export
records `painter_simulator_replayable=false` and the action semantics explicitly.

- `auto` keeps native LIBERO OSC actions; for other backends it derives an
  approximate translation delta, `clip(delta_gain * (target_world - eef_world) /
  0.05, -1, 1)`, with zero rotation and fixed closed gripper. Clipping and the
  backend-specific end-effector reference remain limitations of this legacy path.
- `native` retains the native action shape and meaning. This can be six absolute
  pose values rather than seven normalized LIBERO values.
- Images use LIBERO's OpenGL row order. Missing wrist images, joint and gripper
  values are zero-filled and listed in `painter_missing`. These zeros are not
  observations.
- `rewards` and `dones` use a synthetic terminal marker. `rewards[-1]=1` does not
  establish task success; available collection metadata remains separate.

Use the versioned LeRobot contract when a common action/state representation is
needed. Do not concatenate old and new exports solely because their array shapes
match. Re-export canonical sources and verify the consumer's expected frame,
action semantics, normalization statistics, frequency and end-signal handling.

## Existing ManiSkill recordings

`python -m burger import-maniskill --help` describes the advanced
CPU-only importer for older `DrawSVGParallel-v1` absolute-`pd_ee_pose` recordings.
It requires one source `.h5`, its JSON sidecar, `dataset_summary.json`, the actual
robot-base world position, and by default a reviewed `filter_manifest.json`.

`--allow_unreviewed_summary_selection` explicitly permits using the old summary's
parallel-tail counts without a review manifest. This does not validate drawing
quality. The importer records the selection source and validates dimensions,
numeric values, pose convention and available success flags. It supports the
single-camera `panda_stick` recording layout; it is not a universal ManiSkill
dataset converter.

The importer writes canonical data and a legacy LIBERO-style file, leaving the
source read-only. Re-export the canonical file with `--format lerobot` for the new
contract. Imported output cannot be replayed as MuJoCo state.

## SVG input contract

The trajectory builder accepts paths made of line segments, including relative
line commands and closed paths. It reads XML attributes and namespaces correctly
and preserves document path order. Each path's relative initial move starts at
that path's own origin.

Drawing uses path centerlines in the source coordinate system; fill/stroke
appearance and SVG viewport dimensions do not define robot motion. `fixed`
scaling maps 1024 coordinate units to 0.45 metres. Legacy `max` scaling is
available explicitly and requires a positive largest coordinate.

Flatten curves and arcs before generation. Apply transforms and clipping to the
coordinates first, and remove hidden geometry. Non-path drawing elements,
transforms, CSS geometry, clipping and masks are rejected instead of silently
changing the drawing. Curve sampling tolerance is intentionally not chosen by
this release.
