# Published dataset review

The [BURGER dataset on Hugging Face](https://huggingface.co/datasets/Zhaosying66/BURGER)
contains historical training datasets with more than one action convention.
The common export produced by this code release is documented separately in
[Data format](DATA_FORMAT.md). Matching array dimensions alone does not establish
compatibility.

The initial numeric/video review is pinned to dataset revision
[`1390991bca4508c9f4eb10f69bd29e21d39befba`](https://huggingface.co/datasets/Zhaosying66/BURGER/tree/1390991bca4508c9f4eb10f69bd29e21d39befba),
observed on 2026-10-28. The reference-input follow-up below uses revision
[`06baeab1a192f6d3fce43dfaca555b5ba53c9953`](https://huggingface.co/datasets/Zhaosying66/BURGER/tree/06baeab1a192f6d3fce43dfaca555b5ba53c9953).
The subsequent ManiSkill archive audit uses revision
[`29bca61ec0cb93c81b5786337fc832fe6f06a30b`](https://huggingface.co/datasets/Zhaosying66/BURGER/tree/29bca61ec0cb93c81b5786337fc832fe6f06a30b),
also observed on 2026-10-28. No published data was changed during these reviews.

## Follow-up: all three ManiSkill archive collections

All 293 expected archives are present at the latest audited revision, including
all 111 `left_right` archives. The audit read the **complete episode zero from
every task archive**, using bounded HTTP byte ranges into the uncompressed TARs.

| Collection | Tasks / complete episodes checked | Numeric rows checked | Episodes with terminal-frame defect |
|---|---:|---:|---:|
| `maniskill/single` | 132 | 101,254 | 132 |
| `maniskill/in_out` | 50 | 39,990 | 50 |
| `maniskill/left_right` | 111 | 138,119 | 111 |
| **Total** | **293** | **279,363** | **293** |

Every inspected episode has the same source-confirmed defect described below
for `up_down`: at zero-based row `T - 6`, action XYZ exactly equals the measured
world-frame TCP position, although the controller interprets it in the robot
base frame. The resulting target is displaced by **0.615 m along X**.
The following five actions are `(0, 0, 0.35, pi, 0, 0)` with `end=1`.
Ordinary drawing rows corroborate the coordinate interpretation: per-episode
median next-state tracking error in the robot-base frame ranges from
1.61 to 3.43 mm, rather than the approximately 615 mm world-frame mismatch.

For concrete examples, episode zero in `single/alarm_clock` has 840 rows and
the defective row is 834; `in_out/backpack_contains_notebook` has 842 rows and
the defective row is 836; `left_right/leaf_left_lollipop` has 850 rows and the
defective row is 844. Their next observed X positions move backward by
approximately 7.21, 4.48 and 5.26 cm respectively. The commanded 0.615 m offset
is not a claim that the robot instantly moved that distance.

All 293 episodes pass the other numeric checks performed: finite values, declared
8D state / 7D action shapes, fixed target Euler angles, contiguous frame/global
indices, episode-zero identity, exact float32 20 Hz timestamps, final-five end
flags and preceding-action end state. Quaternion norm errors are below
`1.34e-7`. Structural validity therefore does not establish correct control
targets. A known affected `up_down` episode and an in-memory corrected copy
were used as positive and negative controls for the defect detector.

This adds 293 affected episodes to the earlier 51 `up_down` samples:
**344 inspected ManiSkill episodes have the defect**. Coverage is one first
episode per task plus the earlier extra `up_down` episode, not every episode
in these collections. No new videos or complete archives were downloaded.
The selected Parquet members total 19,247,889 bytes. Each member has a recorded
SHA-256 digest, and HTTP ranges were checked against the declared archive size;
the complete TAR checksums were not verified. Image quality, archive-wide
integrity, later episodes and all remaining failure modes remain outside this
bounded numeric audit.

The common export recommendation and historical conversion limits are documented
in [Data format](DATA_FORMAT.md#why-historical-actions-have-different-widths).
Changing seven actions to eight does not repair this terminal-command defect.

## Original SVG update and action-sequence correspondence

The updated [SVG manifest](https://huggingface.co/datasets/Zhaosying66/BURGER/blob/06baeab1a192f6d3fce43dfaca555b5ba53c9953/svg_samples/manifest.json)
provides **132 original single-object SVGs, one per category**. All 132 files
(2,043,351 bytes) were downloaded and checked against their published SHA-256
values. All parse successfully and produce finite trajectories with the release
defaults, ranging from 426 to 2,531 planned waypoints. The upload explicitly
does not provide episode-to-SVG alignment.

For [the baseball reference](https://huggingface.co/datasets/Zhaosying66/BURGER/blob/06baeab1a192f6d3fce43dfaca555b5ba53c9953/svg_samples/single/baseball/baseball_001.svg),
the release's 1,493 default SVG target rows match the first 1,493 target rows of
published LIBERO baseball episode 0 **bit-for-bit after float32 conversion**.
Appending the current LIBERO tail helper with these explicit settings:

```text
end_hold_steps=0
end_lift_steps=20
end_lift_xy=last
end_lift_height=0.25
end_lift_ramp=1
```

matches its remaining 20 vertical-lift targets. Combining the targets, fixed
orientation/gripper, and final-five end flags reproduces **all 1,513 × 8 action
values exactly**, verified using float32 bit patterns and raw bytes. This is
numeric correspondence established from the data, not a filename-based
assumption that every `_001` input maps to episode zero.

The comparison uses `max_draw_dist=0.008`, fixed scaling, the default rotation,
zero XY offset, and `end_lift_z=0.085`. This SVG is discontinuous, so the last
parameter does not affect it and cannot be identified from this match.
The [README reference command](../README.md#use-a-published-reference-svg)
uses the observed tail settings explicitly. Release defaults were not changed.

The complete LIBERO generation/export command was then run on a Linux GPU server with those
tail settings, seed 7, one worker and 128×128 cameras. Its exported
**1,513 × 8 action array is also bit-identical** to the published sample,
verified independently using both float32 bit patterns and raw bytes.
Both complete 1,513-frame camera videos and the canonical/export row checks
passed. The observed state arrays differ on every row. Inspected frames show
a recognizable baseball with seam strokes, but do not match the historical
images; the final wrist view is partly cropped/occluded.

This establishes the drawing/action sequence for one LIBERO sample. It does
not establish its historical controller, initial joints/velocities, physics,
camera settings, images, or complete producer configuration. The new
single-object references also do not supply the original composition/pairing
inputs for the inspected ManiSkill spatial recordings.

Eleven SVGs have more than 1,500 nominal pen-down targets: backpack, basketball,
bus, camera, electrical fan, radio, school bus, soccer ball, suitcase,
television and wheel. The maximum is 2,045; a 3,000-dot pool exceeds the longest
2,531-waypoint reference plus the tested ending lengths. Actual ink consumption
depends on simulated pen height, and this is not evidence that all eleven
published recordings overflowed. Finer spacing or longer custom tails require
recalculating capacity.

Pen-down diagnostic subsets also need care: 17 baseball targets lie at the
0.035 m threshold within floating-point rounding. Comparing the original
float64 root-frame targets gives 940 nominal pen-down targets; LIBERO's
world-frame comparison gives 923.
The verified exported actions are unaffected. These backend-native pen-down
error means therefore do not use identical row subsets; a common frame/precision
convention remains an open question.

The updated release status also marks `maniskill/single` (132 TARs) and
`maniskill/in_out` (50 TARs) uploaded. `left_right` remains pending, with 30 of
111 TARs present at this earlier revision. Archive contents were not audited in
that SVG follow-up; the subsequent numeric archive audit is documented above.
The complete LIBERO and real Git subtrees and the ManiSkill `up_down` subtree
are unchanged between revisions. The baseball Parquet, feature metadata and
episode mapping were additionally downloaded again and matched their original
SHA-256 hashes. The earlier native-recording findings therefore still apply.

## Initial availability and organization

At the initial revision, the [release status](https://huggingface.co/datasets/Zhaosying66/BURGER/blob/1390991bca4508c9f4eb10f69bd29e21d39befba/release_status.json)
reports three uploaded collections and three pending collections:

| Collection | Status at this revision | Episodes | Frames | State/action dimensions | Cameras |
| --- | --- | ---: | ---: | --- | --- |
| `painter-libero-single-v6` | Uploaded, native folders | 14,751 | 17,386,870 | 8 / 8 | Front and wrist |
| `painter-real` | Uploaded, native folders | 200 | 272,551 | 8 / 7 | One |
| `maniskill/up_down` | Uploaded, native folders | 500 | 475,934 | 8 / 7 | One |
| `maniskill/single` | Pending, 132 planned TAR shards | 14,506 | 13,668,603 | 8 / 7 | One |
| `maniskill/left_right` | Pending, 111 planned TAR shards | 12,198 | 19,406,658 | 8 / 7 | One |
| `maniskill/in_out` | Pending, 50 planned TAR shards | 5,600 | 6,049,039 | 8 / 7 | One |

Pending counts and schemas are manifest declarations, not downloaded-data
verification. All manifests declare LeRobot v2.1, 20 Hz, and 320 × 240 RGB video.
Their subdataset counts sum correctly to their collection totals. The available
file listing contains 15,451 Parquet files and 30,202 videos, matching the three
uploaded collections' declared episode and camera counts. There is no Isaac
collection in this revision.

The repository is a collection container. Each object/task folder is an
independent LeRobot root with its own episode and global row indices. Use that
folder as the dataset root; retain its identity when combining subdatasets.
The pinned [dataset README](https://huggingface.co/datasets/Zhaosying66/BURGER/blob/1390991bca4508c9f4eb10f69bd29e21d39befba/README.md)
describes download layout and the dataset's CC BY-NC 4.0 license.

## Sample coverage and file checks

The numeric audit covers **53 complete episodes, 44,229 rows, and 3,085,692 bytes
of Parquet**: the first episode of all 50 `maniskill/up_down` tasks, one additional
alarm-clock episode, one LIBERO episode, and one real episode. This is a bounded
sample, not an audit of all 500 `up_down` episodes or every released collection.
The initial samples and their video coverage are:

| Collection / task | Episode | Rows | Video coverage |
| --- | ---: | ---: | --- |
| [LIBERO / baseball](https://huggingface.co/datasets/Zhaosying66/BURGER/blob/1390991bca4508c9f4eb10f69bd29e21d39befba/painter-libero-single-v6/baseball/data/chunk-000/episode_000000.parquet) | 0 | 1,513 | Both cameras, fully decoded |
| [ManiSkill / alarm_clock_over_laptop](https://huggingface.co/datasets/Zhaosying66/BURGER/blob/1390991bca4508c9f4eb10f69bd29e21d39befba/maniskill/up_down/alarm_clock_over_laptop/data/chunk-000/episode_000000.parquet) | 0 | 801 | Single camera, fully decoded |
| [ManiSkill / alarm_clock_over_laptop](https://huggingface.co/datasets/Zhaosying66/BURGER/blob/1390991bca4508c9f4eb10f69bd29e21d39befba/maniskill/up_down/alarm_clock_over_laptop/data/chunk-000/episode_000001.parquet) | 1 | 880 | Not downloaded |
| [ManiSkill / basketball_under_wine_bottle](https://huggingface.co/datasets/Zhaosying66/BURGER/blob/1390991bca4508c9f4eb10f69bd29e21d39befba/maniskill/up_down/basketball_under_wine_bottle/data/chunk-000/episode_000000.parquet) | 0 | 789 | Not downloaded |
| [Real / real_3class_30, “a book”](https://huggingface.co/datasets/Zhaosying66/BURGER/blob/1390991bca4508c9f4eb10f69bd29e21d39befba/painter-real/real_3class_30/data/chunk-000/episode_000000.parquet) | 0 | 1,596 | Single camera, fully decoded |

The extended numeric audit downloaded `data/chunk-000/episode_000000.parquet`
from each of the 50 tasks listed in the pinned
[`maniskill/up_down` manifest](https://huggingface.co/datasets/Zhaosying66/BURGER/blob/1390991bca4508c9f4eb10f69bd29e21d39befba/maniskill/up_down/manifest.json):
40,240 rows and 2,804,767 bytes in total. These include the two task-first episodes
already listed above; the remaining 48 tasks had no video downloaded.

All sampled action/state values are finite and have the declared dimensions.
In all 53 episodes, the last action column is one for exactly the final five
rows, and the last state column is the preceding action's value, starting at zero.
For the three episodes with video coverage, the episode metadata row counts, contiguous
frame/global indices, task indices, and exact float32 timestamps at 20 Hz were
also checked.

All four downloaded videos decode completely. They contain respectively
1,513, 1,513, 801, and 1,596 frames, matching their Parquet rows, at 320 × 240 and
20 Hz with AV1 encoding. Manual start/final frame inspection shows blank initial
views and recognizable baseball, alarm-clock-over-laptop, and book strokes.
The sampled ManiSkill final view also shows the robot retreating. These are
observations about the inspected images, not a drawing-success threshold.

## Action and state conventions

The published feature names are generic: `["state"]` and `["actions"]`.
The sampled [LIBERO metadata](https://huggingface.co/datasets/Zhaosying66/BURGER/blob/1390991bca4508c9f4eb10f69bd29e21d39befba/painter-libero-single-v6/baseball/meta/info.json),
[ManiSkill metadata](https://huggingface.co/datasets/Zhaosying66/BURGER/blob/1390991bca4508c9f4eb10f69bd29e21d39befba/maniskill/up_down/alarm_clock_over_laptop/meta/info.json),
and [real metadata](https://huggingface.co/datasets/Zhaosying66/BURGER/blob/1390991bca4508c9f4eb10f69bd29e21d39befba/painter-real/real_3class_30/meta/info.json)
do not declare per-column units, coordinate frames, quaternion order, or a
versioned Painter convention. No `painter_export.json` appears in this revision.

| Source | Observed layout and supporting evidence | Relation to the new exporter |
| --- | --- | --- |
| Published LIBERO sample | Action is `[XYZ, π, 0, 0, 1, end]`; state has XYZ, four unit-norm orientation values, and previous end. Its median action-X minus state-X is approximately 0.615 m. This agrees with the newer absolute-target export pattern. | Same column pattern as the new 8D contract. The published metadata does not identify the exact producer commit or establish every physical-reference detail. It is not an OSC-delta sample. |
| Published ManiSkill samples | Action is `[target XYZ, π, 0, 0, end]`, with no separate gripper column. The historical converter preserves native 6D `pd_ee_pose` actions and appends end; state preserves raw world TCP pose in wxyz order and appends previous end. The controller interprets all targets in robot-root coordinates; the terminal hold incorrectly supplies a world-coordinate value, as described below. | The new exporter explicitly produces pen-tip targets plus constant `pen_gripper=1`, then end: eight values. It also records the contract and uses the corrected collector. Adding a column alone does not reproduce historical trajectories. |
| Published real sample | Action has seven values with fixed `(π, 0, 0)` and terminal end. The four state orientation values are near `[±1, 0, 0, 0]`, while sampled simulated drawing orientations are near `[0, 1, 0, 0]`. | Quaternion order/reference and the real position/contact transform remain unresolved. The sample's action Z reaches −0.04 m while recorded state Z stays above 0.023 m; applying the simulator's Z interpretation would be an assumption. |

The historical ManiSkill representation is visible in the
[converter at commit `d1193612`](https://github.com/Qxjyqzw/painter/blob/d1193612b27212286afb29199fbc1324f4f57de6/painter-main/painter/data_collection/converters/convert_maniskill_data_to_lerobot.py#L194-L240).
The new exporter follows the absolute-target calculation introduced in the
[newer pipeline writer](https://github.com/Qxjyqzw/painter/blob/0dad422b752501d6cbe07b3de98b8998e7038dc5/painter_pipeline/writers.py#L137-L160)
and adds explicit validation and convention metadata.
Historical source establishes how these conventions were implemented; the
dataset does not record which exact commit produced each episode.

## Confirmed historical ManiSkill hold-frame defect

The first episode of **all 50 `up_down` tasks**, plus the additional
`alarm_clock_over_laptop` episode 1, contains one terminal hold row where
`action[:3]` equals `state[:3]` exactly. In every inspected case it occurs at
zero-based row `episode_length - 6`; the three ManiSkill examples listed above use rows
795, 874, and 783.

The source explains why this is wrong:

- [Before the fix, commit `79819d33`, lines 954–965](https://github.com/Qxjyqzw/painter/blob/79819d33bfd4419de22b4278e28286fccc305101/painter/data_collection/maniskill_environments/draw_svg_parallel.py#L954-L965)
  copies the world-frame TCP position into a robot-root-frame action after the
  SVG ends.
- [Commit `11893225`, lines 977–986](https://github.com/Qxjyqzw/painter/blob/1189322566256ce257a14a830f4ecefa36857c50/painter/data_collection/maniskill_environments/draw_svg_parallel.py#L977-L986)
  corrects the fallback using `root_pose.inv() * tcp_pose`.

For a robot base at world X = −0.615 m, the erroneous hold command is displaced
by **0.615 m along X**. In the first alarm-clock episode, row 795 contains:

```text
state XYZ  = (-0.112135, 0.107203, 0.040889)  # world
action XYZ = (-0.112135, 0.107203, 0.040889)  # interpreted as robot root
```

The next recorded state moves backward in X by 5.07 cm. The other two initial
ManiSkill samples move backward by 4.37 cm and 4.07 cm. Each inspected episode
then records the historical five
raise-arm commands `(0, 0, 0.35, π, 0, 0)` with end set to one. During the normal
drawing portion, interpreting targets in the robot-root frame gives a median
next-state error of approximately 3.30–3.39 mm; interpreting them as world
coordinates gives approximately 615 mm. This corroborates the source diagnosis.

The 0.615 m erroneous command offset occurs in all 51 inspected `up_down`
episodes. The immediate physical response varies; a backward first step is not
a universal signature. The initial audit does not establish prevalence among
the remaining 449 `up_down` episodes. The subsequent archive audit above confirms
the same defect in one episode from every `single`, `in_out` and `left_right` task.
Structural finite-value, timestamp, and video-decoding checks do not detect this
physical command error.

The cleaned collector uses root-frame hold targets. A direct check of six fresh
GPU episodes, three each in serial and parallel collection, found no
world-as-root hold or historical five-command raise sequence. The final native
hold target differs from the measured root-frame TCP position by less than
`9.71e-7 m`; the recorded world target differs from the world TCP position by
less than `9.85e-7 m`. These are checks of hold-coordinate consistency, not
drawing-quality scores.

The historical dataset is left unchanged. Relabeling the erroneous action would
not repair the subsequent observations, which were recorded after the actual
command. Excluding or regenerating affected tails, and recomputing end signals,
requires a separate dataset/training decision.

## Reproduction limits and open decisions

- The updated 132-file SVG sample supplies original drawing inputs but no
  episode alignment or complete acquisition recipe. Sampled
  `episode_mapping.jsonl` files map release indices to original indices only;
  they do not identify a source SVG, seed, simulator version or generation
  configuration. No canonical HDF5 recordings were supplied.
- Only episode zero from each of the 293 TARs was numerically audited; later
  episodes, their videos and complete archive checksums were not verified.
  The SVG sample is separate and does not establish complete source inputs
  or episode alignment for the TAR collections.
- Collection names and the dataset README provide simulator/real-robot
  provenance. Exact source commits and original acquisition settings are not
  supplied for the sampled episodes.
- The sampled ManiSkill initial positions vary and its historical tail behavior
  differs from the current collector. Default settings in this code release
  therefore do not promise exact regeneration of those recordings.
- **Unresolved:** the approved adapters and normalization for mixing historical
  7D datasets with the new 8D contract, treatment of defective historical tails,
  and the real dataset's quaternion and position/contact conventions.
- **Unresolved:** complete drawing inputs, episode alignment and per-collection
  generation recipes needed for controlled state/image reproduction. The
  verified baseball action match does not resolve these for other episodes.

The new code can generate and validate new datasets in its documented contract.
It can reproduce the verified baseball action sequence with the explicit
settings above. It does not silently transform or repair published collections.
