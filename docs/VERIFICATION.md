# Release verification

The results below record the release's installation, runtime and published-data checks.

Verified October 28, 2026. ManiSkill and LIBERO GPU checks ran in a fresh Linux
GPU container: **Ubuntu 22.04.5, Linux x86_64, one NVIDIA L40S, driver
580.95.05**. Isaac checks used fresh isolated environments on a Linux GPU server:
**Ubuntu 24.04.5, one NVIDIA L40S, driver 580.178.04**. Both followed the documented
system-package and environment setup. The same GPU server also passed LIBERO checks.

| Check | Result |
|---|---|
| Fresh CPU and LeRobot environments; repeated setup | Passed locally on macOS ARM64 and on both Linux GPU hosts |
| Simulator-independent regression suite | Passed: 110 tests; the real-export marker is tested separately |
| Real LeRobot writer and official PyAV reader | Passed; numeric arrays, end flags, two cameras, alternate world origins, image orientation |
| Source/wheel packaging, shell syntax, Python compilation | Passed |
| ManiSkill installation | Passed: Python 3.10.13, ManiSkill 3.0.1, SAPIEN 3.0.3, Torch 2.6.0+cu124, NumPy 1.26.4 |
| ManiSkill serial and parallel generation + LeRobot readback | Passed after correcting repeated articulation-state application; each run has 3 episodes, lengths 33/42/96, 171 rows |
| LIBERO installation | Passed: Python 3.10.13, MuJoCo 2.3.7, robosuite 1.4.1, NumPy 1.26.4; exactly one OpenCV provider after the setup correction |
| LIBERO serial and parallel generation + LeRobot readback | Passed; each run has 3 episodes, lengths 32/41/95, 168 rows |
| Isaac installation | Passed: Python 3.12.3, Isaac Sim 6.1.0.0, pinned Isaac Lab, Torch 2.11.0+cu128; installation also passed on Ubuntu 22.04 with Python 3.12.13 |
| Isaac serial and parallel generation + LeRobot readback | Passed with corrected articulation and renderer settings: 3 episodes, lengths 33/42/96, 171 rows each |
| Isaac physical base and reset diagnostic | Passed: fixed base throughout 131 post-physics samples, maximum base-position error `2.29e-7 m` and quaternion component error `2.39e-7`; reset joint states matched exactly |
| Optional ManiSkill appearance / randomized starting pose | Passed separate two-environment checks: 2 episodes / 75 canonical rows each, valid cameras, blank resets and random starting poses within configured bounds |

The Ubuntu 22.04 CPU/export environments used Python 3.10.13; the Ubuntu 24.04
environments used Python 3.10.22. Export used LeRobot 0.3.3 and Torch 2.7.1+cpu.
The three unequal-length fixture drawings exercise slot reuse and a partially
filled final batch. All three simulator checks use 20 Hz; this does not imply
that arbitrary datasets or other configurations use the same frequency.

## Independent data inspection

For successful ManiSkill, LIBERO and Isaac runs, inspection separately recomputed every
exported state/action row from the canonical arrays, checked episode/frame IDs
and timestamps, and decoded every frame of every exported camera video.
Numeric export differences were below `1e-7` from float32 rounding.

LIBERO serial/parallel low-dimensional arrays matched exactly. ManiSkill
serial/parallel numeric differences were at most approximately `2.1e-6`;
its reused slot began with the same robot pose, blank camera views and zero ink
as its first episode. Ink counts were monotonic within each episode.
Isaac's serial/parallel numeric differences were at most `1.69e-5`; ink arrays
matched exactly, and the reused slot's initial TCP pose and quaternion matched
its first episode exactly. All twelve Isaac episode/camera videos decoded fully.

Manual inspection of initial/final canonical camera frames found blank resets
and recognizable red fixture strokes, with rounded corners and tracking error.
The initial Isaac renderer configuration produced substantial white speckling
in the raw camera frames. A fresh-camera comparison isolated RTX compatibility
mode as the correction. The final serial and parallel runs used that mode with
rendering caches disabled; inspected start/final frames have clear backgrounds,
blank resets and visible red strokes. Initial images are not pixel-identical
across slots or reuse, so these results do not establish pixel determinism.
These checks do **not** establish a drawing-success threshold or reproduce the
paper's dataset. The recorded metrics remain available; changing controller
gains or accepting/rejecting drawings requires separate research decisions.

## Runtime findings corrected during verification

- ManiSkill's full GPU state application after moving ink repeatedly reapplied
  the fixed robot base and accumulated pose drift. Applying only rigid-actor
  changes resolved serial and parallel failures without loosening validation.
- LIBERO's dependency graph installed two packages providing `cv2`. Setup now
  installs one provider and repairs an existing overlapping installation.
- NVIDIA's noninteractive first-launch EULA requirement is explicit in both the
  root and Isaac READMEs. Installation success is not reported as Isaac runtime
  success.
- Isaac's rendering plugins require Ubuntu's `libxt6` package, now included in
  the documented GPU-server setup.
- The imported Isaac robot carries both Physics and Newton articulation-root
  schemas. The supported PhysX fragment configuration relocates them together
  and preserves referenced USD properties, avoiding duplicate roots and a
  failure in the legacy property writer. The physical diagnostic verified the
  configured base pose after initialization and partial resets.
- Isaac's real-time path tracer requires compatibility mode for the existing
  indirect-diffuse and reflection controls to take effect. Enabling it removed
  the observed speckles. The supported rendering options now reflect the
  installed runtime: non-temporal FXAA or no antialiasing, sampled direct
  lighting, and a clearly labeled direct-light sample count.

The generic `scripts/verify_all.sh` runner retains host information, dependency
snapshots, per-step logs and PASS/FAIL results. New invocations write a fresh
`validation/verification/run-*` directory and attempt each selected backend
independently with a timeout. The recorded run IDs above refer to the October 28
executions. Their original logs, including initial failures and later successful
reruns, remain retained; renaming the runner did not repeat those checks.

These fixture checks verify generation and serialization, not an author-approved
drawing-quality threshold or the paper's acquisition recipe. The separate
[published-dataset review](PUBLISHED_DATASET_REVIEW.md) compares the available
historical recordings and documents their reproduction limits.

## Uploaded reference SVG follow-up

At Hugging Face revision `06baeab1a192f6d3fce43dfaca555b5ba53c9953`, all
132 original SVG samples (2,043,351 bytes) passed SHA-256 validation, parsing
and finite-waypoint construction using the unchanged release code.
The README's pinned download, checksum and waypoint-inspection commands were
also executed successfully.

The baseball reference was then generated on the existing Ubuntu 24.04 / L40S
GPU server using Isaac's release defaults: seed 7, one environment, 128×128 front
and wrist cameras, and the instruction “draw a baseball.” The resulting
1,494 rows are 1,493 planned waypoints plus one correctly framed hold.
Generation, LeRobot readback, every numeric row and both complete 1,494-frame
videos passed independent checks. Ink increased monotonically from zero to
940, matching the 940 planned pen-down targets. Inspected frames show a blank
start and a recognizable baseball with seam strokes; the wrist view crops a
small part of the right edge. This verifies a real reference input without
claiming a historical Isaac recording or matched images.

A separate CPU comparison uses the uploaded SVG and the current LIBERO tail
helper with an explicit 20-step lift to 0.25 m. All 1,513 × 8 reconstructed
float32 action values are bit-identical to published LIBERO baseball episode 0,
including its end flags.

The complete LIBERO setup, generation and export command then passed on the
same GPU server, using Python 3.10.22 and the pinned LIBERO/MuJoCo stack. It
produced 1,513 rows and 957 ink marks. Independent checks validated every row
and fully decoded both 1,513-frame videos. The actual exported action array is
also bit-identical to the published 1,513 × 8 float32 array, including signed
zero, under both bit-pattern and raw-byte comparison. Recorded states differ
on every row; initial conditions, physics and rendering were not recovered
from a historical recipe. The complete parameters and reproduction limits are
in the [reference-input review](PUBLISHED_DATASET_REVIEW.md#original-svg-update-and-action-sequence-correspondence).
