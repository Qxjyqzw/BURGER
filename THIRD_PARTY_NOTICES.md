# Third-party notices

The root MIT license is retained from the source repository. It does not
relicense external simulator software, datasets, or bundled assets.

## ManiSkill

[ManiSkill 3.0.1's published license statement](https://pypi.org/project/mani_skill/3.0.1/)
distinguishes permissively licensed environment code from assets licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The full texts are retained under `LICENSES/`.

The Isaac `assets/maniskill/` directory contains copied robot descriptions,
meshes, table geometry, and textures. Shared pipeline and LIBERO texture
directories contain copies or derived textures from those assets.
These assets must not be treated as MIT or Apache-only. Previous asset notices
that contained only Apache-2.0 have been corrected.

The exact original per-mesh sources, upstream asset archive revisions, and any
additional attribution obligations need owner confirmation before publication.
The copied source notice
does not establish provenance that is absent from the development branches.

## Simulator and export dependencies

Setup downloads ManiSkill, SAPIEN, MuJoCo, robosuite, LIBERO, Isaac Sim,
Isaac Lab, LeRobot, and their dependencies from their upstream distributors.
Their own licenses and notices continue to apply. Isaac Sim has its own license
acceptance process; setup and verification do not accept it for you.
LeRobot is used as a dependency; legacy copied training/deployment code is
excluded from this release.

The SVG fixtures under `examples/` were created for this release and are covered
by the root MIT license. They are infrastructure test inputs, not the paper's
benchmark dataset.
