# ADR-0030 — Seed lenses from printed boards with the radial constraint

- **Status:** Accepted (recorded 2026-11-06)
- **Relates to:** ARC-CALIB, ARC-RIG, FR-CALIB-001, FR-CALIB-010, FR-RIG-001, FR-RIG-003

## Context

The rig pipeline resected boards with a pinhole plus RadTan fit from `cv2.calibrateCamera`, fused
them into a target, then fitted each lens to that target. A pinhole bearing does not exist past 90
degrees, so wide lenses produced a wrong target, the target error entered every lens fit, and wrong
lenses gave wrong extrinsics. Truth measured after each stage on six rendered 25-camera rigs at
a7238e7: target error 19 to 87 mm median, lens error up to 727 px inside the observed cone, extrinsic
seeds up to 49 degrees and 10 m. Bundle adjustment recovered most runs from there, which hid the cause
and made outcomes depend on which basin a run reached: a long chain ended unconverged, and Double
Sphere `xi` took the wrong sign in some cameras.

## Decision

Lens seeding comes first, and uses only geometry known exactly: one printed board at a time.

1. **Poses.** The radial alignment constraint (Tsai 1987; Thirthala and Pollefeys 2005) gives
   `R, t_x, t_y` per view from a linear solve, with no lens knowledge and for any incidence angle.
   Each view starts from a least-median fit over a fixed sample set, because a least-squares start
   cannot survive gross blunders. Samples are seeded by the data size, so results repeat.
2. **Back-projection.** A polynomial shared by the views and each view's `t_z` come from one linear
   system (Scaramuzza et al. 2006). Each planar view keeps the branch this shared lens explains best,
   and corners this step cannot explain are rejected, as a blunder along the radial line passes step 1.
3. **Model.** For fixed shape parameters a camera model is linear in `fx, fy`, so these are eliminated
   and the one- or two-dimensional shape is searched on a grid, with no sign assumption on `xi`.
4. **Selection.** Every distinct basin gets a full refinement with free poses; the lowest truncated
   pixel cost wins, among fits whose axial focal stays within a factor of two of the seed's. Without
   that anchor a refine can absorb blunders into a tiny focal at a low residual.

Boards are then resected with the fitted lens and fused, so the target no longer depends on a pinhole.
The rig median focal check no longer overrides a board-level fit: across mixed lenses that median
reset correct cameras. Existing seed paths remain the fallback when too few views exist or the fit is
inconsistent. Supplied intrinsics still take precedence.

## Consequences

- Seeding is deterministic, as it no longer calls `cv2.calibrateCamera`.
- Seed cost is linear in views. Reconstruction becomes slower for rigs with many boards and faster
  overall, since later stages start close to the solution.
- All views fronto-parallel leave `t_z` and focal unresolved, as in any planar calibration. The fit then
  falls back or is flagged by the existing gates.
- The principal point starts at the image centre; a decentred lens enters only as seed error and is
  refined afterwards.
- A radially asymmetric lens breaks the constraint in proportion to its asymmetry. This is a seed, and
  its accuracy claim rests on the refinement that follows it.

The evaluation is in the pull request and in `tests/geometry/test_radial.py`,
`tests/calib/test_lens_seed.py` and `tests/rig/test_board_level_seed.py`. No threshold here was fitted to
a dataset: the 5 px fit check only selects the fallback path, the 3 px truncation only compares
candidates, and the factor of two matches the existing front-end focal anchor.
