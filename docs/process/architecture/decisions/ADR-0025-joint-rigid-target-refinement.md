# ADR-0025 — Joint rigid multi-board target refinement by default

- **Status:** Accepted (recorded 2026-10-22)
- **Deciders:** maintainer
- **Relates to:** ARC-RIG, FR-RIG-026, FR-RIG-025
- **Supersedes:** the `refine_structure: false` default of commit 9cd1356

## Context

`build_objects` fuses a multi-board target from averaged single-view PnP. No bundle-adjustment
stage refines the resulting inter-board poses, so their error stays frozen into
`Object3D.pts_3d` unless a target-refinement stage corrects it. Until 9cd1356 that stage was
MC-Calib's per-point `refineObject` (on whenever `number_board > 1`). 9cd1356 made it opt-in
because free per-point motion can absorb detector bias. Its validation measured within-board
distance change, which cannot see inter-board pose error.

Measured on the MC-Calib real sequences (held-out frames, shared pose seeds, intrinsics fixed at
the published values) and on the Seltos two-camera capture with a double-sided board (3-fold frame
hold-out; dataset-gated, not distributed):

| held-out RMS px | off (9cd1356) | free | rigid (joint) | bend (joint) |
|---|---|---|---|---|
| Seq00 | 0.902 | 0.681 | 0.627 | 0.612 |
| Seq01 | 0.364 | 0.338 | 0.344 | 0.335 |
| Seq04 | 1.543 | 0.710 (48 px with PnP seeding) | 0.438 | 0.415 |
| Seq05 | 0.364 | 0.351 | 0.351 | 0.351 |
| Seltos 2-camera (mean of 3 folds) | 1.067 | 0.823 | 0.927 | 0.889 |

Single-board sequences are unchanged in every mode. Alternating structure steps with the bundle
adjustment (MC-Calib's scheme, and a first rigid prototype) zig-zags and stalls: on a synthetic
1-degree / 14 mm inter-board error it stopped 5.5 mm short after eight rounds. On the real
sequences, free mode's *training* RMS is above rigid's despite far more freedom.

## Decision

1. `structure_mode` selects target refinement: `rigid` (default), `bend`, `free`, `off`.
   The legacy `refine_structure: true` means `free`.
2. `rigid` adds each non-reference board's 6-DOF pose about its centroid to the **shared block
   of the joint Schur bundle adjustment** (`bundle.refine(refine_boards=True)`), with an analytic
   Jacobian `J_point · R_c R_o · [-[q]_× | I]`. The reference board fixes the gauge. Printed board
   shape is held, so per-corner detector bias cannot be absorbed. `T_co_b` is updated with the
   same motion.
3. `bend` adds three mean-centred quadratic out-of-plane bow coefficients per board
   (`u², v², uv`), for physically warped targets. It is not the default: it adds parameters that
   the MC-Calib targets do not need, and the gain there is 2–5% held-out.
4. `free` keeps per-point refinement and its prior. The prior is now weighted at the estimated
   corner noise (`σ_px²/σ_target²`), so `structure_prior_sigma` is a physical length.

## Consequences

- Multi-board extrinsics recover the inter-board pose instead of inheriting PnP-averaging error
  (Seq04 rotation difference to the published fit 0.49° → 0.14°).
- On a target that is not rigid (the Seltos double-sided board), `rigid` leaves residual
  deformation. `bend` recovers part of it; measuring the board decides between `bend` and
  `free`.
- One extra 6-column block per board in the shared system; runtime within the alternation's.
- `bend` is experimental. In the full pipeline from raw images (intrinsics estimated), its
  median held-out error is lower than `rigid` on Seq00 and Seq04, but robust pose fitting
  against the bowed target fails on 10–18 held-out frames and diverges on a few more (held-out
  RMS 1.1 px and 4.3 px against 0.56 px and 0.41 px). Pose initialization for non-planar
  bowed boards is the follow-up.

## Verification

- `tests/rig/test_rigid_target_refinement.py`: finite-difference check of the board-pose and bow
  Jacobian columns; recovery of a perturbed board with exact shape; bend recovers a 12 mm bow
  that rigid cannot.
- `tests/rig/test_image_truth_benchmark.py`: rendered-image ground truth; rigid beats a frozen
  inter-board pose.
