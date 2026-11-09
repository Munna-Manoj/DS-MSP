# ADR-0033 — Check every camera's lens after a rig calibration, and name what the audit finds

- **Status:** Accepted (recorded 2026-11-08)
- **Relates to:** ARC-RIG, FR-RIG-021, FR-RIG-028, FR-RIG-030

## Context

A camera that sees few frames can end with a wrong lens while every stage reports `CONVERGED`. On
synthetic rigs with one camera cut to 1 or 2 frames the lens was wrong by more than 1 px over the
image in 103 of 168 runs. Three things let this through:

- The observability audit's default mode (`audit_gate: warn`) attached its result to the returned
  dict and nothing else: no warning, no line in `calibration_acceptance.json`.
- No check looked at the lens in pixels. The frame-clustered sandwich covariance is estimated from
  the frames a camera sees and is overconfident with one or two of them.
- When the reference camera was the weak one, the weak directions spread over every pose and were
  reported as "a global gauge freedom … should be impossible", which names neither the camera nor the
  cause. A 25-camera corridor chain got the same message for a bend in the middle of the chain.

## Decision

1. **The audit reaches the user.** Under `warn`, findings raise a `RuntimeWarning` and are recorded as
   `observability` in the acceptance file.
2. **Lens uncertainty in pixels** (`lens_gate`). The classical covariance `σ² S⁻¹` of the shared
   parameters (mrcal's `σ² (JᵀJ)⁻¹`, with the Schur bread `S` the uncertainty step already forms) is
   propagated by square root, `σ ‖G_u L‖`, to a 24 x 18 pixel grid. The part of the lens Jacobian a
   camera rotation absorbs is removed first, as mrcal does. The median is widened by
   `t(0.975, G−1)/z(0.975)` for a camera that sees `G` frames and is unbounded at one frame or along a
   null direction of `S`. Only the cameras whose lens touches a null direction are unbounded.
3. **Parameters at an optimizer bound** are listed per camera and fail the same gate.
4. **Split cross-validation** (`crossval_gate`, mrcal), **opt-in**. The frames are cut into 8
   contiguous blocks in capture order and alternate blocks go to each half; neighbouring video frames
   share their errors, so even and odd frames would compare a capture with itself. The rig is re-solved
   on each half, warm-started from the full solution; half the median lens difference over the grid
   cells both halves observed, after the best rotation, estimates the full-data lens error. It assumes
   no model and no linearity. It is off by default and under production:
   - Both halves still come from one capture, so the check misses errors the capture shares (lighting,
     board positions, detector bias). On TUM-VI it reads 0.055 to 0.34 px per recording, while the
     lenses of two recordings differ by 0.27 and 0.35 px (medians).
   - It cannot run when the reference camera sees the board in one stretch only (rendered S2, a room
     split into zones).
   - The lens is validated instead with a second, independent capture.
5. **Naming by variance share.** Each audit finding lists the parameters whose Belsley-Kuh-Welsch
   variance-decomposition proportion in that direction exceeds 0.5 (lower bounds from the computed
   weak directions). A direction spread over the poses is a global gauge only if it is null to
   rounding (|ratio| ≤ 1e-12). Otherwise it is named by its variance share: `reference_camera_weak`
   when the reference camera's lens carries it, `extrinsic_chain_weak` with the cameras whose poses do,
   or `rig_weak`.
6. Both lens checks compare with `max_lens_sigma_px`, by default the configured `noise_bound`. The
   lens σ check warns by default and refuses under `profile: production`; the split check runs only
   when `crossval_gate` is set. Both are skipped when intrinsics are fixed.
7. The per-frame Gram products of the covariance are formed with BLAS instead of three-operand
   `einsum`, which numpy runs as a plain loop.

## Consequences

- Synthetic matrix, 420 runs, 1,960 cameras, 7 lens families from 68 to 249 degrees, ground-truth lens
  error measured after the best rotation:
  - 128 of 139 lenses with a median error above 1 px are flagged.
  - 69 of 1,821 good lenses are flagged. 64 of them are on cameras with one or two frames, where a
    correct lens cannot be told from a wrong one.
  - The 11 misses are 10 reference cameras with 4 to 8 frames (true 1.06 to 1.66 px over the image,
    0.2 to 0.5 px where observed; predicted 0.76 to 0.93 px) and one camera with 3 frames.
  - The split check did not fire on this matrix: it needs three frames per half, and no camera there
    had a model that does not fit.
- Real data, against `main` in the same batch:
  - The calibration is unchanged: extrinsics, intrinsics and held-out error are identical on the 10
    MC-Calib and two-camera RealSense datasets, and the output files are byte-identical on the 12 rendered
    25-camera runs.
  - The lens σ check flags no camera on 7 MC-Calib datasets (34 cameras, widened σ 0.03 to 0.28 px)
    or on the rendered rigs (300 cameras, at most 0.024 px). The two-camera RealSense rig fixes its intrinsics, so
    the lens checks are skipped.
  - The split check, when enabled, estimates 0.08 to 0.57 px on 28 cameras of six MC-Calib datasets
    and at most 0.16 px on the rendered rigs (124 of 150 cameras per repeat; S2 is not applicable).
    On Seq05 it estimates 0.97 to 1.07 px on all six cameras, which have 315 to 451 frames and a
    lens σ of 0.17 to 0.24 px, and three of them cross the 1 px limit. The cause is not established.
    MC-Calib agreement and held-out error on Seq05 are the same as on `main`.
- TUM-VI, 8 calibration recordings at 1024 px with Kalibr's AprilGrid detections: the split estimate
  is 0.055 to 0.34 px per recording (even/odd halves gave 0.001 to 0.014 px), while the lenses fitted
  to two different recordings differ by 0.27 and 0.35 px (medians of the two cameras).
- The split check, when enabled, costs one extra joint BA per half. With the BLAS change, the rendered 25-camera runs
  are 3 to 30% faster than `main` in the same batch.
- At 100 cameras the covariance step was 77% of a 4.4-hour run (12,282 s of 15,905 s); the matrix
  products give the same values to 1e-15 relative and were 60 times faster on a 3,000-point frame.

The evaluation is in the pull request, `tests/rig/test_lens_check.py` and `tests/rig/test_audit.py`.
The widening factor, the median over the grid, the 0.5 variance share and the factor of two in the
split are statistical definitions, not fitted to a dataset. The 1e-12 gauge ratio and the 1e-6 null and
energy shares sit between measured values, which are recorded in the code.
