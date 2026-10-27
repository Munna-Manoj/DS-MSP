# ADR-0028 — OCam does not determine camera roll in a rig

- **Status:** Accepted (recorded 2026-10-27)
- **Deciders:** maintainer
- **Relates to:** ARC-MODELS, ARC-RIG, FR-MODEL-001, ADR-0026, ADR-0027

## Context

OCamCalib (`ocam`) maps a ray to a pixel through a polynomial `w(rho)` and an affine stretch:

    [u; v] = A · rho · (X, Y) / sqrt(X² + Y²) + [cx; cy],   A = [[c, d], [e, 1]]

`rho` depends only on the angle from the optical axis. Rig runs with an OCam camera never report
an extrinsic uncertainty, so we checked whether the model defines the camera orientation.

## The degeneracy

`A` has three free entries and a fixed `1`, so it can represent any 2×2 matrix up to scale.
Roll the camera by θ about its optical axis. This rotates `(X, Y)` by `Rz(θ)` and leaves `rho`
unchanged. The unrolled camera gives the same pixels with

    k = cos θ − e · sin θ,   A' = A · Rz(θ) / k,   a_i' = a_i · k^(1−i)

The polynomial change scales the root `rho` by `k`, and dividing `A` by `k` undoes that scale.
The identity is exact for every θ with `k > 0`, not only in a limit.
`tests/models/test_ocam_model.py` checks it at 1°, 10° and 30° to 1e-9 px (measured 1e-13 px).

So the intrinsics `(c, d, e, a0..a4)` can absorb a camera roll. In single-camera calibration this
does no harm: the board poses absorb the roll. In a rig, roll between cameras is part of
the extrinsics we want, and the pixels cannot fix it.

Unlike DS⁺ (ADR-0027), this freedom has exactly zero gradient. The solver never moves along it,
so the roll stays near its initial value instead of drifting to a bound.

## Evidence on real data

MC-Calib Real_Images, Seq00–Seq05, 120 rig runs with the `main` solver before #87 and 20 model
assignments per sequence. Errors are relative to camera 0, compared with MC-Calib, and split into
camera-frame axes. To keep other models' errors out, the table uses only runs with no DS⁺ camera
whose camera 0 is not OCam.

| model | cameras | roll, median / p95 | tilt, median / p95 | total, median / p95 |
|---|---|---|---|---|
| OCam | 22 | 0.099° / 0.19° | 0.12° / 0.35° | 0.17° / 0.43° |
| KB | 49 | 0.024° / 0.18° | 0.14° / 0.41° | 0.15° / 0.41° |
| DS | 42 | 0.036° / 0.13° | 0.14° / 0.40° | 0.15° / 0.41° |
| UCM | 53 | 0.028° / 0.11° | 0.14° / 0.54° | 0.15° / 0.54° |
| EUCM | 42 | 0.029° / 0.11° | 0.14° / 0.56° | 0.15° / 0.56° |
| RadTan | 48 | 0.025° / 0.14° | 0.05° / 0.25° | 0.06° / 0.25° |

OCam's overall accuracy on these runs is in the same range as KB, DS, UCM and EUCM. Its total
error is close to theirs, and its tilt p95 is the lowest of the five. Only RadTan is
clearly better.

The one consistent difference is roll. Comparing the same camera of the same sequence, OCam's
roll error is larger than DS, UCM, EUCM and RadTan by a median 0.06–0.09°, and the 95% intervals
(bootstrap over sequences) exclude zero. Against KB the difference is +0.067°, but the interval
includes zero. Tilt shows no consistent difference, except that RadTan is better.

This offset is small because, in these runs, the solver leaves roll where it started. The problem
is not size: no amount of data corrects a wrong starting roll, and the uncertainty is undefined.
The uncertainty gate (ADR-0026) shows this. All 27 runs with an OCam camera and no DS⁺ report no
rotation σ, because the information matrix is singular. Of the 47 runs with neither model, 9
report no σ.

## Decision

1. OCam stays a supported model. It is fine for single-camera calibration, conversion and
   undistortion.
2. The README, the model docstring and the rig guides say that OCam does not determine camera
   roll in a rig, and point here. For rig extrinsics use KB or DS for fisheye lenses and RadTan
   for pinhole lenses.
3. No code path refuses OCam in a rig.

## Consequences

- Users see why an OCam rig fails the uncertainty gate even when the reprojection error is low.
- A fix is available if needed: holding `d = 0` removes the freedom exactly, since
  `tan θ = d / c` rotates any `A` to that form. It changes the OCam parameter set, so it needs
  its own ADR.
