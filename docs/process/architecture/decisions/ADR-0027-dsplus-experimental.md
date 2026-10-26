# ADR-0027 — DS⁺ is an experimental model, not for production

- **Status:** Accepted (recorded 2026-10-26)
- **Deciders:** maintainer
- **Relates to:** ARC-MODELS, ARC-RIG, FR-MODEL-001..005, ADR-0005, ADR-0023
- **Supersedes:** the "best accuracy" recommendation for DS⁺ in the README and guides

## Context

ADR-0005 added DS⁺ (`dsplus`) for its single-camera fit: 0.224 px vs KB 0.285 px on a demanding
158° lens. The README then recommended it, including for rigs. Rig runs that use it are far from
the MC-Calib reference, while their reprojection error matches the other models.

DS⁺ is `pixel = K · H_tau · D_lambda · S_alpha(bearing)`. The tilt stage `H_tau` divides by
`w = 1 + tau_x·x + tau_y·y`.

## The degeneracy

With a perspective core (`alpha = 0`, `lambda1 = lambda2 = 0`), rotating the camera by θ about
x gives exactly the same pixels as the unrotated camera with

    tau_y = tan θ,  fx' = fx / cos θ,  fy' = fy / cos² θ,  cy' = cy − fy · tan θ

and the same holds about y with `tau_x` and `cx`. So the intrinsics can absorb a camera rotation
about x or y. `tests/models/test_dsplus_model.py` checks this identity to 1e-9 px.

Away from the pinhole limit the identity is close, not exact. On a synthetic near-pinhole lens,
DS⁺ leaves 0.24% / 0.75% / 1.6% of a rotation's pixel signature unexplained at 20° / 35° / 50°
half-FOV. KB, DS, UCM and EUCM leave 3.3% / 10.9% / 24%. On a 190° fisheye the difference
disappears.

In single-camera calibration this is harmless: the board poses take up the rotation. In a rig it
is not: the rotation between cameras is what we want to measure.

## Evidence on real data

MC-Calib Real_Images, 7 sequences × 20 model assignments (140 rig runs), compared with MC-Calib's
own extrinsics. "Worst pair" is the largest rotation difference over all camera pairs of a run.

| runs | n | worst-pair rotation, median / max | max centre difference | held-out RMS median |
|---|---|---|---|---|
| KB / RadTan / DS / UCM / EUCM only | 57 | 0.16° / 2.0° | 18 mm | 0.39 px |
| with OCam, no DS⁺ | 33 | 0.26° / 11.0° | 706 mm | 0.33 px |
| with DS⁺ | 50 | 4.59° / 29.8° | 612 mm | 0.40 px |

The DS⁺ error follows the degeneracy. Across 280 runs (two solver versions), the 106 DS⁺
cameras with a tilt error above 2° have `tau_y ≈ −tan(error)`: correlation −0.986, slope −0.991.
The largest errors stop near 26.6° = atan(0.5), the `tau` bound.

DS⁺ also drops Double Sphere's `xi`. It fits poorly on lenses that need that term.

## Decision

1. DS⁺ stays in the library as an **experimental** model. Its uses are studying camera-model
   behaviour and serving as a worked example for writing a new model.
2. The README, the model docstring and every guide that offers `dsplus` say that it is not for
   production, and point here. Production recommendations are KB or DS for fisheye lenses and
   RadTan for pinhole lenses.
3. No code path refuses DS⁺, so existing configs keep running.

## Consequences

- Users who chose DS⁺ for its lowest reprojection error now see why that is not enough.
- A low reprojection error is no longer read as a reason to pick a model for a rig.
- Removing DS⁺, or replacing its tilt with a constrained term, needs its own ADR and a
  deprecation release.
