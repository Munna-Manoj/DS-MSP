# ADR-0021 — Deterministic bearing-space GNC-TLS PnP with guarded compatibility refinement

- **Status:** Accepted (recorded 2026-08-08)
- **Deciders:** maintainer
- **Relates to:** ARC-OPS, ARC-GEOMETRY, ARC-RIG, FR-OPS-003, FR-CALIB-002, FR-CORE-001
- **Extends:** ADR-0020 (the shared full-sphere chordal pose objective)
- **Compatibility:** `solve_pnp_ransac` remains public; this decision changes the recommended and
  rig-calibration robust front end, not the availability of classic RANSAC

## Context

ADR-0018 and ADR-0019 made the non-coplanar and coplanar linear PnP seeds operate on unit
bearings across the complete model-valid sphere. ADR-0020 then made pose refinement use the same
full-sphere chordal geometry. Two robustness gaps remained:

1. Rig calibration still selected its initial pose with random minimal-set RANSAC. Its answer was
   reproducible only because a seed was fixed, and robustness depended on drawing an uncontaminated
   four- or six-point subset.
2. On the adequately sized bearing path, public `solve_pnp_ransac(refine=True)` accepted the flag
   but returned the consensus-refitted linear hypothesis without performing the requested nonlinear
   polish.

The recommended robust estimator needs to be camera-model-neutral after unprojection, deterministic,
usable for coplanar and non-coplanar targets, explicit about its assumed feature-noise bound, and safe
for valid peripheral bearings whose `z` coordinate is zero or negative. Existing callers that
specifically selected RANSAC still need compatible random-minimal-set behavior.

The chordal objective also has a direct directional-statistics interpretation. For observed and
predicted unit directions `f,d` on S²,

`||d - f||² = 2(1 - fᵀd)`.

At a fixed concentration `kappa`, an isotropic von Mises-Fisher observation model has negative log
likelihood `constant - kappa fᵀd`; the unweighted base residual is therefore an affine rescaling of
squared chordal cost for inliers. A single scalar concentration does not, however, turn an angular
bound into the same pixel bound at every location of a wide-FOV image. The public pixel-bound path
uses a fixed ray-varying anisotropic metric derived from the camera contract. It is not a single
fixed-concentration isotropic vMF likelihood. A plain vMF maximum-likelihood fit is also not a
gross-outlier estimator; TLS truncation and graduated weights supply that separate robustness.

## Decision

1. **Centralize the camera-neutral bearing pose operations.** Geometry owns the chordal residual and
   analytic left-SE(3) Jacobian, rescoring, manifold-LM consensus polish, and its acceptance guard.
   Public operations unproject pixels once and evaluate the camera contract's analytic
   `project_jacobian` once at each observed ray to precompute a fixed local metric. The iterative
   optimizer contains no concrete camera equations, does not reproject its state, and never divides
   by bearing `z`.

2. **Make deterministic GNC-TLS the recommended robust PnP API.**
   `solve_pnp_robust(model, object_points, image_points, *, noise_bound_px=3.0,
   max_iters=100, refine=True)` starts from an all-correspondence bearing DLT for a non-coplanar
   target or an all-correspondence bearing homography for a coplanar target. It then alternates
   weighted chordal pose updates with closed-form GNC-TLS weights against the caller's explicit
   locally pixel-calibrated noise bound. For observed unit ray `f_i` and point-projection Jacobian
   `J_i`, the fixed whitener satisfies
   `W_iᵀW_i = J_iᵀJ_i + s_rad² f_i f_iᵀ`. The first term matches local pixel error on the tangent
   plane; the radial completion prevents the antipodal signed ray from entering the projective
   Jacobian's null space. Final weights produce a hard inlier mask. Four valid bearings are required
   for a coplanar target and six for a non-coplanar target.

3. **Guard the optional consensus polish.** The first candidate is fit on the selected consensus.
   One deterministic least-trimmed update then keeps the same consensus cardinality while allowing a
   threshold-edge observation to be exchanged for a lower-residual one; this avoids freezing an
   imperfect discrete label set. The candidate is then rescored on every valid bearing. It is
   accepted only when inlier support does not decrease and the all-data truncated residual
   (MSAC-style) score does not increase. Failure, a non-finite candidate, or a harmful candidate
   leaves the supported robust pose unchanged.

4. **Use the deterministic estimator in rig calibration.** Per-view rig pose gating calls
   `solve_pnp_robust`; `robust_pose_irls` also uses the same GNC-TLS estimator when it needs a warm
   start. A strict GNC consensus is refined on that clean subset, and when every corner is accepted
   the final IRLS keeps every corner. A non-coplanar 4--5 point view cannot determine the six-point
   bearing DLT; if those rays fit a numerically usable open hemisphere, deterministic SQPnP in a
   ray-aligned virtual perspective chart supplies only the missing pose seed, after which
   refinement returns to the bearing metric. A failed robust consensus or unsupported small
   full-sphere geometry fails
   explicitly rather than optimizing from identity. The later joint rig bundle adjustment keeps its
   existing robust treatment.

   The earlier model-aware intrinsic bootstrap is a distinct use of GNC: pixels are temporarily
   interpreted through a provisional pinhole before the unknown fisheye model exists. Its internal
   TLS bound is 5 px, not the public PnP default of 3 px, because that bound must include provisional
   projection-family mismatch as well as feature noise. The value remains well separated from the
   front-end's manufactured 40 px gross-blunder validation.

5. **Retain RANSAC as an explicit compatibility API.** `solve_pnp_ransac` continues seeded random
   minimal-set sampling and preserves its return shape and controls. Its adequately sized bearing
   paths now honor `refine=True` with the same guarded full-sphere polish. Its undersized fallback
   remains the normalized-plane/OpenCV path over usable forward correspondences.

6. **Expose the new operation consistently.** `solve_pnp_robust` is exported from `ds_msp.ops` and
   the package root for every modern `CameraModel`; the legacy `DoubleSphereCamera` convenience
   facade exposes the corresponding method and derivative contract too. This is a backward-compatible
   feature addition; no existing public entry point is removed.

7. **Do not claim certification.** This implementation is deterministic and non-minimal, but the
   weighted pose subproblem is solved by local manifold LM from a deterministic linear seed. It does
   not inherit the no-initial-guess or global/certifiable guarantees available to GNC constructions
   whose variable update is globally solved.

## Verification

- `tests/ops/test_pose_gnc.py` contaminates non-coplanar, coplanar, and entirely negative-forward
  Double Sphere scenes with 70%, 60%, and 60% planted pixel outliers. It verifies accurate recovery,
  rejects every planted outlier in those fixtures, and proves bit-for-bit repeatability. Separate
  regressions verify that a 4 px bound keeps 100/100 clean 1 px-noisy DS rays at 95--110 degrees,
  while the six-model DS/UCM/EUCM/KB/OCam/DS+ sweep keeps 120/120 rays at 92--100 degrees. The
  legacy camera wrapper remains operational.
- `tests/ops/test_pose_bearing_refine.py` proves that the RANSAC `refine` flag improves an ordinary
  noisy forward consensus, actually uses an all-negative-`z` consensus, and rejects candidate
  polishes that reduce support or worsen the truncated score.
- `tests/ops/test_pnp_wide_fov.py`, `tests/ops/test_pose_wide_fov.py`, and
  `tests/ops/test_pose_wide_fov_coplanar.py` remain the full-sphere regression authority for both
  target geometries.
- `tests/geometry/test_bearing.py` and the ADR-0020 Jacobian/refinement suites continue to verify the
  shared chordal primitive and its antipodal behavior.
- `tests/core/test_gnc_tls_final_weights.py` proves returned TLS labels are recomputed against the
  returned state rather than left stale from the preceding variable update.
- `tests/rig/test_pose_irls_bearing_native.py` exercises deterministic 5-point non-coplanar seeding
  both in the ordinary forward chart and with every physical-camera ray past 90 degrees.

These fixtures demonstrate the stated scenarios; they do not establish a universal numerical
breakdown point for every scene geometry or outlier distribution.

## Consequences

**Positive**

- The recommended and rig-calibration robust pose front ends no longer depend on a lucky random
  minimal subset.
- One explicit `noise_bound_px`, interpreted through fixed model-derived local metrics, connects the
  caller's measurement model to GNC-TLS instead of deriving scale from a majority-contaminated
  sample median or assuming one focal scale is valid across the image.
- Robust seeding, scoring, and polishing preserve every model-valid direction over S².
- Existing RANSAC callers retain their selected algorithm, while `refine=True` now fulfills its
  documented contract on bearing-capable inputs.

**Negative / costs**

- GNC-TLS performs repeated weighted nonlinear pose solves and can cost more than an easy RANSAC
  case.
- The result still depends on scene observability, the deterministic linear seed, and a physically
  meaningful noise bound.
- Maintaining both the recommended deterministic API and the compatibility RANSAC API increases the
  public surface and requires both paths to remain tested.

## Alternatives considered

- **Keep RANSAC as the recommended and rig default.** Rejected for those roles: random minimal-set
  discovery is avoidable now that a deterministic all-data robust path exists. RANSAC remains
  available for callers that explicitly want its behavior.
- **Use plain fixed-concentration vMF maximum likelihood.** Rejected as the robust front end: it is
  equivalent to untruncated chordal least squares for the isotropic base model, does not encode the
  ray-varying pixel covariance, and does not reject gross mismatches.
- **Fit a vMF/uniform mixture with EM.** Deferred. It is a plausible probabilistic alternative but
  introduces mixture priors, concentration estimation, and local EM assignments without replacing
  the need for careful initialization.
- **Claim the global guarantees of certifiable GNC pose estimators.** Rejected: the current local LM
  inner step does not satisfy the premise required for that claim.

## References

- R. A. Fisher, “Dispersion on a Sphere,” *Proceedings of the Royal Society A*, 1953,
  <https://doi.org/10.1098/rspa.1953.0064>.
- Y. Guan and W. A. P. Smith, “Structure-from-Motion in Spherical Video Using the von
  Mises-Fisher Distribution,” *IEEE Transactions on Image Processing*, 2017,
  <https://doi.org/10.1109/TIP.2016.2621662>.
- H. Yang, P. Antonante, V. Tzoumas, and L. Carlone, “Graduated Non-Convexity for Robust Spatial
  Perception: From Non-Minimal Solvers to Global Outlier Rejection,” *IEEE Robotics and Automation
  Letters*, 2020, <https://doi.org/10.1109/LRA.2020.2965893>.
