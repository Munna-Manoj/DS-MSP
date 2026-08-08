# ADR-0018 — Bearing-vector DLT for non-coplanar PnP beyond 90° off-axis

- **Status:** Accepted (recorded 2026-07-07)
- **Deciders:** maintainer
- **Relates to:** ARC-OPS, ARC-GEOMETRY, ARC-RIG, FR-OPS-003, FR-CALIB-002
- **Supersedes:** —
- **Subsequent decisions:** ADR-0019 closes the coplanar gap; ADR-0020 closes the
  refinement gap; ADR-0021 replaces RANSAC in the recommended and rig robust-PnP paths while
  retaining the explicitly named compatibility API.

## Context

`ds_msp/ops/pose.py` (`solve_pnp`, `solve_pnp_ransac`) and `ds_msp/rig/pose_init.py`
(`estimate_pose_ransac`, `robust_pose_irls`) unprojected pixels to bearing rays, then filtered
`rays[:, 2] > 1e-6` and solved PnP on the normalized `z = 1` plane via `cv2.solvePnP` / the
non-coplanar DLT in `ds_msp/geometry/resection.py::_pose_dlt_normalized`. This is not a tunable
threshold — dividing by `z` to reach the normalized plane is mathematically undefined for
`z ≤ 0`, i.e. any bearing at or past 90° off the optical axis.

DS-MSP's wide-FOV models (Double Sphere, EUCM, Kannala–Brandt, OCam) are documented and
validated well past 90° (DS to 195° per Usenko et al. 2018). A reproducible synthetic case
(bearings 95–112° off-axis, `model.project()`/`unproject()` both certifying 100% validity) drove
`solve_pnp`, `solve_pnp_ransac`, `estimate_pose_ransac`, and `robust_pose_irls` to `ok=False`/
`None` on every call — total pose failure on data the camera model itself certifies as good. The
module docstrings claimed "any fisheye/omni model" without the forward-hemisphere caveat that
only `solve_pnp_ransac` stated correctly. `ds_msp/mvg/two_view.py` already documents and uses the
correct wide-FOV convention for two-view geometry ("front" = positive depth along the bearing
vector, `λ > 0`, not `z > 0`) — proof the codebase already knew the right principle; it was
simply never applied in the PnP/resection layer.

## Decision

1. **A bearing-vector-native linear DLT**, `_pose_dlt_bearing(X, rays)`
   (`ds_msp/geometry/resection.py`), generalizes the existing `_pose_dlt_normalized`. For a
   bearing `f_i` observing world point `X_i` under pose `(R, t)`, `f_i ∥ (R X_i + t)` exactly,
   giving the linear constraint `f_i × (R X_i + t) = 0` (2 independent rows per point, same
   count as the classic point-DLT). Stacked into `A·vec(P) = 0`, solved by SVD null-space,
   `R` recovered by projecting the 3×3 block onto SO(3), sign/scale fixed by cheirality
   `λ = f·(RX+t) > 0` — matching `two_view.py`'s convention, not `z > 0`.
   Citations: Hartley & Zisserman, *Multiple View Geometry*, 2nd ed., §7.1 (general
   resectioning DLT, cross-product form); Kneip & Furgale, *OpenGV*, ICRA 2014, §III
   (generalized-camera absolute-pose DLT on bearing vectors).
   Proven (not assumed): reduces exactly to `_pose_dlt_normalized` when every `f_i = (u_i,
   v_i, 1)` (<1e-9 agreement, independently re-verified); zero-noise full-sphere manufactured
   recovery to machine precision (`max|R−R_true|` 3.3e-16, `max|t−t_true|` 2.8e-16 over 60
   points, 23 past 90°, max 176°).
2. **Non-coplanar PnP entry points route to the bearing DLT** when the target is non-coplanar
   (`_is_coplanar` check, reused from the existing branch already present in
   `ransac_pnp_normalized`) — `solve_pnp`, `solve_pnp_ransac`
   (`ds_msp/ops/pose.py`), `estimate_pose_ransac` (`ds_msp/rig/pose_init.py`), and
   `ransac_pnp_normalized` (new `rays=` kwarg, `ds_msp/geometry/resection.py`). RANSAC inlier
   scoring for these paths uses the angular bearing residual `acos(clip(f_pred·f_obs))`,
   gated at `thresh_px / focal` radians — verified to reduce to the pixel-space gate near the
   axis and to tighten (not loosen) at the periphery for DS's own sample parameters (3.75 px
   equivalent at axis → 1.68 px at 110°), so it cannot be more permissive to bad matches than
   the legacy gate was.
3. **At the time of this decision, `robust_pose_irls`'s IRLS refine stayed normalized-plane
   (`z > 0`) only** — it was a genuine
   scope boundary, not silently dropped: refining against a *minority* forward-point subset can
   pull the pose away from the (unrefined but full-data) bearing warm start, so the function
   now returns whichever of {warm-start, refined} scores lower on the **full** bearing-angle
   residual (all usable points, any `z` sign), never just the `z > 0` subset the refine itself
   optimized. Verified over 10 random seeds with a 30-peripheral/6-forward split: the function
   never regresses the warm-start pose (either keeps it or improves on it) — see
   `ds_msp/rig/pose_init.py::robust_pose_irls` docstring for the exact mechanism.
   ADR-0020 subsequently replaces this refine residual with a full-sphere chordal one.
4. **The coplanar / homography path (`_pose_planar_normalized`) was untouched by this ADR**
   and remained `z > 0`-only. ADR-0019 subsequently adds the bearing homography.
5. **Docstrings were corrected to match this ADR's then-current coverage.** ADR-0019 and
   ADR-0020 update them again as the two explicit gaps close.

## Consequences

**Positive**
- The audited failure mode (100% PnP failure on `project()`-valid wide-FOV data) is fixed for
  all four affected entry points, for the non-coplanar case — verified via a fails-before/
  passes-after regression test (`tests/ops/test_pose_wide_fov.py`) reproducing the exact
  95–112°-off-axis scenario: `ok=False` → `ok=True`, 0.0° rotation error, ~9e-16 translation
  error, 40/40 inliers; a 25%-injected-outlier variant holds pose (<0.5°/<0.05) and correctly
  flags every injected outlier.
- Zero regression: non-coplanar `z > 0` data reduces to the prior method's exact answer
  (<1e-9); coplanar and legacy call sites (`ds_msp.rig.calibrate`, `test_robust_init.py`) are
  bit-identical (untouched code path, `rays=None` default).
- Targeted PnP, Jacobian, contract, and broader regression suites were green after the
  change. Aggregate repository counts are not frozen here because they go stale as unrelated
  tests are added.

**Negative / costs**
- At acceptance, coplanar targets past 90° and bearing-native IRLS refinement were still
  unsupported. ADR-0019 and ADR-0020 subsequently close those gaps.
- Both the legacy and new DLT lack a
  near-degeneracy guard: a near-planar-but-technically-non-coplanar target (e.g. a slightly
  warped board) can route to the general DLT and produce a poor pose with no warning. Flagged
  separately because the legacy path was equally exposed.

## Scope explicitly deferred (not accidental omissions)

- **Coplanar/homography wide-FOV PnP** — deferred here and completed by ADR-0019.
- **Fully bearing-native IRLS refinement** — deferred here and completed by ADR-0020.
- **Near-degeneracy SVD guard** on `_pose_dlt_bearing`/`_pose_dlt_normalized` (return `None`
  rather than a silently poor pose near-planar) — pre-existing gap in the legacy method too,
  not introduced by this change.

## Alternatives considered

- *A generic focal-scale/threshold change to the existing pinhole PnP.* Rejected — provably
  impossible: the `z = 1` plane cannot represent `z ≤ 0` bearings at all, at any threshold.
- *A learned or iterative-only (no closed-form init) wide-FOV pose estimator.* Rejected as
  first cut — a closed-form linear DLT generalizing the codebase's own existing pattern
  (`dlt_projection`/`_pose_dlt_normalized`) is simpler, matches "delete before optimizing," and
  needed no new dependency or new analytic Jacobian.
- *Rewriting `robust_pose_irls`'s refine residual to be bearing-native immediately.* Deferred
  here to keep the seed-stage change reviewable; completed by ADR-0020 with a separately
  derived and finite-difference-tested Jacobian.
