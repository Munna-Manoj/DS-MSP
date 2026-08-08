# ADR-0020 — Fully bearing-native pose refinement with a chordal residual

- **Status:** Accepted (recorded 2026-07-08)
- **Deciders:** maintainer
- **Relates to:** ARC-RIG, ARC-GEOMETRY, FR-CALIB-002, FR-OPS-003
- **Supersedes:** ADR-0018 §3 and ADR-0019 §3 (normalized-plane-only IRLS refinement)
- **Subsequent decision:** ADR-0021 replaces RANSAC in the recommended and rig robust-PnP paths;
  the explicitly named compatibility API remains available.

## Context

ADR-0018 and ADR-0019 made the non-coplanar and coplanar PnP seeds full-sphere, but
`robust_pose_irls` still refined on the normalized plane. It divided by predicted `z`, kept
only `z > 0`, and therefore could not use a valid fisheye bearing at or past 90° off-axis.
A zero-forward view returned the RANSAC seed unchanged; a mixed view optimized only its
forward subset.

One candidate was a two-component tangent-plane residual,
`[e_u·d, e_v·d]`, where `d` is the predicted unit bearing and `(e_u, e_v)` spans the plane
orthogonal to the observation `f`. That residual is finite on the full sphere, but its norm is
`|sin(theta)|`: it is zero for both `d = f` and the antipode `d = -f`. Treating a 180° error as
a perfect fit is an unacceptable ambiguity, and the same formula was duplicated in rig and
two-view bundle adjustment.

## Decision

1. **Use one shared chordal bearing primitive.**
   `ds_msp.geometry.bearing.chordal_bearing_residual_jacobian` computes

   `r = normalize(y) - normalize(f)`

   and `dr/dy = (I - d dᵀ) / ||y||`. Invalid zero-length rows return zero residual/Jacobian
   with `valid=False`. The implementation is NumPy-only and lives in the neutral geometry
   layer so pose initialization, rig BA, and multi-view BA use the same primitive.

2. **Use the chordal residual in `robust_pose_irls`.** For camera-frame point
   `Pc = R X + t`, the pixel-equivalent residual is `e = focal · (d - f)`. Its cost is

   `||e||² = 2 focal² (1 - cos(theta))`.

   It is zero only when predicted and observed rays agree, and increases monotonically to its
   maximum at 180°. The analytic Jacobian preserves this module's existing left-composed SE(3)
   convention:

   `d(e)/d(delta) = focal · (I - d dᵀ)/||Pc|| · [I | -hat(Pc)]`.

   Each observation is now one 3-component robust block. Studentization, kernel weighting,
   and GNC remain per correspondence; all valid rays participate regardless of `z` sign.

3. **Remove the same antipodal ambiguity from the other bearing optimizers.**
   `ds_msp.rig.bundle` angular mode and `ds_msp.mvg.bundle.refine_two_view` now consume the
   shared chordal primitive and use 3-component robust blocks. The duplicated
   `_tangent_basis` helpers are removed.

4. **Keep the warm-start safety net.** At the time of this decision, the returned pose was
   whichever of the RANSAC seed and refined candidate had lower full-data bearing cost. This
   cheaply prevents a bad
   Gauss–Newton step from making the data fit worse.

## Verification

- `tests/geometry/test_bearing.py` proves the antipode has residual norm 2 rather than zero
  and that squared chordal cost is monotone from 0° through 180°.
- `tests/rig/test_pose_irls_bearing_jacobian.py` finite-difference-checks the left-SE(3)
  analytic Jacobian on forward and full-sphere scenes and has a direct antipode regression.
- `tests/rig/test_bundle_jacobian.py` continues to finite-difference-check rig angular BA;
  `tests/mvg/test_bundle.py` continues to exercise two-view refinement and large rotations.
- `tests/rig/test_pose_irls_bearing_native.py` uses two distinct fixtures:
  - a mixed full-sphere Double Sphere scene whose 15 seeds contain 20.7–36.0% peripheral rays
    (median 29.5%), with 2 px noise. Median rotation error improves from 0.162° to 0.044°
    (72.6%); median translation error improves from 3.78 mm to 2.01 mm (46.7%);
  - a literal zero-forward scene (`ray_z <= 0` for every retained point), which proves the
    refinement changes and improves the seed instead of passing it through.
- Existing forward-only, coplanar, non-coplanar, outlier, reconstruction, and calibration
  suites remain the regression authority. Repository-wide counts are intentionally not frozen
  in this ADR because unrelated tests make aggregate totals stale.

## Consequences

**Positive**

- PnP is now bearing-native from seed through refinement for both target geometries.
- No antipodal observation can register as a zero-residual match.
- One geometry primitive replaces three copies of the same bearing residual construction.
- Near a correct solution, chordal and angular distance have the same first-order scale, while
  chordal cost stays well-defined and monotone over the complete sphere.

**Negative / costs**

- A unit direction has two local degrees of freedom, but the chordal residual stores three
  components. The third component is redundant to first order near the solution; the small
  extra work buys a basis-free residual and globally unambiguous cost.
- The exact antipode is still a stationary *maximum* of any smooth rotationally symmetric
  sphere cost. Its residual and cost are now maximal, not falsely zero, but a pose initialized
  exactly there still needs a non-antipodal seed. The bearing RANSAC warm start supplied that
  seed when this ADR was recorded; ADR-0021 now supplies deterministic GNC-TLS in the recommended
  and rig paths.
- Two pose perturbation conventions remain: left composition in `pose_init.py`, right
  composition in the bundle modules. Each Jacobian is derived and finite-difference-tested
  against its own retraction.

## Alternatives considered

- **Keep the tangent-plane residual and rely on the warm start.** Rejected: a residual that is
  exactly zero for the wrong ray violates the estimator contract even if current seeds usually
  avoid it.
- **Use the geodesic logarithm on the sphere.** Rejected for this path: it needs an arbitrary
  branch at the antipode and more delicate derivatives, while chordal cost provides the needed
  monotonicity with a simple analytic Jacobian.
- **Move pose refinement to the generic LM solver.** Deferred: the bespoke loop has
  studentized-leverage weighting that the generic solver does not currently expose.
- **Remove the warm-start comparison.** Rejected: it is inexpensive and protects against a
  pathological update without weakening the optimized residual.
