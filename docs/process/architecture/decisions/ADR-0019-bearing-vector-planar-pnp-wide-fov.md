# ADR-0019 — Bearing-vector homography for coplanar (planar-board) PnP beyond 90° off-axis

- **Status:** Accepted (recorded 2026-07-08)
- **Deciders:** maintainer
- **Relates to:** ARC-OPS, ARC-GEOMETRY, ARC-RIG, FR-OPS-003, FR-CALIB-002
- **Supersedes:** — (closes the coplanar gap ADR-0018 explicitly deferred)
- **Subsequent decision:** ADR-0020 closes the refinement gap.

## Context

ADR-0018 fixed non-coplanar wide-FOV PnP (a bearing-vector DLT, valid past 90° off-axis) but
explicitly left the **coplanar** path — `_pose_planar_normalized`, a plane homography on the
normalized `z = 1` plane — forward-hemisphere (`z > 0`) only, because the general 3×4 DLT is
degenerate for coplanar points and the homography solve needs its own derivation. A single
planar calibration board (the common real case — one ChArUco/checkerboard) tilted or imaged
close enough that corners exceed 90° off-axis therefore silently lost every peripheral corner,
or failed outright if too few forward corners remained. This bound `ops.solve_pnp`,
`ops.solve_pnp_ransac`, `rig.pose_init.estimate_pose_ransac`, `rig.pose_init.robust_pose_irls`
(its warm-start), and — critically — `rig.reconstruct.reconstruct_object`'s per-board
resection (`init_models`, MC-Calib's `cam_params_path` init), whose whole stated purpose is
correctness on wide-FOV lenses.

## Decision

1. **A bearing-vector-native homography solver**, `_pose_planar_bearing(X, rays)`
   (`ds_msp/geometry/resection.py`), generalizes `_pose_planar_normalized` exactly as ADR-0018's
   `_pose_dlt_bearing` generalized `_pose_dlt_normalized`. A coplanar point is
   `P = c0 + a·e1 + b·e2` (plane basis from PCA); under `(R, t)` its camera-frame point is
   `Xc = H·[a,b,1]`, `H = [R e1 | R e2 | R c0 + t]` (3×3) — exact for any `z` sign. Since the
   bearing `f ∥ Xc`, the cross-product constraint `f × (H·[a,b,1]) = 0` holds directly (2
   independent rows/point, same count as the legacy homography DLT). `H`'s column decomposition
   into `(R, t)` is unchanged (depends only on `H`'s structure); sign/cheirality use the same
   `λ = f·(RX+t) > 0` convention as `_pose_dlt_bearing`, not `z > 0`.
   Citations: Zhang, *A Flexible New Technique for Camera Calibration*, TPAMI 2000 (the
   column-recovery step, unchanged); the bearing-vector generalization follows the same
   cross-product principle as Hartley & Zisserman §7.1 / Kneip & Furgale OpenGV ICRA 2014 §III
   already applied to the non-coplanar case in ADR-0018.
   Proven (not assumed): reduces exactly to `_pose_planar_normalized` when every `f_i ∥ (u_i,
   v_i, 1)` with `z > 0` (<1e-9 agreement); zero-noise recovery to machine precision on a board
   tilted ~70° with corners past 90° off-axis (`max|R-R_true|` <1e-8, over 48 points, several
   past 90°, max ~121°); graceful degradation under 0.002 rad bearing noise (<0.5° rotation,
   <0.02 translation error).
2. **A latent sign bug found and fixed in both the new and the pre-existing homography solver**:
   the plane basis's third (normal) vector was taken directly from `np.linalg.svd`'s smallest
   singular vector (`Vt[2]`), whose sign is arbitrary — it can be antiparallel to
   `cross(e1, e2)`, silently turning `[e1, e2, Vt[2]]` from a rotation into a reflection and the
   recovered pose into a mirrored, wrong answer. Confirmed via a manufactured-recovery test at a
   ~70° board tilt (`max|R-R_true|` was 1.9, not <1e-8, isolated to exactly the entries touching
   the normal axis). Fixed in both `_pose_planar_normalized` and `_pose_planar_bearing` by
   computing the normal as `cross(e1, e2)` explicitly rather than trusting `Vt[2]`'s sign. Near
   fronto-parallel views (small `R`) `Vt[2]` and `cross(e1,e2)` happen to agree, which is why
   this had not surfaced in the existing (narrow-FOV, near-fronto-parallel) test suite — it is a
   real, previously-latent bug in the *existing*, already-shipped `_pose_planar_normalized`, not
   something newly introduced by this change, and it specifically manifests at the large board
   tilts this ADR's fix now makes representable in the first place.
3. **Both PnP entry points route coplanar targets to the bearing homography** when `rays` is
   supplied: `ransac_pnp_normalized` (`ds_msp/geometry/resection.py`, new
   `_ransac_pnp_planar_bearing` minimal-sample-4 RANSAC wrapper), `solve_pnp`,
   `solve_pnp_ransac` (`ds_msp/ops/pose.py`), and `estimate_pose_ransac`
   (`ds_msp/rig/pose_init.py`) — the same dispatcher and call sites ADR-0018 already wired for
   the non-coplanar case, now handling both branches uniformly. `robust_pose_irls`'s warm-start
   inherited the fix transitively (unchanged code, delegates to `estimate_pose_ransac`). At
   acceptance its IRLS refine remained normalized-plane-only; ADR-0020 subsequently replaces
   that residual with a full-sphere chordal one.
4. **`rig.reconstruct.reconstruct_object`'s `init_models` path benefits directly** — per-board
   resection there is `robust_pose_irls`, so a genuinely wide-FOV camera (not
   `DoubleSphereModel.sample()`'s narrower default; a `xi=0.3, alpha=0.6` ~180°-class instance)
   resecting a close, steeply tilted board with corners past 90° off-axis now recovers correct
   fused geometry — verified end-to-end
   (`tests/rig/test_reconstruct.py::test_reconstruct_with_model_aware_init_models_wide_fov`,
   median geometry error <3mm, max <1cm, on a scene with 80+ genuinely peripheral corners across
   40 frames).

## Consequences

**Positive**
- The audited gap (coplanar wide-FOV PnP unsupported) is closed for all four PnP entry points
  plus `reconstruct_object`'s board resection — verified via manufactured-solution tests
  (`tests/calib/test_pose_planar_bearing.py`) and end-to-end regression tests
  (`tests/ops/test_pose_wide_fov_coplanar.py`, `tests/rig/test_reconstruct.py`) reproducing a
  board tilted enough to put corners past 90° off-axis: all four PnP entry points and the
  fused-object reconstruction recover ground truth to near machine precision on clean data.
- A genuine, previously-latent sign bug in the already-shipped `_pose_planar_normalized` is
  fixed as a byproduct — any existing caller resecting a steeply-tilted board (even one that
  never puts corners past 90°) was at risk of a silently mirrored pose; this is now closed for
  every caller, not just the new wide-FOV path.
- Zero regression: coplanar `z > 0` data reduces to the prior method's exact answer (<1e-9,
  post sign-fix). Aggregate repository counts are intentionally not frozen here.

**Negative / costs**
- At acceptance, `robust_pose_irls` still refined only on the normalized plane. ADR-0020
  subsequently closes that gap for both target geometries.
- The near-degeneracy guard gap flagged in ADR-0018 (no warning when a target is *near*-planar
  but routed to the general DLT) is unchanged by this ADR; still a fast-follow, not blocking.

## Scope explicitly deferred (not accidental omissions)

- **Fully bearing-native IRLS refinement** for coplanar targets — deferred here and completed
  by ADR-0020.
- **Near-degeneracy SVD guard** — pre-existing gap, unchanged by this ADR.

## Alternatives considered

- *A bearing-vector-native P3P (Grunert/Fischler-Bolles/Kneip quartic minimal solver)*, which is
  coplanarity-agnostic by construction and could replace both the coplanar and non-coplanar
  minimal solvers with one algorithm. Rejected for this change — a real quartic root-finding
  implementation is a larger, riskier derivation than generalizing the existing, already-proven
  homography-DLT pattern the same way ADR-0018 generalized the general DLT; "delete before
  optimizing" favors the smaller, directly-analogous fix. Left as a future consolidation option
  if the two-solver split becomes a maintenance burden.
- *Leaving the coplanar case unsupported and only documenting the limitation.* Rejected — a
  single planar board is the common real calibration scenario, not an edge case; ADR-0018 itself
  flagged this as the highest-priority remaining gap.
