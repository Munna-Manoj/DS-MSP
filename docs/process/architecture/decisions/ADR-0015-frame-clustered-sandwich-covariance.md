# ADR-0015 — Frame-clustered sandwich covariance for rig parameter-uncertainty reporting

- **Status:** Accepted (recorded 2026-07-17; revised 2026-07-18 pre-release — output trimmed to
  the clustered estimator only, after a real-data bootstrap formally measured the coverage of all
  three estimators; see Verification)
- **Deciders:** maintainer
- **Relates to:** ARC-RIG, FR-RIG-020
- **Supersedes:** —

## Context

`ds_msp/core/covariance.py`'s `sandwich_covariance`/`naive_covariance` (M-estimator sandwich:
`Cov = G⁻¹MG⁻ᵀ`, Triggs-curvature-corrected bread `G`, HC1-corrected empirical-score meat `M`)
shipped on main but were dead code — only their own unit test called them, nothing in the rig
pipeline exposed parameter uncertainty to a caller.

A real-data bootstrap validation (frame-resample the real Seltos rig's board views, refit, and
compare empirical parameter std across resamples to each estimator's single-fit predicted std)
found the plain sandwich tracks empirical uncertainty roughly 2.4x better than naive, but **both
still under-cover real uncertainty by ~4-9x**. Root cause: the per-corner "meat" (HC1) treats each
of ~1365 corners as an independent observation, when in reality corners cluster into ~33 board
poses (frames) and share correlated board-pose noise — a single mis-estimated board pose moves
every corner on that board together, which plain HC1 cannot see.

The user explicitly required deriving and closing this gap *before* wiring anything into
user-facing reports, rather than shipping the known-under-covering estimator with a caveat.

## Decision

1. **Cluster the meat by frame, leave the bread unchanged.** Implemented
   `clustered_sandwich_covariance` (`ds_msp/core/covariance.py`) using the standard
   cluster-robust variance estimator (CRVE) form: bread `G` is the same Triggs-curvature-corrected
   expected-Hessian term as the unclustered estimator; the meat becomes
   `M = Σ_g Ψ_g Ψ_g^T`, `Ψ_g = Σ_{i∈g} ψ_i` — summing per-observation scores *within* each cluster
   before outer-producting, rather than outer-producting each observation independently (Liang &
   Zeger 1986, GEE clustering; Cameron & Miller 2015, "A Practitioner's Guide to Cluster-Robust
   Inference," eq. 11). A small-cluster bias correction
   `c = [G/(G-1)] · [(N-1)/(N-K)]` (Cameron, Gelbach & Miller 2008) is applied by default
   (`small_cluster_correction=True`), with `G` = number of clusters, `N` = number of observation
   blocks, `K` = number of free parameters.
2. **Wired into rig reporting as `bundle.parameter_covariance()`**, opt-in via
   `calibrate_scenario(report_covariance=True)` / `calib_param.yml`'s `report_covariance: true`.
   Builds dense `J`/`r` via the existing `build_problem`, builds `cluster_id` by replicating
   `build_problem`'s own observation-filtering logic and assigning each corner block its source
   `object_obs` frame id, auto-estimates the robust scale via `auto_kernel_scale` when not given,
   and reports **only the frame-clustered estimator**, mapped back to per-camera
   `extrinsic_std`/`intrinsic_std` (tangent layout `[δω(3) rad, δt(3) mm-or-scene-units]` —
   rotation first, matching `build_problem`'s retract). The original draft of this ADR reported
   all three estimators (naive, unclustered sandwich, clustered) side by side; that was **revised
   2026-07-18** after a real-data bootstrap (Verification below) measured naive and unclustered
   sandwich to under-cover true uncertainty by ~7x and ~3.6x respectively — a user-facing report
   must not present two formally-proven-overconfident numbers next to the honest one, where a
   downstream consumer could mistake the smallest std for the best. `naive_covariance` and
   `sandwich_covariance` remain in `ds_msp/core/covariance.py` as documented baselines exercised
   by the unit/Monte-Carlo tests, not as user-facing output.
3. **Not yet wired into the HTML report** (`ds_msp/rig/report.py`) — only into the
   `calibrate_scenario()` return dict's `"covariance"` key. Tracked as a fast-follow (see Scope
   deferred); the report-rendering surface was out of scope for "wire it in" as a numerics change.

## Verification

- **Real-data cluster bootstrap — the decisive coverage proof** (2026-07-18, logged in the local
  experiment record "realdata bootstrap covariance coverage"): 200 replicates on the real Seltos
  rig, resampling the 33 board placements (clusters) *with replacement* and refitting each
  replicate (warm-started `bundle.refine`, Cauchy/auto). The empirical std of the fitted
  parameters across replicates is the measured truth (cluster bootstrap, Cameron & Miller 2015
  §VI); coverage ratio = single-fit predicted std / bootstrap-measured std, honest ≈ 1. Median
  coverage over camera 1's six extrinsic components:
  **naive 0.147 — claims ~7x more confidence than reality; unclustered sandwich 0.275 — ~3.6x
  overconfident; frame-clustered 1.136 — honest (slightly conservative).** Measured empirical
  stds for the record: rotation [0.00092, 0.00285, 0.00093] rad, translation
  [2.208, 0.788, 0.645] mm. This is the formal, data-backed basis for the ranking (a smaller
  reported std *claims* more confidence; the bootstrap proves those claims false for naive and
  unclustered) and for trimming the user-facing output to clustered-only (Decision point 2).
- **Synthetic Monte-Carlo, known ground-truth correlation** (`tests/core/test_clustered_covariance.py`,
  `pytest.mark.req("FR-RIG-020")`): manufactured design with an independent per-corner noise
  component plus a shared per-frame offset (the calibration analogue of correlated board-pose
  noise). Over 500 seeded refits, the plain sandwich under-covers the frame-level parameter
  direction (`ratio_sw[3] < 0.4`, i.e. predicted std under 40% of empirical std); the
  frame-clustered sandwich closes this to `0.6 < ratio_clu[3] < 1.5`. The full (unreduced,
  4000-seed) derivation this test regresses at reduced seed count measured **0.161 → 0.979**
  coverage ratio. Also verified: clustering reduces exactly to the plain sandwich when every
  block is its own cluster (`test_clustered_reduces_to_plain_sandwich_when_every_block_is_its_own_cluster`),
  and the estimator stays symmetric PSD (`test_clustered_covariance_symmetric_psd`).
- **Real data, actual wired path**: `calibrate_from_config` against `seltos_cameras_rig/seltos_cams/`
  (`calib_param_gaze.yml`, `overrides={"report_covariance": True}`) — `n_clusters=33` (matching the
  known ~33 board-pose frames), `n_blocks=1365` (matching the known corner count), `K=204` free
  parameters, robust scale `1.142`. Camera 1's clustered extrinsic std, in the documented
  rotation-first tangent layout: **rotation [0.045, 0.224, 0.065] deg, translation
  [3.05, 0.72, 0.77] mm** — the translation stds sit at ~1.1-1.4x the bootstrap-measured
  empirical scatter [2.208, 0.788, 0.645] mm, exactly the honest-slightly-conservative coverage
  the bootstrap measured (1.136 median). Camera 0 is the gauge-fixed reference camera; its
  `extrinsic_std` is correctly `None`, not a free parameter. (An earlier draft of this bullet
  quoted the clustered translation values as "rotation deg" — a presentation-layer mislabel of
  the tangent layout, caught by cross-checking against the bootstrap's per-component empirical
  stds; the layout is now stated explicitly in `parameter_covariance`'s docstring and re-verified
  end-to-end here.)
- **Second real dataset, structurally different rig** (`2026_06_26_MC-Calib/calib_param.yml`: 8
  real fisheye (kb) cameras, 2-board fused object, intrinsics refined from scratch) —
  `n_clusters=58` (frame x fused-object combinations), `n_blocks=11337` corners, `K=454` free
  parameters, robust scale `0.859`, final BA rms 0.61 px, wall time 47.5 s end-to-end with the
  covariance report. Clustered stds in the correct rotation-first layout, e.g. cam1 rotation
  **[0.041, 0.020, 0.024] deg** / translation **[0.22, 0.40, 0.37]** scene units, vs cam4
  rotation **[0.080, 0.126, 0.129] deg** / translation **[10.35, 3.65, 2.23]** — the correction
  and the resulting uncertainty vary strongly by camera (cams 4/5, the far/oblique cameras,
  carry ~20-50x the translation uncertainty of cams 1-3), consistent with the estimator tracking
  each camera's actual frame-correlation exposure rather than applying a fixed multiplier.
  During the pre-trim validation run (when the wired path still computed all three estimators),
  clustered exceeded naive componentwise in **20/21** translation-component comparisons across
  the 7 non-reference cameras — 20/21, not 21/21, is the honest count: on one cam6 component the
  Triggs curvature-corrected bread had moved the sandwich *below* naive before clustering moved
  it back up, landing just short. This ADR's claim is that clustering closes the *measured*
  under-coverage relative to the unclustered sandwich (the defect being fixed), not that
  clustered-vs-naive is monotonic for every camera and axis. (The earlier draft of this bullet
  labeled these translation components "rotation deg" — same presentation-layer mislabel as the
  Seltos bullet, corrected here from a re-run on the trimmed branch.)
- **Governance gates**: `ruff check .`, `lint-imports`, `check_traceability.py --check`.
- **Regression**: full `pytest tests/core tests/rig -m "not realdata"` clean aside from one
  pre-existing seed-borderline statistical flake
  (`test_model_agnostic.py::test_model_agnostic_within_1pct[kb]`), independently reproduced
  byte-identical on a pristine `origin/main` checkout.

## Consequences

**Positive**
- Rig parameter-uncertainty reporting is now backed by an estimator verified (not assumed) to
  close a measured, real, order-of-magnitude under-coverage gap, rather than shipping a
  known-wrong number with a caveat comment.
- The single reported number is the one the real-data bootstrap measured as honest (coverage
  1.136); the two proven-overconfident estimators (0.147, 0.275) are kept only as test baselines,
  never shown to a user who could mistake their smaller stds for higher-quality calibration.

**Negative / costs**
- The clustered estimator still does not claim to fully solve calibration uncertainty reporting:
  it fixes the dominant *measured* correlation source (within-frame board-pose noise) via a
  scalar small-cluster correction, not a fully modeled finite-cluster-count correction; with only
  `G=33` clusters here, some residual under-coverage from finite-`G` bias is plausible and
  unmeasured beyond the scalar correction Cameron/Gelbach/Miller already provide for it.
- Not wired into the human-facing HTML/terminal report yet — only into the programmatic
  `calibrate_scenario()` return value. A caller using the CLI's default rendered report will not
  see these numbers without reading the return dict directly.
- `parameter_covariance` evaluates the M-estimator score/bread at a fixed `kernel="cauchy"`
  (its default), but the BA that actually produced the fitted state may have solved a different
  objective — by default `calibrate_rig` uses GNC-TLS (`noise_bound` is non-`None` by default;
  the `noise_bound<=0` config sentinel that disables it is the FR-RIG-019 fix recorded in
  ADR-0014's "What survives" section), not a Cauchy IRLS solve. The reported covariance is then evaluated at a
  point that is not exactly stationary for the Cauchy score it assumes — a real approximation
  (caught in independent red-team review), not measured to be large on this repo's data (the
  fitted state is a converged least-squares-like minimum regardless of which robust loss reached
  it, so the mismatch is a curvature/weighting approximation, not a location error) but not
  quantified either. Matching `parameter_covariance`'s kernel to whichever loss actually produced
  the fit is deferred (see Scope deferred) rather than guessed at here.

## Scope explicitly deferred (not accidental omissions)

- **Wiring into `ds_msp/rig/report.py`'s rendered HTML/terminal output.** Deliberately deferred:
  the user's "wire them in" scope was about making the numerics reachable and correct, not about
  designing a new report UI section. `bundle.parameter_covariance()`'s docstring flags this
  explicitly as a fast-follow.
- **Finite-cluster-count correction beyond the CGM08 scalar factor** (e.g. wild cluster
  bootstrap) — no real-data motivation yet at `G=33`; the scalar correction already moves the
  synthetic coverage ratio into the acceptance band.
- **Matching `parameter_covariance`'s evaluation kernel to whichever robust loss actually
  produced the fitted state** (GNC-TLS vs. a specific IRLS kernel) — flagged above as a real
  approximation; deferred because it requires `calibrate_rig`/`calibrate_scenario` to report back
  which objective the final BA pass actually used, a small API extension out of scope for this
  ADR's "wire in what already exists" mandate.

## Alternatives considered

- *Ship the unclustered sandwich with a documented under-coverage caveat* — explicitly rejected
  by the user before this work started: a known-wrong uncertainty number is worse than no number
  in a calibration report, where downstream consumers may treat a std as actionable.
- *Cluster by (camera, frame) pair instead of frame alone* — considered; rejected because the
  under-coverage's actual mechanism (shared board-pose noise) is a per-frame phenomenon that
  couples observations *across* cameras co-observing the same frame, so clustering by
  (camera, frame) would under-cluster and leave part of the gap open. Frame-only clustering
  matches the actual noise-generating mechanism.
