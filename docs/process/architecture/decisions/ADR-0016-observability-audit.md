# ADR-0016 — Default-on observability audit with named weak directions

- **Status:** Accepted (recorded 2026-07-18)
- **Deciders:** maintainer
- **Relates to:** ARC-RIG, FR-RIG-021
- **Supersedes:** —

## Context

The rig solver silently absorbs unobservability: a near-singular normal matrix is rescued by
escalating Cholesky jitter (`ds_msp/core/optimize.py`, `_solve_damped`) and the pipeline
returns a confident-looking calibration with no warning that some parameter combination was
never constrained by the capture. This is the *silent-confident-wrong* failure class — a
sub-pixel reprojection RMS can coexist with a focal/distortion combination that is free to
drift. A 13-tool audit (Kalibr, MC-Calib, Basalt, OpenCV, camodocal, multical, MATLAB,
COLMAP, BabelCalib, AprilCal, Calibration Wizard, GeoCalib, AnyCalib) found **no calibration
library fires an observability/degeneracy diagnostic before returning a wrong fit** —
Kalibr/Basalt/MC-Calib log nullspaces for internal debugging only; AprilCal (IROS 2013) and
Calibration Wizard (ICCV 2019) did the covariance-guidance math as standalone capture tools.

## Decision

1. **Equilibrate before eigen-analysis — mandatory, not cosmetic.** The BA Jacobian's
   columns mix units (px/rad, px/length, px/px, px/1), so raw eigenvalues of `H = JᵀWJ` and
   raw eigenvector components are unit artefacts (measured: healthy 3-camera rig
   `cond(H) = 1.7e11` vs `cond(Ĥ) = 2.1e6`). Van der Sluis diagonal equilibration
   (`Ĥ = D H D`, `D = diag(1/√diag H)`, Numer. Math. 14, 1969 — the same column scaling
   Ceres applies as `jacobi_scaling`) makes `Ĥ` the weighted-Jacobian correlation matrix:
   dimensionless, ≈1-referenced eigenvalues; comparable eigenvector energies. Naming is
   proven **unit-invariant only under equilibration** (m→mm regression test).
   Core math in `ds_msp/core/observability.py` (NumPy-only, ADR-0004 respected); rig
   semantics in `ds_msp/rig/audit.py`.
2. **Two-tier thresholds locked from measured spectra, not theory.** Characterization
   (experiment log 2026-07-18) measured structural degeneracies (gauge modes, the planar
   focal↔ξ coupling) at equilibrated-eigenvalue ratios ≤1e-10 and the softest *healthy*
   directions (e.g. RadTan k2/k3 near-collinearity) at ≥1e-5: a ~6-order empty gap.
   Critical `tau_rel=1e-6` (named findings, gate-relevant) and soft `1e-3` (counted,
   summarized, never alarmed) sit inside it. A single-tier 1e-3 design was measured to
   over-fire on healthy captures (5 findings incl. a false gauge label) and was rejected —
   alarm fatigue would kill the feature's trust value.
3. **Findings are named and actionable**, matched to the known fisheye degeneracy
   signatures with capture advice: focal↔distortion planar coupling (Usenko 3DV 2018;
   Hartley–Zisserman planar gauge) → "add tilted views"; outer-FOV parameters with an empty
   periphery annulus → "capture near the image edges"; weak frames (merged into ranges);
   weak camera extrinsics (with co-observation counts); degenerate capture motion
   (OpenVINS-style discipline); global gauge — which is a *bug detector*: the shipped
   layout pins the datum by construction, and the positive control (deliberately unfixing
   the reference camera) fires **exactly 6** gauge findings. Hessian-free coverage stats
   (equal-area radial occupancy; orientation-tensor tilt diversity) corroborate and
   disambiguate the advice.
4. **Default-on (`audit_gate: warn`), `refuse` and `off` opt-in.** Cost is one dense
   `eigh` of a K×K matrix (K ≈ 200–500) — well under one BA iteration, which factors
   same-size systems repeatedly. The value (catching a capture the user did not suspect)
   evaporates if opt-in; `refuse` (raise instead of returning a silently under-constrained
   calibration) matches `reproj_gate_px`'s opt-in hard-gate precedent. Surfaced in the
   terminal report (`render_audit`: one quiet line when clean) and the
   `calibrate_scenario()` return dict.

## Verification

- Well-conditioned 3D-target rig: silent (`n_weak=0`, no findings), `cond(Ĥ) < 1e8`, and
  `cond(H) > 100·cond(Ĥ)` (the units artefact is real).
- Gauge positive control: unfixing the reference camera fires exactly 6 `global_gauge`
  findings at ratios <1e-10; the shipped layout stays `gauge_ok`.
- No-tilt planar DS capture: names `fx,fy↔xi` at ratio 1.8e-13 with tilt advice.
- Centered-board capture: periphery flagged, outer-annulus occupancy 0%.
- Unit-invariance: naming survives a m→mm re-expression only under equilibration; the
  raw-H naming flips (regression-tested).
- Core linear algebra: planted exact-duplicate column found with >0.95 energy on the pair;
  scale-invariance of `equilibrate` to per-column unit changes; coverage formulas on
  synthetic annuli/normal sets. All `@pytest.mark.req("FR-RIG-021")`.

## Consequences

- DS-MSP becomes the first calibration library whose *default* output states whether the
  capture actually constrains the parameters it reports — with named directions and
  concrete capture advice, not a bare condition number.
- The audit reads the same `build_problem` Jacobian the BA uses (no second derivation to
  drift); `fix_intrinsics` audits only the pose/extrinsic state.
- One dense eigh per calibration of added cost (measured trivial at K≈200–500; the K³
  growth means a future 100-camera rig should switch to the Schur-reduced spectrum — noted
  as deferred scope).
- Soft directions are deliberately under-reported (count only) — a conscious trade against
  alarm fatigue; the full spectrum remains available programmatically.

## Alternatives considered

- *Raw-Hessian condition number* — rejected: measured to be a unit artefact (1.7e11 on a
  healthy rig); any threshold on it is meaningless across datasets.
- *Single-tier threshold at 1e-3* — rejected by measurement (over-fires; false gauge label
  on a healthy capture).
- *Opt-in audit* — rejected: the user who most needs it would not enable it; the
  studentize post-mortem (ADR-0014) showed internal machinery without user-visible truth
  value is bloat, and the audit's truth value is precisely its default-on surfacing.
- *Next-best-view suggestion engine* (AprilCal/Calibration Wizard style) — deferred, not
  rejected: the audit's findings already name what coverage is missing; an active guidance
  loop is a capture-time feature, out of scope for the calibration report.
