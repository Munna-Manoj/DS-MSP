# ADR-0014 — Studentized (bounded-influence) IRLS in the sparse Schur solver: REJECTED

- **Status:** Rejected (recorded 2026-07-18; supersedes the 2026-07-17 Accepted draft of this ADR,
  which was never released)
- **Deciders:** maintainer
- **Relates to:** ARC-RIG, FR-RIG-019 (which now covers only the `noise_bound` configurability fix
  that survives from this work)
- **Supersedes:** —

## Context

The 2026-07-16 solver-hardening update added studentized (bounded-influence, Mallows-type) IRLS —
`studentized_scale_factors` in `ds_msp/core/robust.py` — to the dense `lm_solve` path only. The
rig's actual bundle-adjustment solver is the sparse Schur-complement solver `schur_lm`
(`ds_msp/core/optimize.py`), so the feature was unreachable from the rig CLI/config. A full sparse
derivation and wiring was built and validated (exact closed-form hat diagonal from the arrowhead
Schur marginals, per Triggs et al. 2000 §6; FD parity vs a dense reference to 1.3e-15;
`studentize` threaded `schur_lm` → `bundle.refine` → `calibrate_rig` → `calib_param.yml`).

Before merging, the maintainer required a formal keep-or-remove audit: prove whether the feature
adds measurable value on real data, or is bloat.

## Decision

**Do not ship studentized IRLS in the sparse rig path.** The full sparse implementation
(`_hat_arrow`, `_deflation_blocks_schur`, `studentized_scale_factors_schur`, the `studentize`
parameter on `schur_lm`/`refine()`/`calibrate_scenario`/`RigConfig`, and its dedicated test file)
is removed from the branch. The dense-path implementation on `main` (`lm_solve`,
`robust_pose_irls`/`pose_init.py`) is untouched — explicit maintainer decision 2026-07-18.

**What survives from this work** (re-scoped under FR-RIG-019): the `noise_bound` config fix in
`ds_msp/rig/calib_param.py` — `noise_bound` now honors `--set`/overrides like every other field,
and a value `<= 0` is the config-layer sentinel for `None` (GNC-TLS disabled). Before this fix
there was no way to express "plain robust IRLS, no GNC-TLS" from a config file at all — a real,
independently useful configurability gap, kept with its three regression tests.

## Why rejected — the formal trichotomy (measured, real Seltos rig data)

The maintainer's challenge: "either the algorithm can help but you implemented it incorrectly, or
we are using it correctly [and it should help]." Resolved by a
formal trichotomy experiment (2026-07-18), whose load-bearing numbers are quoted below:

1. **Implemented incorrectly? No — proven correct.** The hat matrix of an undamped,
   user-weight-only LS problem is an orthogonal projector, so `Σ_i tr(H_ii) = K_free` is an exact
   structural identity. Measured through the wired sparse path on the real 1365-corner problem:
   **Σtr(H_ii) = 203.998 vs K_free = 204**, on top of the FD parity vs dense (max abs err
   1.3e-15). Real leverage is also genuinely non-uniform (eigmax H_ii: median 0.078, p95 0.240,
   max 0.747) — the no-op is not explained by "no leverage present."

2. **Correctly used and should have helped? No — benefit absent even in the forced worst case.**
   Thinning one both-camera frame to 5+4 corners with one corner corrupted by 40 px drove the
   in-situ leverage to **h = 0.747–0.967** (textbook masking territory, Hoaglin & Welsch 1978).
   Over 20 trials against the same-thinning clean reference: plain Cauchy median error
   **0.004 mm / 0.0002°** vs studentized Cauchy **0.010 mm / 0.0013°** — plain wins **19/20**.
   Mechanism: classical masking assumes the fit is computed *with* the outlier at full weight so
   `Var(r_i)=σ²(1−h_i)` shrinks its residual. Redescending IRLS from a **robust init**
   (front-end RANSAC-PnP, warm starts) sees the outlier's raw ~40 px residual against a ~1 px MAD
   scale at iteration 1 → Cauchy weight ≈ 1e-3 immediately → it never acquires influence. Masking
   is an *init-basin* phenomenon, and this pipeline structurally never provides a contaminated
   init. Meanwhile studentization inflates the residuals of *clean* high-leverage corners
   (deflation factor up to ~20x at h=0.95) into the down-weight zone — the known efficiency cost
   of bounded-influence/GM estimators (Hampel et al. 1986; Krasker & Welsch 1982) — hence
   studentized is consistently slightly *worse*, never better.

3. **Costs are real.** 3.5x wall time on the 8-camera MC-Calib rig (51.0 s vs 14.5 s — the
   earlier "no measurable slowdown" claim from the 2-camera rig did not hold at scale), a second
   undamped arrow assembly per iteration in the hot path, and a two-knob activation
   (`studentize: true` **and** `noise_bound: 0`) that invites misconfiguration.

Predicted no-op magnitude on the unmodified real datasets matches: ≤1.6e-2 mm (Seltos),
≤3.8e-3 mm (MC-Calib) extrinsic deltas — the *expected* outcome of a *correct* implementation in
a regime its assumptions exclude.

## Consequences

- The rig hot path stays free of ~480 lines of formally-proven-valueless machinery; wall time at
  8 cameras stays at ~14.5 s instead of 51 s when the flag would have been enabled.
- The `noise_bound` configurability fix ships independently (FR-RIG-019, three tests).
- If a future pipeline *does* feed the sparse solver a non-robust init (e.g. a linear-only
  front-end with no RANSAC gate), this analysis does not apply and the decision should be
  revisited — the sparse derivation and its verification record are preserved in this ADR and the
  experiment log for that eventuality.

## Alternatives considered

- *Ship it anyway, default-off* — rejected: default-off code that is formally shown to make
  results slightly worse whenever enabled on this pipeline's regime is pure bloat plus a
  misconfiguration hazard, failing the maintainer's explicit "no functions with no practical
  implication" criterion.
- *Keep only `schur_lm(studentize=)` without rig wiring* — rejected: that is exactly the
  dead-code state this work set out to eliminate.
