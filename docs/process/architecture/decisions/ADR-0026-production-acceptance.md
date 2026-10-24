# ADR-0026 — Production acceptance: uncertainty gate, strict configuration, rejection ledger

- **Status:** Accepted (recorded 2026-10-22)
- **Deciders:** maintainer
- **Relates to:** ARC-RIG, FR-RIG-027, FR-RIG-028, FR-RIG-020, FR-RIG-021, ADR-0023
- **Supersedes:** dense-Jacobian evaluation in `parameter_covariance`

## Context

Training reprojection RMS says nothing about how well a capture constrains the extrinsics. In
a rendered 18-frame converging capture, freeing the intrinsics left the extrinsics 0.23° /
1.7 mm off while training RMS stayed at the noise floor. The frame-clustered sandwich
covariance (FR-RIG-020) measured honest coverage on real data, but it was opt-in, reported only
per-parameter std, and built a dense Jacobian over every frame pose, which is too large for real
rigs. The configuration silently ignored unknown keys and several MC-Calib keys, including
`boards_index`: an MC-Calib config with physical board ids got the wrong marker offsets, so
five of the seven MC-Calib real sequences found no boards from raw images. `--set` overrides
reached only some fields; `--set cam_prefix=...` or `--set distortion_model=1` was silently
dropped.

## Decision

1. `bundle.shared_covariance` computes the same estimator exactly through the Schur structure:
   `Cov_ss = c · S⁻¹ (Σ_g v_g v_gᵀ) S⁻¹`, with frame poses and, for `rigid`/`bend`, board
   poses marginalised. It inverts with Jacobi scaling and no jitter: a 1e-12 trace-relative
   Tikhonov term moved ill-conditioned intrinsic σ by 0.6%. `parameter_covariance` delegates
   to it. The bread's Triggs curvature is clamped at zero per residual
   (`ω′ ≥ −ω/(2s)`, Triggs et al. 2000 §4.3): a redescending kernel otherwise gives negative
   curvature for residuals beyond its scale, and frames dominated by gross outliers made the
   bread indefinite (75 of 1207 frame blocks on MC-Calib Seq03), which overflowed the Schur
   inverse and crashed the run. A Schur complement that is still non-finite or indefinite,
   or singular in a camera direction, fails closed: the pair σ is unbounded, recorded as
   `null` with `identifiable: false`, a warning is raised, and no limit can pass. One
   singularity is expected and does not fail the rest. Boards of an object that appear in
   the same frame are linked; a linked group without the reference board can move rigidly
   against it, and the frame poses follow. Those 6 board-pose directions are found from
   the board graph, not from an eigenvalue cutoff, and projected out before the inverse.
   They are reported unbounded (`unidentified_structure`). Before this, 7 of 56 two-camera
   fits on MC-Calib Seq01/Seq04 reported no σ at all. A cutoff cannot separate them:
   absorbed directions keep ≤ 6e-13 of their information there, while EUCM's
   near-redundant intrinsics keep 2e-10 and are still identified.
2. Every calibration propagates it to each camera pair's 1σ relative rotation (deg) and camera
   centre (scene units) and records them in `calibration_acceptance.json`.
   `uncertainty_gate: refuse` blocks export above `max_rotation_sigma_deg` / `max_centre_sigma`.
   With `G` frame clusters the σ is estimated from ~`G` effective samples, so the gate compares
   `σ · t_{0.975,G−1}/z_{0.975}` (`*_sigma_t`; ×1.15 at 10 frames, ×1.04 at 30). A 200-capture
   Monte Carlo per condition measured 2σ coverage 0.913 without it at 10 frames (Student-t
   predicts 0.923) and 0.94–0.97 with it across all positive conditions.
3. `profile: production` defaults `audit_gate` and `uncertainty_gate` to `refuse` (limits
   required) and turns on hard rejection of board observations above `ransac_threshold`; every
   dropped observation is written to the acceptance ledger with its reason. Explicit config
   values still win.
4. Unknown keys, in the file or in overrides, warn (production refuses them). Ignored MC-Calib
   keys warn with what DS-MSP does instead. `boards_index` sets per-board marker-id offsets.
   An override replaces the file value for every field, including board geometry and
   sequences.

## Consequences

- The σ is honest for random error: per-axis z-scores ≈ 1 over repeated synthetic captures, and
  the weak free-intrinsic capture reports a 12× larger σ.
- It does **not** include systematic detector bias. On the rendered fisheye rig, the
  ChArUco-under-distortion bias gives z ≈ 3–4. Production limits therefore need a separate bias
  budget, from the image-truth benchmark or a physical check (ADR-0023 accuracy profile).
- Configurations that relied on silently ignored keys now see warnings.

## Verification

- `tests/rig/test_shared_covariance.py`: equality with the (well-conditioned) dense estimator,
  honesty against truth, marginalisation never shrinks σ, an absorbed board leaves the
  cameras bounded, an ill-conditioned but identified `S` stays bounded.
- `tests/rig/test_production_profile.py`: profile defaults, strict keys, `boards_index`
  detection, refusal before export, rejection ledger.
