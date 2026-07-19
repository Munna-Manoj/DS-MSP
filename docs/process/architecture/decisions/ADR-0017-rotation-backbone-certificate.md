# ADR-0017 — Opt-in global-optimality certificate for the extrinsic-rotation backbone

- **Status:** Accepted (recorded 2026-07-18)
- **Deciders:** maintainer
- **Relates to:** ARC-RIG, FR-RIG-022, ADR-0016 (the other half of the calibration trust layer)
- **Supersedes:** —

## Context

Every stage of the rig pipeline is a local solver from a seed; GNC widens the robust basin
but certifies nothing. A user cannot distinguish "global optimum" from "nice-looking local
minimum" — and no calibration tool anywhere gives them the means to (13-tool gap audit,
2026-07-18: the strongest global-optimality move in any calibration library is robust
*initialization*). Meanwhile the certifiable-estimation literature has a mature,
a-posteriori, cheap answer for the rotation sub-problem: rotation-synchronization strong
duality (Eriksson, Olsson, Kahl, Chin, CVPR 2018; Rosen, Carlone, Bandeira, Leonard,
SE-Sync, IJRR 2019; Dellaert et al., Shonan, ECCV 2020). **Prior art is cited, not
claimed away**: GTSAM ships the identical construction (`ShonanAveraging::checkOptimality`)
for SLAM factor graphs, as do TEASER++/SE-Sync reference implementations. What no
*camera-calibration* library has ever done is surface it as a user-facing calibration-trust
output; the 2025 certifiable-BA literature (arXiv 2502.04640, 2506.23808) assumes known
pinhole intrinsics and does not cover fisheye rig calibration.

## Decision

1. **Certify the rotation backbone as a rotation-synchronization problem over the
   bipartite camera × board-placement graph.** Each observation's robust-PnP pose
   (`ObjectObs.T_c_o`) contributes the measurement `R̃_ct = R_c_g · R_g_o`, weighted by
   corner count (normalized). Node rotations are **frame-from-world** (`Z_c = R_c_g`,
   `Z_t = R_g_oᵀ`), so `R̃_ct = Z_c Z_tᵀ` — the convention the column-stacked connection
   Laplacian encodes. This convention is *load-bearing*: the transposed choice mis-poses
   the problem silently (first wiring measured a certified-but-58°-away optimum at zero
   noise; caught by the mandatory zero-noise sign test, now a regression test).
2. **Refine, then certify.** The BA rotations minimize the reprojection cost, not the
   chordal cost, and the certificate is only meaningful at a first-order critical point
   (a non-stationary point makes `Λ` non-symmetric and `λ_min` spuriously negative). So:
   Riemannian gradient descent on the chordal cost from the BA warm start (`refine_chordal`,
   gradient FD-checked ≤1e-5 in the jac gate, actual measured 2.2e-8), then build
   `Λ_i = Sym((L R)_i R_iᵀ)` and check `S = L − Λ ⪰ 0` by `eigh`, per connected component,
   with the exact 3-dim-per-component gauge nullspace expected at machine zero and a
   scale-relative tolerance `η = λ_min/mean(deg) ≥ −1e-6` (measured: gauge zeros ~1e-15,
   planted wrong basin −0.13 — eight orders of separation).
3. **The BA-vs-chordal distance is a mandatory output with its own verdict, split by node
   type.** Measured: under pure measurement noise, `d(BA, chordal)` tracks the edge
   residual ~1:1 (0.86° at 1° noise, 4.3° at 5°); with a camera's calibrated rotation
   planted 60° wrong, the warm-started refinement escapes to the global basin, certifies
   it, and `d = 57.3°` against 0.82° residuals. Cameras are the calibration output, so
   the verdict uses `d_cam_deg` (`ba_consistent = d_cam ≤ max(3·median residual, 0.5°)`);
   `d_frame_deg` is reported separately — a large `d_frame` with small `d_cam` means a
   board-placement node was dragged by a bad measurement, not a wrong calibration
   (measured on the Seltos rig: one 92° PnP pose produced d_frame 26.8° / d_cam 1.8°,
   which the unsplit distance mis-read as a wrong basin). This turns certification into a
   **positive wrong-basin detection**: "the global optimum was found and certified, and
   the calibrated camera rotations are NOT it." Three user-facing verdicts: CERTIFIED
   (+consistent), WRONG-BASIN WARNING (certified, inconsistent), NOT CERTIFIED
   (inconclusive).
4. **Gross PnP measurement outliers are trimmed and reported, never silently kept or
   dropped.** Both real datasets contained wrong per-view PnP poses that the robust BA
   correctly down-weighted but that would contaminate the certified problem (the
   garbage-in caveat made concrete): Seltos had one 92° flipped pose; MC-Calib had twelve,
   five of them 160–177° near-antipodal fisheye-PnP flips (the ADR-0013 failure mode),
   which stalled the chordal refinement (grad 3.8e-2) and failed the certificate
   marginally (η = −1.29e-6). Two-pass scheme: pass 1 refines and scores every edge by
   the **minimum** of its rotation residual at the BA configuration and at the refined
   solution — an edge is an outlier only if inconsistent with *both*, which keeps a
   wrong-basin camera's edges (consistent with the refined optimum; trimming them would
   eat the evidence) and keeps good edges of a frame node dragged by a bad sibling
   (consistent with the BA solution). Edges beyond `max(6·median, 15°)` are trimmed
   (measured separation: inlier medians 0.38–2.3° vs outliers at 17–177°); pass 2
   certifies the cleaned graph. The outlier list (`outlier_edges`, camera + frame +
   residual) is itself a first-class diagnostic — it names exactly which captures have
   wrong per-view poses.
5. **Honest scope, verbatim in the output** (soundness audit 2026-07-18, worked from the
   fetched papers): a PASS proves global optimality of the *weighted chordal
   rotation-averaging cost on this measurement graph* — one-sided error only, no false
   positives regardless of relaxation tightness. It does **not** certify the PnP inputs
   (mutually-consistent wrong measurements certify cleanly — the trim narrows this to
   *coherently* wrong inputs), translations, intrinsics, or the reprojection-BA optimum
   itself.
6. **Opt-in** (`certify: true` / `--set certify=1`), matching `report_covariance`: it is a
   deeper proof, not a cheap default check (the chordal refinement costs seconds at
   calibration scale). Wired through `calibrate_scenario` → `calib_param.yml` → CLI
   (`certificate:` line in the terminal report).

## Verification

- **Math sign tests** (`tests/core/test_rotsync.py`, FR-RIG-022): noise-free exact solution
  certifies with exactly 3 machine-zero gauge eigenvalues and a clean gap; a planted
  non-global critical point is refused (`η < −1e-6`); moderate (5°) noise still certifies;
  chordal gradient FD-checked (`-m jac`).
- **Pipeline tests** (`tests/rig/test_certify.py`): zero-noise `d = 0.0000°` certified;
  noisy d tracks the residual median; planted 60°-wrong camera → `certified=True,
  ba_consistent=False, d_cam > 30°`, message contains WRONG-BASIN, and the trim does NOT
  eat its (mutually consistent) edges; a planted single 92°-flipped measurement is
  trimmed, named in `outlier_edges` with its camera/frame identity, and the calibration
  still certifies consistent; no-measurements case skips gracefully (`certified=None`).
- **Real-data validation** (2026-07-18, both rigs): Seltos 2-cam — CERTIFIED + consistent,
  `d_cam 0.34°` vs median residual 0.38°, one outlier named (cam0 frame 13); MC-Calib
  8-cam fisheye — CERTIFIED + consistent (η = −1.7e-15), `d_cam 6.2°` vs median residual
  2.34°, twelve outliers named (worst 176.6°). Before the trim, Seltos raised a *false*
  WRONG-BASIN warning and MC-Calib failed certification outright.
- Characterization (2026-07-18): convention bug caught by the zero-noise sign test,
  measured regime separations quoted above, wall time ~2 s at 3 cam × 20 frames.

## Consequences

- DS-MSP becomes the first camera-calibration library whose report can say, with a proof:
  "this extrinsic rotation configuration is the global optimum of its measurement graph" —
  or, stronger, positively flag a wrong-basin BA result that reprojection statistics alone
  can look acceptable on.
- The certificate inherits the robust front-end's weights: it certifies the weighted
  problem the pipeline actually solved. Garbage-in-certified-garbage is documented in the
  output message itself.
- Dense `eigh` on 3n×3n (n = cameras + board placements; ~1500×1500 at heavy captures) —
  trivial at calibration scale, unfit for SLAM-scale graphs (out of scope).
- Translations and intrinsics remain uncertified — future work would need a different
  relaxation (SE(3) sync certifies translations too; fisheye intrinsics are
  non-polynomial and have no known certificate).

## Alternatives considered

- *Full SE-Sync (SE(3)) certificate including translations* — deferred: heavier machinery
  (Riemannian staircase) for a second-order gain; the rotation backbone is where basin
  errors live (hand-eye/init failures are rotational).
- *Certify the BA rotations directly without chordal refinement* — rejected: vacuous
  (non-stationary point), measured to produce spuriously negative eigenvalues.
- *Ship as default-on* — rejected: seconds of cost and a subtle interpretation contract;
  the default-on trust surface is ADR-0016's audit, with the certificate as the deeper
  opt-in proof — mirroring the covariance precedent (cheap honest default, deeper opt-in).
