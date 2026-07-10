# ADR-0013 — Robust reporting for gross-outlier board detections, with an opt-in hard-drop gate

- **Status:** Accepted (recorded 2026-07-11)
- **Deciders:** maintainer
- **Relates to:** ARC-RIG, FR-RIG-018
- **Supersedes:** —

## Context

A ChArUco board that some OpenCV builds mis-decode (correct board, wrong corner ids) resects to
a plausible-but-wrong pose with a **low** per-board reprojection RMS, so no resection-level check
catches it. It only reveals itself once composed with the assembled rig (`T_c_g @ T_g_o`), where
it reprojects at 100-300px. The robust joint BA (GNC-TLS/Cauchy) already down-weights this kind
of blunder correctly in the *estimate* — `T_c_g`/intrinsics/median error stay correct — but the
observation remains in `object_obs`, so every metric computed over *all* corners (max, rms, the
BA's own readout) still counts it: a correctly-calibrated rig then displays as a ~25px failure,
and the live 3D view's per-camera depth (driven by mean reprojection error) sinks the affected
camera to the pond floor, so only the "good" camera appears to have calibrated at all.

Confirmed on the real non-overlapping Seltos rig (`seltos_cameras_rig/seltos_cams/`, the same
dataset ADR-0011 validates against) under an OpenCV build that mis-decodes 3 of 35 frames: max
per-camera error 294.6px, BA rms 30.6px, despite the recovered extrinsic being correct (1.193m /
178.6deg, matching ADR-0011's numbers to 3 places). MC-Calib's own config exposes exactly this
class of problem: `ransac_threshold` ("keep it high, remove only strong outliers") — parsed into
`CalibConfig` and documented in both shipped YAML templates, but never actually applied anywhere
in the pipeline before this ADR.

Two candidate fixes were prototyped in sequence, and the second correctly subsumes the first:

1. **Hard-drop.** After the robust BA, drop any observation whose rig-composed reprojection
   exceeds `ransac_threshold`/`reproj_gate_px`, then re-solve on the cleaned set. This fixes the
   metrics (the blunder is gone) and matches MC-Calib's literal semantics. Verified on the real
   mis-decoding build: max error 294.6px -> 2.61px, BA rms 30.6px -> 0.73px, extrinsic unchanged.
2. **Down-weight + report the robust picture, don't drop.** Directly comparing the two on the
   same real data: dropping the blunder vs. leaving it down-weighted gives **identical
   intrinsics** and an extrinsic within **0.1deg / 3.5mm** — the estimate was never the problem.
   What was wrong is that the reporting layer (terminal table, live-view depth) did not
   distinguish "down-weighted blunder" from "the fit is bad," so it presented a correct
   calibration as if it had failed. Fixing the *reporting* is a smaller, more honest
   intervention than silently discarding data the estimator already handled correctly, and it
   generalizes to any gross-outlier cause, not just this OpenCV-build-specific one.

## Decision

Ship both, with (2) as the default and (1) as an explicit opt-in — never dropping data the
caller didn't ask to have dropped:

1. **Robust reporting is always on** (`ds_msp/rig/report.py`). `ErrorStats` gains `inlier_rms`
   (RMS over corners below a gross-outlier line, default `GROSS_PX = 5.0`) and `n_gross` (count
   at/above it). `render_report`'s per-camera table adds an `inl_rms` column and, only when
   `n_gross > 0` for some camera, an explanatory note pointing the reader at `median`/`inl_rms`
   instead of `max`/`rms`. Nothing is hidden — the raw max/rms/columns stay exactly as before —
   the robust numbers are added alongside them, not in place of them.
2. **Robust live-view depth is always on** (`ds_msp/rig/web3d.py`). `_camera_frame_errors` uses
   the **median** (not mean) per-camera reprojection error, and the browser-side depth mapping
   caps the error that drives it (`DEPTH_ERR_CAP`) so one blown-up camera cannot compress every
   other camera against the pond floor. Belt-and-suspenders: the Python-side median already
   fixes the common case; the JS-side cap guards the pathological all-corners-bad frame.
3. **The hard-drop gate is opt-in, off by default.** `calibrate_rig(reproj_gate_px=...)` and
   `calibrate_scenario(reproj_gate_px=...)` thread an optional gate through to a new
   `_reject_outlier_observations`/`_observation_reproj_rms` pair in `calibrate.py`: after the
   robust BA, drop any observation whose rig-composed reprojection exceeds `gate_px`, mutate the
   caller's `object_obs` list in place (so downstream metrics/saved output see the cleaned set),
   and re-solve. **Not auto-wired from `calibrate_from_config`** — the config-driven CLI entry
   point relies on (1)+(2) by default; a caller wanting MC-Calib's literal `ransac_threshold`
   hard-drop semantics passes `reproj_gate_px=cfg.ransac_threshold` explicitly. The shipped YAML
   templates' `ransac_threshold` field is documented as parsed-but-not-auto-applied (see Scope
   deferred) rather than silently doing nothing with no explanation, which was the pre-ADR state.

## Verification

- **Real data, reproduced independently of the authoring session** (this ADR's own review, cv2
  5.0.0 — a clean build): re-ran `calibrate_from_config` against `seltos_cameras_rig/seltos_cams/`
  and confirmed the report shows `n_gross: {0: 0, 1: 0}`, `inlier_rms == rms` for both cameras
  (0.797px / 0.572px, matching ADR-0011 to 3 places), and `_reject_outlier_observations` at
  `gate_px=10.0` drops **0 of 64** real observations — the gate is a genuine no-op on clean data,
  not merely asserted. At an aggressive `gate_px=1.5`, only 1/64 drops (max real per-observation
  RMS here is 1.57px), showing the gate responds sensibly rather than being trivially inert.
- **Real data, mis-decoding build** (from the authoring session, cv2 4.10, not reproducible on
  this machine — flagged, not silently assumed): max camera error 294.6px -> 2.61px, BA rms
  30.6px -> 0.73px, extrinsic unchanged (1.193m/178.6deg) with the gate; without the gate,
  down-weighting alone gives intrinsics identical to the gated run and an extrinsic within
  0.1deg/3.5mm — the basis for choosing (2) as the default in the Decision above.
  Table shows cam1 raw rms 42.7px alongside `inl_rms` 0.636px and the blunder note; live-view
  per-camera depth-error cam0 0.40 / cam1 0.55 (both stay afloat).
- **New synthetic regression coverage** (`tests/rig/test_outlier_rejection.py`,
  `tests/rig/test_report.py`, `tests/rig/test_web3d.py` — portable, no specific OpenCV build
  required): a manufactured whole-observation corruption (mimicking a mis-decoded board) is
  correctly separated from clean observations by `_observation_reproj_rms`; the gate is a no-op
  on clean synthetic data and drops exactly the corrupted observation (plus, in a 2-camera rig,
  possibly its same-frame sibling — the shared per-frame pose estimate can itself be dragged
  when only 2 views back it, a real cascade through `RigState.object_poses`, not a gate defect);
  `calibrate_rig(reproj_gate_px=...)` recovers the correct extrinsic (<2% baseline, <1deg
  rotation) after gating a corrupted observation out. `ErrorStats.inlier_rms`/`n_gross` verified
  against a known array at the `GROSS_PX` boundary; `render_report`'s note appears only when
  `n_gross > 0` and correctly points at `inl_rms`. `_camera_frame_errors` verified to return the
  median (not mean) of per-point errors on a manufactured single-corner blunder, confirming the
  "one bad corner can't sink the camera" property directly rather than by inspection.
- **Governance gates**: `ruff check .`, `lint-imports` (6/6 contracts kept),
  `check_traceability.py --check` all pass.

## Consequences

**Positive**
- A correctly-calibrated rig with a handful of mis-detections now reads as correct (median /
  `inl_rms` visibly good, a clear note explaining the raw max/rms), instead of presenting as an
  unexplained ~25px failure — the actual user-facing symptom this ADR traces back to.
- The estimate was never wrong; only the reporting was misleading. Fixing reporting is a smaller
  intervention than fixing an estimator that already worked, and it doesn't discard real
  detections a future refinement stage (e.g. `refine_object_structure`) could still use.
- MC-Calib's `ransac_threshold` hard-drop semantics remain available, verified to reproduce the
  originating bug's fix, for a caller that specifically wants literal parity or wants a cleaned
  `object_obs` written back out (e.g. to `save_path`).

**Negative / costs**
- The shipped `ransac_threshold` config field is still not applied by `calibrate_from_config`
  by default (see Scope deferred) — a user relying solely on the YAML template's inline comment
  ("removes only gross outliers") without reading this ADR could reasonably expect auto-drop
  behavior they will not get from the CLI path.
- `GROSS_PX = 5.0` is a fixed module-level constant, not yet config-driven — a rig with a
  legitimately noisier but still-correct sub-5px fit and a different rig with genuine 5px+
  systematic error would both be reported identically as "no gross outliers"; not exercised by
  any real dataset seen so far, but not independently guarded either.

## Scope explicitly deferred (not accidental omissions)

- **Wiring `cfg.ransac_threshold` into `calibrate_from_config`'s `reproj_gate_px`.** Considered
  and rejected for this ADR: the down-weight+report default is deliberately the safer behavior
  (never silently discards a detection), and auto-wiring would reintroduce an implicit
  drop-by-default the Decision above specifically chose not to ship. The shipped YAML templates'
  `ransac_threshold: 10` comment should be read as "the value MC-Calib-parity callers would use
  with `reproj_gate_px` explicitly," not as live CLI behavior — worth a follow-up doc/comment
  pass on the templates themselves if this proves confusing in practice.
- **A config-driven `gross_px` threshold** — `report.GROSS_PX` stays a fixed constant; making it
  configurable was out of scope for the reporting fix and has no real-data motivation yet.

## Alternatives considered

- *Auto-apply the hard-drop gate by default from `calibrate_from_config`* — rejected: the
  measured intrinsics/extrinsic are identical whether the blunder is dropped or down-weighted,
  so auto-dropping buys no estimate-quality improvement over the default, at the cost of
  silently discarding a real detection a caller might want back (debugging, re-labeling,
  `save_reprojection` output) without being asked.
- *Raise on any gross outlier instead of reporting* — rejected: the estimator handles it
  correctly; failing the run over a condition the BA already resolves would be a worse user
  experience than the pre-ADR misleading-metrics failure it replaces.
