# Dependency compatibility & known issues

DS-MSP's math foundation (`ds_msp/core`, `data`, `geometry`, `models`) is NumPy-only and
enforced so by import-linter + a contract test, which keeps the numerics immune to
third-party drift. The detection front-end, however, wraps OpenCV — and OpenCV's
*behavior* can change between its own releases independently of your Python version.
Confirmed instances are recorded here so nobody has to rediscover them.

## Tested versions

| dependency | floor | tested with | notes |
|---|---|---|---|
| `opencv-python` | `>= 4.7` | 4.7 – 5.0 | `cv2.aruco.CharucoDetector` (used by the ChArUco front-end) exists only from 4.7. |
| `numpy` | — | 1.26 / 2.x | Math foundation. |
| `scipy` | — | 1.11+ | Optional layers only (`calib`/`rig` solvers never require it in `core`). |

## Known per-version behavioral caveats

### OpenCV 5.0: ChArUco corner coordinates shift by half a pixel

OpenCV 5.0 changed the ChArUco corner coordinate convention by exactly half a pixel
relative to 4.x builds. Measured on the MC-Calib Blender Scenario_2 reference data
(2026-07-19): every corner detected by OpenCV 5.0.0 sits at a constant offset of
`(-0.4995, -0.4990)` px from the same corners detected by the OpenCV 4.x build that
produced MC-Calib's `detected_keypoints_data.yml`; the residual scatter after removing
that constant is ≤ 0.022 px (detection quality itself is unchanged).

Impact:

- **Self-consistent runs are unaffected.** If all detections in a calibration come from
  the same OpenCV build, the half-pixel convention is absorbed into the principal point
  (`cx`, `cy` move by ~0.5 px) and reprojection quality is identical.
- **Mixing conventions biases the fit.** Reusing corner files detected under 4.x together
  with fresh 5.x detections (or comparing against 4.x-era reference keypoints) introduces
  a systematic half-pixel disagreement. The parity test
  (`tests/calib/test_charuco.py::test_parity_vs_mccalib_keypoints`) is convention-aware:
  it accepts a global ~0 or half-pixel offset and still fails on any non-uniform
  discrepancy (the mis-decode class of ADR-0012).

### OpenCV builds that mis-decode ChArUco ids (ADR-0012)

A specific OpenCV build was observed to mis-decode ChArUco corner ids on a correct board,
resecting to a plausible-but-wrong pose with a low per-board residual. Down-weighting
alone hides this; the pipeline therefore reports `inlier_rms`/`n_gross` and offers an
opt-in `reproj_gate_px` hard gate. See ADR-0012 and FR-RIG-018.

## Reporting a new instance

If a dependency upgrade changes calibration output silently (no exception, different
numbers), please open an issue with: dependency + both versions, a minimal input, and the
two outputs. Confirmed instances get (1) an algorithmic mitigation close to the source of
the risk and (2) an entry here — both, per the project's dependency-variability policy.
