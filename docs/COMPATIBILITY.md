# Dependency compatibility & known issues

DS-MSP's math foundation (`ds_msp/core`, `data`, `geometry`, `models`) is NumPy-only and
enforced so by import-linter plus a contract test. This limits the numerical dependency
surface; it does not make floating-point results immune to changes in NumPy or the linked
BLAS/LAPACK implementation. The detection front-end wraps OpenCV, whose behavior can also
change between releases independently of Python. Confirmed instances are recorded here so
nobody has to rediscover them.

## Tested versions

| dependency | floor | tested with | notes |
|---|---|---|---|
| `opencv-python` | `>= 4.7,<5.1` | 4.7 - 5.0 | `cv2.aruco.CharucoDetector` exists only from 4.7; later 5.x releases remain excluded until their coordinate convention is measured. |
| `numpy` | -- | 1.26 / 2.x | Math foundation. |
| `scipy` | -- | 1.11+ | Optional layers only (`calib`/`rig` solvers never require it in `core`). |

## Known per-version behavioral caveats

### OpenCV 5.0: ChArUco corner coordinates shift by half a pixel

OpenCV 5.0 changed the ChArUco corner coordinate convention by exactly half a pixel
relative to 4.x builds. Measured on the MC-Calib Blender Scenario_2 reference data
(2026-07-19): every corner detected by OpenCV 5.0.0 sits at a constant offset of
`(-0.4995, -0.4990)` px from the same corners detected by the OpenCV 4.x build that
produced MC-Calib's `detected_keypoints_data.yml`; the residual scatter after removing
that constant is ≤ 0.022 px (detection quality itself is unchanged).

Mitigation and impact:

- **Fresh detections are normalized at the source.** `ds_msp.detect.charuco.detect_image`
  adds `(0.5, 0.5)` to OpenCV 5.0.x output, producing the OpenCV-4/MC-Calib convention
  before observations enter calibration. The original strict parity test therefore applies
  unchanged on every supported OpenCV version.
- **Previously stored unnormalized 5.0.x detections remain different.** Mixing those files
  with canonical detections introduces a systematic half-pixel disagreement; regenerate
  them or translate both coordinates by `(0.5, 0.5)` once at ingestion.

### OpenCV builds that mis-decode ChArUco ids (ADR-0013)

A specific OpenCV build was observed to mis-decode ChArUco corner ids on a correct board,
resecting to a plausible-but-wrong pose with a low per-board residual. Down-weighting
alone hides this; the pipeline therefore reports `inlier_rms`/`n_gross` and offers an
opt-in `reproj_gate_px` hard gate. See ADR-0013 and FR-RIG-018.

## Reporting a new instance

If a dependency upgrade changes calibration output silently (no exception, different
numbers), please open an issue with: dependency + both versions, a minimal input, and the
two outputs. Confirmed instances get (1) an algorithmic mitigation close to the source of
the risk and (2) an entry here — both, per the project's dependency-variability policy.
