# ADR-0024 — One pixel-centre convention for every detector

- **Status:** Accepted (recorded 2026-10-22)
- **Deciders:** maintainer
- **Relates to:** ARC-DETECT, FR-CALIB-004, IFC keypoint cache
- **Supersedes:** `_canonical_charuco_corners` normalising to the OpenCV 4.x ChArUco convention

## Context

Camera models, `findChessboardCornersSB` and `cornerSubPix` place integer coordinates at pixel
centres. Against rendered ground truth (5,699 corners; perspective, radial distortion, noise),
raw OpenCV 4.12 `CharucoDetector` corners are **(+0.491, +0.490) px** off that convention, while
`cornerSubPix` lands at (0.001, −0.000) px. OpenCV 5.0 reports ChArUco corners half a pixel
lower than 4.x, i.e. centred. DS-MSP normalised everything to the 4.x convention, so it added
the bias on 5.0 too — the production pin (`opencv-python==5.0.0.93`) included.

With intrinsics refined, the offset moves the principal point by exactly 0.5 px and leaves the
extrinsics unchanged. With intrinsics held at independently estimated values, it leaks into the
extrinsics: on MC-Calib Seq04 the camera-centre difference to the published fit is 0.29 →
0.16 units after correction.

## Decision

1. `_canonical_charuco_corners` converts to the pixel-centre convention: −0.5 px on OpenCV 4.x,
   unchanged on 5.0.x, and it refuses unmeasured versions. It runs before the optional
   `cornerSubPix`.
2. Keypoint caches written by DS-MSP carry `pixel_convention: pixel_center`. Readers apply a
   label or an explicit `keypoints_pixel_convention` (`pixel_center` | `opencv4_charuco`).
   Unlabeled caches are read unchanged with a warning, because their producer is unknown;
   caches from earlier DS-MSP on OpenCV 4.x must be migrated with `opencv4_charuco` or
   re-detected.
3. A detector-level truth test is the guard (mean bias < 0.05 px). Extrinsic tests alone
   cannot catch this: on the pinhole benchmark the old convention moves rotation only
   0.004° → 0.009°.

## Consequences

- Principal points from ChArUco calibrations shift by −0.5 px against earlier DS-MSP results;
  extrinsics with refined intrinsics are unchanged.
- After the fix, bias measures 0.01 px on 4.12 and 0.002 px on 5.0.
- Strong distortion still biases ChArUco's homography-interpolated corners (about 0.06–0.18 px
  directional, 0.25–0.4 px RMS on a double-sphere camera). Native-model corner refinement is
  the open follow-up; a larger `cornerSubPix` window makes it worse because the window reaches
  the markers.

## Verification

- `tests/calib/test_charuco_compat.py`: version rules and rendered-truth bias.
- `tests/rig/test_image_truth_benchmark.py`: full pipeline on rendered images.
