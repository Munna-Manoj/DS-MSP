# Calibrate a single camera

Recover one camera's **intrinsics** — focal length, principal point, distortion — from a
folder of calibration-board images, driven by a single config.

This is the single-camera analogue of [`ds_msp[rig]`](RIG_CALIBRATION_GUIDE.md)'s
config-driven pattern. Pick a board type
(checkerboard / <abbr title="A chessboard with ArUco markers embedded in each square, so
corners stay identifiable even partially occluded.">ChArUco</abbr> / AprilGrid) and a camera
model (`radtan`, `kb`, `ucm`, `eucm`, `ds`, `ocam`, `dsplus`). Detection is a
swappable front end — every board type feeds the same tested, model-agnostic
bundle-adjustment backend.

**Prerequisites**

- `ds-msp` installed: `pip install ds-msp` (or `git clone` + `pip install -e .`).
- Checkerboard/ChArUco boards: nothing extra — `opencv-python` ships with the base install.
- AprilGrid boards: the detection extra, `pip install ds-msp[calib]`.
- A calibration board and photos of it — see [§2](#2-prepare-your-data) if you don't have one
  yet.

## Quick start

Write yourself a starter config, edit it, and run it:

<div class="termy">

```console
$ pip install ds-msp
$ ds-msp-calibrate --init-config calib_config.yml
wrote base calib_config template to: calib_config.yml
```

</div>

Edit the file — board type/geometry, `images_path`, `camera_model` — then run it:

<div class="termy">

```console
$ ds-msp-calibrate --config calib_config.yml
```

</div>

/// tip
`ds-msp-calibrate` is a real console command from `pip install ds-msp` alone — no repo clone
needed. A git-clone checkout can equivalently run `python scripts/calibrate.py`; it's the exact
same CLI either way.
///

---

## 1. What you get out

<div class="termy">

```console
$ ds-msp-calibrate --config calib_config.yml --quiet
=== charuco / kb: 58 images, 2592x1800 ===

board: charuco   model: kb   images: 58/58 detected

     n    mean  median     p95     max     rms
----------------------------------------------
  1954   0.372   0.287   0.959   3.761   0.509

verdict: PASS  median 0.287px, p95 0.959px <= 1.00/3.00px

wrote Results_dsmsp_calib_charuco/camchain.yaml
```

</div>

| File | Contents |
|------|----------|
| `<save_path>/camchain.yaml` | Kalibr-format intrinsics — `camera_model`, `intrinsics`, `distortion_model`, `distortion_coeffs`, `resolution` (see [§5](#5-load-the-result-back-into-a-camera-instance) to load it back into a `CameraModel`) |
| console report | detection yield (images detected vs. total), the full reprojection-error **distribution** (mean/median/p95/max/<abbr title="Root Mean Square">RMS</abbr> — not just one RMS number), and a PASS/WARN/FAIL verdict |

The process exit code is `0` for PASS/WARN, `1` for FAIL. A CI/cron caller can tell success
from failure without parsing text.

---

## 2. Prepare your data

Single-camera calibration just needs **one flat folder of images** — any filenames, any
naming.

- Unlike `ds_msp[rig]`'s strict `Cam_001/00000.png` multi-camera folder convention, there's no
  naming scheme to follow.
- You can mix resolutions across separate runs (not within one run — intrinsics are for one
  physical camera/lens).

```
my_capture/
├── calib_0000.png
├── calib_0001.png
├── calib_0002.png
└── ...
```

Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`. Use `pattern` in the
config (or `--pattern` on the CLI) to select a subset by glob, e.g. `"*.png"`.

### The calibration target

| Board | Best for | Notes |
|---|---|---|
| **ChArUco** | Fisheye / wide-<abbr title="Field Of View — the angular extent of the scene a lens captures.">FOV</abbr> (recommended default) | Marker IDs disambiguate partial/oblique views; supports **multiple simultaneous board definitions** — see `boards:` in [§3](#3-make-a-config) |
| **AprilGrid** | Fisheye / wide-FOV | Kalibr's own recommended target for exactly this case |
| **Checkerboard** | Narrow/mid FOV only (≲120°) | `cv2.findChessboardCornersSB` (Duda & Frese, BMVC 2018) — **not** documented or benchmarked as fisheye-robust; MC-Calib has no checkerboard support at all, and Kalibr's own docs recommend AprilGrid over it for wide lenses. See `ds_msp/detect/checkerboard.py`'s module docstring and [§7](#7-worked-example-robustness-real-data) for a real, measured example of this failure mode |

You must know the board's physical geometry (interior corner rows/cols, square size in
metres) and put it in the config.

---

## 3. Make a config

Generate a fully-commented starter and edit it:

<div class="termy">

```console
$ ds-msp-calibrate --init-config calib_config.yml
wrote base calib_config template to: calib_config.yml
```

</div>

This copies [`ds_msp/calib/configs/calib_config.template.yml`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/calib/configs/calib_config.template.yml).
Below is every field that matters.

### Board geometry — *must match your printed board*

| Key | Meaning |
|-----|---------|
| `board.type` | `checkerboard` \| `charuco` \| `aprilgrid` |
| `board.rows`, `board.cols` | **interior corner** counts (checkerboard/charuco) — count corners where four squares meet, not squares |
| `board.square_size` | physical square size, metres. **This sets the metric scale** of the fit |
| `board.length_marker` | (ChArUco only) inner ArUco marker length; defaults to `0.75 * square_size` if unspecified |
| `board.legacy` | (ChArUco only) `true` matches boards generated before OpenCV 4.7's marker-pattern change |
| `board.boards` | (ChArUco only) a list of `{rows, cols, square_size, length_marker}` dicts for **multiple simultaneous physical boards** — each board sighting (whichever board, in whichever image) becomes its own independent observation, no rigid-fusion setup needed |
| `board.tag_rows`, `board.tag_cols`, `board.tag_size`, `board.tag_spacing`, `board.family` | (AprilGrid only) grid geometry and AprilTag family (default `t36h11`) |

### Camera model

| Key | Meaning |
|-----|---------|
| `camera_model` | `radtan`, `kb`, `ucm`, `eucm`, `ds`, `ocam`, or `dsplus` |

See [Choosing a model](explain/choosing_a_camera_model.md) for how to pick one for your
lens's FOV.

### Inputs / outputs

| Key | Meaning |
|-----|---------|
| `images_path` | folder of calibration images |
| `pattern` | glob pattern selecting files within it (default `"*"`) |
| `save_path` | output directory (`camchain.yaml` written here); omit to skip writing a file |
| `output_format` | only `kalibr` is supported today |

### Bundle-adjustment tuning

| Key | Meaning |
|-----|---------|
| `robust` | robust loss kernel: `cauchy` (default), `huber`, `geman_mcclure`, `barron`, `none` |
| `robust_scale` | `"auto"` or a fixed float |
| `gnc` | graduated non-convexity (more robust to heavy outlier contamination, slower) |
| `multi_start` | try multiple focal-length starting points (default `true`) |
| `n_restarts`, `seed`, `max_nfev` | multi-start count, RNG seed, max solver iterations |

/// note
Relative paths (`images_path`, `save_path`) resolve against the **config file's** directory.
///

Override any value on the CLI without editing the file:

<div class="termy">

```console
$ ds-msp-calibrate --config calib_config.yml --set board.rows=7 --set camera_model=dsplus
```

</div>

---

## 4. Run it

<div class="termy">

```console
$ ds-msp-calibrate --config calib_config.yml
```

</div>

What happens internally:

1. **Detect** the configured board in every image (live per-image progress on a <abbr
   title="TeleTYpewriter — an interactive terminal, as opposed to output piped to a file.">TTY</abbr>).
2. **Seed** a from-scratch pinhole `K` from the detected correspondences.
3. **Bundle-adjust** the chosen model's intrinsics against every detected view, robustly.
4. **Report** the full reprojection-error distribution plus a PASS/WARN/FAIL verdict.
5. **Write** `camchain.yaml` to `save_path`.

Non-config, one-off flags also work for a quick check:

<div class="termy">

```console
$ ds-msp-calibrate ./images --board checkerboard --rows 6 --cols 9 --square-size 0.025 --model ds
```

</div>

/// tip
Silence the live progress lines (e.g. for CI logs) with `--quiet`. Tune the PASS/WARN verdict
thresholds with `--pass-px`/`--warn-px` (defaults: 1.0 / 3.0 px median).
///

---

## 5. Load the result back into a camera instance

`camchain.yaml` is meant to be loaded straight back into a ready, usable
[`CameraModel`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/core/contracts.py) —
not just inspected as a file:

```python
import ds_msp.calib as calib

cam = calib.load_camera("Results_dsmsp_calib_charuco/camchain.yaml")
print(cam)                      # e.g. KannalaBrandtModel(fx=996.1, fy=996.0, cx=1276.8, ...)

uv, valid = cam.project(points_3d)   # points_3d: (N, 3) camera-frame points, metres
                                      # uv: (N, 2) pixels, ready to use immediately
```

`load_camera` works for every model calibration can produce with `output_format: kalibr`:
`radtan`, `ds`, `ucm`, `eucm`, `kb`, `dsplus`.

/// note
`ocam`'s polynomial parameterization has no Kalibr-native or DS-MSP-extended representation
yet — a documented gap, not a silent one. `to_kalibr_cam`/`save_kalibr` raise a clear
`ValueError` for it rather than writing something wrong.
///

If you need the resolution too:

```python
from ds_msp.io.kalibr import load_kalibr_with_resolution

cam, (width, height) = load_kalibr_with_resolution("camchain.yaml")
```

---

## 6. Detection front end vs. the backend

Every board type is just a different way to turn images into 2D↔3D correspondences
(`ds_msp.calib.board.Board.detect() -> List[Observation]`). The bundle-adjustment backend
(`ds_msp.calib.bundle.calibrate`) that fits the chosen camera model neither knows nor cares
which board produced them.

- **ChArUco** supports multiple simultaneous board definitions with **zero fusion machinery**
  — single-camera calibration only needs independent *views* of a *known* pattern. That's
  unlike `ds_msp[rig]`'s multi-board rigid-object fusion, a genuinely different, harder
  problem: many boards fixed to each other, seen by many cameras, sharing one 3D point cloud.
- **AprilGrid** slots into the exact same shape: one grid sighting per image becomes one
  observation.
- **Checkerboard** is single-board only in this release — see [§7](#7-worked-example-robustness-real-data)
  for why that's a real, measured limitation on wide-FOV fisheye, not a missing feature.

See [ADR-0009](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/process/architecture/decisions/ADR-0009-board-protocol.md) for the full design
rationale.

---

## 7. Worked example & robustness (real data)

On a real 58-image, 2592×1800 fisheye single-camera capture (an MC-Calib-style dataset, not
included in this repo), the **same physical ChArUco board** was calibrated two ways:

- once via its ArUco markers,
- once as a plain (6,6) checkerboard, markers ignored entirely.

### Results

| Board | Model | Images detected | Observations | RMS px | Median px | p95 px | Max px |
|---|---|---|---|---|---|---|---|
| ChArUco | KB | **58/58 (100%)** | 1954 | 0.509 | 0.287 | 0.959 | 3.76 |
| ChArUco | DS+ | **58/58 (100%)** | 1954 | **0.447** | **0.222** | **0.841** | **3.28** |
| Checkerboard | KB | **11/58 (19%)** | 396 | 1.396 | 0.216 | 0.579 | 26.78 |
| Checkerboard | DS+ | **11/58 (19%)** | 396 | 1.405 | 0.179 | 0.525 | 27.08 |

### Takeaways (measured, not asserted)

- **ChArUco: 100% detection. Checkerboard: 19%.** Exactly the failure mode [§2](#2-prepare-your-data)
  warns about — plain-checkerboard detection struggles on wide-FOV fisheye, compounded here by
  the embedded ArUco markers occupying some of the checkerboard's own squares.
- **DS+ beats KB on every metric on the fair, complete comparison** (ChArUco, all 58 images):
  RMS 0.447 vs. 0.509, median 0.222 vs. 0.287, max 3.28 vs. 3.76 px.
- Checkerboard's *median* looks competitive but is survivorship bias — it only succeeded on
  the 11 easiest (least-distorted) images. Its RMS (3× worse) and max error (7× worse) show
  that even within that lucky small sample, some detections are quite poor.
- **Both beat MC-Calib's own published reference** for this camera (median 0.481 px, mean
  0.593 px, `extra_yamls/reprojection_error_data.yml`).

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| "no images directory given" | neither `images_path` (config) nor the CLI's positional argument was set |
| "no board detected in any of the given images" | wrong `board.rows`/`board.cols` (remember: interior corners, not squares), wrong `board.type` for the physical board, or images genuinely don't show the board |
| Low detection yield with `board.type: checkerboard` on a wide lens | expected — see [§2](#2-prepare-your-data)/[§7](#7-worked-example-robustness-real-data); switch to `charuco` or `aprilgrid` |
| `cv2.error` about `markerLength`/`squareLength` | `board.length_marker` conflicts with `board.square_size` — leave it unset to use the `0.75×` default, or set it below `square_size` |
| `ValueError: No Kalibr mapping for model 'ocam'` | documented gap ([§5](#5-load-the-result-back-into-a-camera-instance)) — `ocam`'s polynomial has no Kalibr representation yet; calibrate with another model, or use the in-memory `result["model"]` directly without `output_format: kalibr` |
| Wrong real-world scale | `board.square_size` must be the **physically measured** square size |
| Want to silence live progress (CI logs) | `--quiet`, or `verbose: false` in the config |

---

### See also

- [`ds_msp/calib/configs/calib_config.template.yml`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/calib/configs/calib_config.template.yml) — annotated base config
- [ADR-0009](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/process/architecture/decisions/ADR-0009-board-protocol.md) — the `Board` protocol design
- [`RIG_CALIBRATION_GUIDE.md`](RIG_CALIBRATION_GUIDE.md) — the multi-camera rig analogue (extrinsics + intrinsics, ChArUco only)
- [`docs/learn/`](learn/README.md) — the geometry curriculum (camera models, robust detection, evaluation)
