# Calibrate a camera

Recover a camera's **intrinsics** from images of a calibration board, start to finish.

This page gives you two working recipes and tells you which to pick. It's a task recipe,
not a tutorial — for *why* the optimizer constrains distortion the way it does, see
[Projection validity and FOV](../explain/projection_validity_and_fov.md).

For the full guided walk-through with diagnostics, see the
[calibration capstone](../learn/capstone_calibrating_a_real_camera.md).

**Prerequisites**
- DS-MSP installed (`pip install ds_msp`).
- Recipe 1 (AprilGrid, Python API): the detection extra and a board dataset —
  `pip install ds_msp[calib]` plus AprilGrid footage (e.g. TUM-VI calib).
- Recipe 2 (`ds-msp-calibrate` CLI): nothing extra for checkerboard/<abbr title="A chessboard
  with ArUco markers embedded in each square, so corners stay identifiable even partially
  occluded.">ChArUco</abbr> boards — real photos of your printed board and `opencv-python`
  (which ships with the base install) are enough. AprilGrid boards through the CLI still need
  the same `pip install ds_msp[calib]` extra as recipe 1.

## Pick your recipe

Choose by what data you have and whether you want to write Python.

| You have… | Use | Model | Detector |
| :-- | :-- | :-- | :-- |
| AprilGrid footage and want the Python API — or you already have your own 2D↔3D correspondences | [Recipe 1 — generic calibrator](#recipe-1-generic-calibrator-any-model) | any (`KannalaBrandtModel`, `DoubleSphereModel`, …) | `detect_aprilgrid` (needs `[calib]`) |
| A folder of checkerboard/ChArUco/AprilGrid images and want a config-driven run, no Python | [Recipe 2 — the ds-msp-calibrate CLI](#recipe-2-the-ds-msp-calibrate-cli-no-python) | any of the 8 models (`radtan`, `kb`, `ucm`, `eucm`, `ds`, `ocam`, `dsplus`, `eucmplus`) | built-in board detectors (`ds_msp.calib.board`) |
| Only an existing Kalibr/Basalt YAML | nothing to calibrate — load it directly | as written | n/a |

## Recipe 1 — generic calibrator (any model)

The modern path.
[`ds_msp.calib.calibrate`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/calib/bundle.py)
<abbr title="Jointly refine a camera model and every per-image pose by minimizing total
reprojection error.">bundle-adjusts</abbr> *any* `CameraModel` from 3D↔2D correspondences,
using the model's analytic projection Jacobian and a robust loss.

You supply correspondences one of two ways: detect them from AprilGrid frames (steps 1–2
below), or build them yourself from known board geometry and call `calibrate` directly (step 3).

### 1. Detect AprilGrid corners

[`detect_aprilgrid`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/calib/detect.py)
takes a list of image paths and returns one `{tag_id: (4, 2) pixels}` dict per kept frame. It
needs TUM-VI's calibration footage (`bash scripts/download_datasets.sh tumvi`), so this excerpt
doesn't run standalone — the real, runnable version is
[`examples/03_calibrate_tumvi_aprilgrid.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/examples/03_calibrate_tumvi_aprilgrid.py#L92-L96):

```python
import glob
from ds_msp.calib import detect_aprilgrid

frames = sorted(glob.glob(
    "datasets/tumvi/dataset-calib-cam1_512_16/mav0/cam0/data/*.png"))
detections = detect_aprilgrid(frames, family="t36h11", scales=(1, 2, 3))
print(len(detections))            # -> number of frames that kept >= min_tags (6) tags
```

`len(detections)` is normally smaller than `len(frames)`. `detect_aprilgrid` silently drops
any frame that resolves fewer than `min_tags` (6) tags — a shorter list means weak frames
were filtered, not that detection failed.

/// note
`detect_aprilgrid` needs the optional `aprilgrid` backend (`pip install ds_msp[calib]`). It
defaults to Kalibr's 2-cell tag border, which stock AprilTag-3 / aruco get wrong. The
`scales=(1, 2, 3)` multi-pass recovers peripheral tags the fisheye shrinks below the
detector's size gate.
///

### 2. Build 3D↔2D correspondences

[`AprilGridTarget`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/calib/targets.py)
holds the board geometry and turns the detections into the three lists `calibrate` consumes.
Set `tag_rows`, `tag_cols`, `tag_size` (metres), and `tag_spacing` (Kalibr's gap/size ratio)
to match your printed board. Same non-standalone caveat as step 1 — continued from
[`examples/03`](https://github.com/Munna-Manoj/DS-MSP/blob/main/examples/03_calibrate_tumvi_aprilgrid.py#L95-L99):

```python
# continues from step 1 (detections)
from ds_msp.calib import AprilGridTarget

target = AprilGridTarget(tag_rows=6, tag_cols=6, tag_size=0.088, tag_spacing=0.3)
X_world, keypoints, visibility = target.build_correspondences(detections)
# X_world[i]:   (Nᵢ, 3) board points, metres
# keypoints[i]: (Nᵢ, 2) detected pixels
# visibility[i]:(Nᵢ,)   bool mask of corners to use in image i
print(len(X_world), X_world[0].shape)   # -> n_kept_frames, (Nᵢ, 3)
```

`build_correspondences` marks every returned corner `True` — it only returns corners it
detected.

/// tip
To exclude specific corners (say, ones you flagged as mis-detected), replace an entry in
`visibility` with your own boolean mask before passing it to `calibrate`. And `tag_size` only
sets absolute scale — it moves the recovered extrinsic translations, not `fx, fy, cx, cy` or
distortion. A slightly wrong board size still gives correct intrinsics.
///

### 3. Bundle-adjust from a seed model

Pass a **seed model** — its type picks the model to fit, its values seed the intrinsics — plus
the three correspondence lists.

`loss="cauchy"` down-weights mis-detected corners instead of letting one drag the fit;
`f_scale` is the residual scale in pixels where down-weighting kicks in.

This example builds its own correspondences from a real bundled checkerboard fixture, so it
runs standalone with no dataset download:

{* docs_src/how_to/calibrate_any_model/bundle_adjust.py hl[30:33] *}

<!-- termynal -->
```
$ python -m docs_src.how_to.calibrate_any_model.bundle_adjust
True
0.5768
DoubleSphereModel(fx=737.994, fy=737.123, cx=951.364, cy=517.716, xi=0.2350, alpha=0.8203)
```

`result` is a dict with `model` (the fitted model), `poses` (a `(rvec, tvec)` per image),
`rms_px` (reprojection <abbr title="Root Mean Square">RMS</abbr> over valid corners, the same
under any loss), and `success`.

Read `success` and `rms_px` as a pass/fail check. `success` reports whether the optimizer
converged.

This bundled example lands at `rms_px ≈ 0.58` from only two checkerboard views — real
detected corners, not synthetic.

An `rms_px` above roughly 1 px on richer data points to a bad seed, a wrong board geometry,
or the wrong model type.

/// tip
Continuing from steps 1–2 (AprilGrid correspondences) instead? The call is identical — pass
the `X_world, keypoints, visibility` from `target.build_correspondences(...)` and a seed
matching your board's camera, e.g. `KannalaBrandtModel(fx=180, fy=180, cx=W/2, cy=H/2)` for a
512×512 TUM-VI frame. See the exact call in
[`examples/03`, lines 104–112](https://github.com/Munna-Manoj/DS-MSP/blob/main/examples/03_calibrate_tumvi_aprilgrid.py#L104-L112).
///

> The [capstone](../learn/capstone_calibrating_a_real_camera.md) runs the AprilGrid path
> end-to-end on real TUM-VI cam0 footage with detection diagnostics and an
> against-the-published-YAML check. Follow it for the full procedure and the verified accuracy
> number.

To fit a different model, swap the seed for another `ds_msp.models` class — e.g.
`KannalaBrandtModel(fx, fy, cx, cy)` or `EUCMModel(fx, fy, cx, cy, alpha, beta)`. The seed's
type picks the model to fit.

Double Sphere distortion stays in its well-posed range automatically; see
[Projection validity and FOV](../explain/projection_validity_and_fov.md).

## Recipe 2 — the `ds-msp-calibrate` CLI (no Python)

If you have a folder of board images and don't want to write any Python,
[`ds-msp-calibrate`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/calib/cli.py) is a
config-driven console command. It runs the same detect → bundle-adjust → report pipeline as
recipe 1.

It works for checkerboard, ChArUco, or AprilGrid boards, and any of the eight camera models.

### 1. Write and edit a config

<!-- termynal -->
```
$ pip install ds-msp
$ ds-msp-calibrate --init-config calib_config.yml
wrote base calib_config template to: calib_config.yml
```

`opencv-python` (the checkerboard/ChArUco detector) ships with the base install, so this needs
no extras.

This copies
[`calib_config.template.yml`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/calib/configs/calib_config.template.yml)
next to your images. Edit the fields that matter — `board.type`, `board.rows`/`board.cols`
(**interior** corners, not squares), `board.square_size` (metres — this sets the fit's metric
scale), `camera_model`, and `images_path`:

```yaml
board:
  type: charuco        # checkerboard | charuco | aprilgrid
  rows: 6
  cols: 6
  square_size: 0.1
  legacy: true          # only needed for boards generated before OpenCV 4.7
camera_model: kb        # radtan | ds | ucm | eucm | kb | ocam | dsplus | eucmplus
images_path: "calibration_images_0"
save_path: "Results_dsmsp_calib_charuco"
```

### 2. Run it

This is the real, verified output of that exact config against a 58-image, 2592×1800 fisheye
capture of a 6×6 ChArUco board:

<!-- termynal -->
```
$ ds-msp-calibrate --config calib_config.yml --quiet
=== charuco / kb: 58 images, 2592x1800 ===

board: charuco   model: kb   images: 58/58 detected
     n    mean  median     p95     max     rms
----------------------------------------------
  1949   0.372   0.286   0.963   3.768   0.509

verdict: PASS  median 0.286px, p95 0.963px <= 1.00/3.00px

wrote Results_dsmsp_calib_charuco/camchain.yaml
```

That's the same two things recipe 1 gives you: a fitted model and a reprojection-error number.

It adds the full error distribution (mean/median/p95/max/rms, not just one number), plus a
PASS/WARN/FAIL verdict whose exit code (`0`/`1`) a CI job can check directly.

### 3. Load the result back

```python
import ds_msp.calib as calib

cam = calib.load_camera("Results_dsmsp_calib_charuco/camchain.yaml")
print(cam)
# -> KannalaBrandtModel(fx=996.099, fy=995.973, cx=1276.329, cy=876.123,
#                        k=[0.24009, -0.02803, -0.09016, 0.02923])

uv, valid = cam.project(points_3d)   # points_3d: (N, 3) camera-frame points, metres
```

`load_camera` reads the `camchain.yaml` this run just wrote and returns a ready `CameraModel` —
`cam.project(...)` works immediately, no re-parsing.

It works for every model the CLI can write (`radtan`, `ds`, `ucm`, `eucm`, `kb`, `dsplus`,
`eucmplus`).

`ocam` has no Kalibr representation yet, so `load_camera`/`save_kalibr` raise a documented
`ValueError` for it instead of writing something wrong — calibrate `ocam` and use
`result["model"]` in memory instead.

/// note
The `robust`/`robust_scale`/`gnc` config keys are recipe 2's equivalent of recipe 1's
`loss`/`f_scale` arguments to `calibrate(...)` — the same Cauchy-by-default robust kernel,
tuned from the config instead of a Python call. See §3 of the
[CALIBRATE_GUIDE](../CALIBRATE_GUIDE.md#3-make-a-config) for the full field reference, why
ChArUco beats plain checkerboard on wide <abbr title="Field of View">FOV</abbr> (100% vs 19%
detection yield, measured on this same capture), and troubleshooting.
///

## Try it yourself

Re-run the fit with plain L2 instead of the robust default. Predict first: with a few
mis-localized peripheral corners, does the reprojection error go up or down?

The robust kernel keeps every corner but down-weights large residuals, which typically
tightens the *median* — but don't expect every statistic to move the same way. On this exact
dataset, switching to L2:

- worsens mean (0.372 → 0.381) and median (0.286 → 0.309),
- but *improves* p95 (0.963 → 0.944), max (3.768 → 2.373), and rms (0.509 → 0.481) — the
  outliers L2 lets pull the fit happen to land closer to a few peripheral corners, at the cost
  of the bulk.

**Read the whole distribution, not one number** — that's the actual lesson, not "robust always
wins."

On recipe 1: pass `loss="linear"` to `calibrate(...)` and compare `result["rms_px"]`.

On recipe 2: re-run with `ds-msp-calibrate --config calib_config.yml --set robust=none` (the
CLI's config equivalent — there's no `loss=` kwarg here). Compare the full
mean/median/p95/max/rms distribution against the `cauchy` run above, not just one field.

## Next steps

- **Walk it through with diagnostics:** [calibration capstone](../learn/capstone_calibrating_a_real_camera.md).
- **Understand the parameter domain:** [Projection validity and FOV](../explain/projection_validity_and_fov.md).
- **Convert the fitted model** to another representation, or undistort images with it — see the
  other [how-to recipes](README.md).
