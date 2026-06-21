# Calibrate a camera

Recover a camera's intrinsics from images of a calibration board, start to finish. This page
gives you two working recipes and tells you which to pick. It is a task recipe — for *why* the
optimizer constrains distortion the way it does, see
[Projection validity and FOV](../explain/projection_validity_and_fov.md); for the full guided
walk-through with diagnostics, see the
[calibration capstone](../learn/capstone_calibrating_a_real_camera.md).

> **Prerequisites**
> - DS-MSP installed (`pip install ds_msp`).
> - For recipe 1 (AprilGrid): the detection extra and a board dataset —
>   `pip install ds_msp[calib]` plus AprilGrid footage (e.g. TUM-VI calib).
> - For recipe 2 (bundled): nothing extra — the COCO checkerboard data ships in the repo.

## Pick your recipe

Choose by what data and board you have.

| You have… | Use | Model | Detector |
| :-- | :-- | :-- | :-- |
| AprilGrid board footage (TUM-VI, EuRoC, Kalibr) | [Recipe 1 — generic calibrator](#recipe-1-generic-calibrator-any-model) | any (`KannalaBrandtModel`, `DoubleSphereModel`, …) | `detect_aprilgrid` (needs `[calib]`) |
| COCO-style checkerboard annotations (`anns.json`) | [Recipe 2 — bundled Double Sphere script](#recipe-2-bundled-double-sphere-script) | Double Sphere only | none — corners are in the JSON |
| Only an existing Kalibr/Basalt YAML | nothing to calibrate — load it directly | as written | n/a |

## Recipe 1 — generic calibrator (any model)

The modern path. [`ds_msp.calib.calibrate`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/calib/bundle.py)
bundle-adjusts *any* `CameraModel` from 3D↔2D correspondences using the model's analytic
projection Jacobian and a robust loss. You supply correspondences three ways: detect them from
AprilGrid frames (below), or build them yourself and skip straight to step 3.

### 1. Detect AprilGrid corners

[`detect_aprilgrid`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/calib/detect.py)
takes a list of image paths and returns one `{tag_id: (4, 2) pixels}` dict per kept frame.

```python
import glob
from ds_msp.calib import detect_aprilgrid

frames = sorted(glob.glob(
    "datasets/tumvi/dataset-calib-cam1_512_16/mav0/cam0/data/*.png"))
detections = detect_aprilgrid(frames, family="t36h11", scales=(1, 2, 3))
print(len(detections))            # -> number of frames that kept >= min_tags (6) tags
```

Notice that `len(detections)` is normally smaller than `len(frames)`. `detect_aprilgrid` silently
drops any frame that resolves fewer than `min_tags` (6) tags, so a shorter list means weak frames
were filtered — not that detection failed.

> **Note** `detect_aprilgrid` needs the optional `aprilgrid` backend (`pip install ds_msp[calib]`).
> It defaults to Kalibr's 2-cell tag border, which stock AprilTag-3 / aruco get wrong. The
> `scales=(1, 2, 3)` multi-pass recovers peripheral tags the fisheye shrinks below the detector's
> size gate.

### 2. Build 3D↔2D correspondences

[`AprilGridTarget`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/calib/targets.py)
holds the board geometry and turns the detections into the three lists `calibrate` consumes.
Set `tag_rows`, `tag_cols`, `tag_size` (metres), and `tag_spacing` (Kalibr's gap/size ratio)
to match your printed board.

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

Notice that `build_correspondences` marks every returned corner `True` — it only returns corners
it detected. To exclude specific corners (say, ones you flagged as mis-detected), replace an entry
in `visibility` with your own boolean mask before passing it to `calibrate`.

> **Note** `tag_size` sets only absolute scale — it moves the recovered extrinsic translations,
> not `fx, fy, cx, cy` or distortion. A slightly wrong board size still gives correct intrinsics.

### 3. Bundle-adjust from a seed model

Pass a seed model (its **type** picks the model to fit; its values seed the intrinsics) and the
three lists. `loss="cauchy"` down-weights mis-detected corners instead of letting one drag the
fit; `f_scale` is the residual scale in pixels at which down-weighting kicks in.

```python
# continues from step 2 (X_world, keypoints, visibility)
from ds_msp.calib import calibrate
from ds_msp.models import KannalaBrandtModel

seed = KannalaBrandtModel(fx=180, fy=180, cx=256, cy=256)   # 512x512 TUM-VI frame
result = calibrate(seed, X_world, keypoints, visibility,
                   loss="cauchy", f_scale=0.5)

print(result["success"])     # -> True
print(result["rms_px"])      # -> sub-pixel reprojection RMS
model = result["model"]      # fitted KannalaBrandtModel
print(model.params)          # -> [fx, fy, cx, cy, k1, k2, k3, k4]
```

`result` is a dict with `model` (the fitted model), `poses` (a `(rvec, tvec)` per image),
`rms_px` (true reprojection RMS over valid corners, the same under any loss), and `success`.

Read the two as a pass/fail check. `success` reports whether the optimizer converged. A good fit
on real AprilGrid footage lands at sub-pixel `rms_px`; an `rms_px` above roughly 1 px points to a
bad seed, a wrong board geometry (`tag_rows`, `tag_cols`, `tag_spacing`), or the wrong model type.

> The [capstone](../learn/capstone_calibrating_a_real_camera.md) runs this end-to-end on real
> TUM-VI cam0 footage with detection diagnostics and an against-the-published-YAML check. Follow
> it for the full procedure and the verified accuracy number.

To fit a different model, swap the seed for another `ds_msp.models` class — e.g.
`DoubleSphereModel(fx, fy, cx, cy, xi, alpha)` or `EUCMModel(fx, fy, cx, cy, alpha, beta)`. The
seed's type picks the model to fit. Double Sphere distortion stays in its well-posed range
automatically; see [Projection validity and FOV](../explain/projection_validity_and_fov.md).

## Recipe 2 — bundled Double Sphere script

If your corners are already in a COCO-style `anns.json` (the format shipped with the repo), the
bundled scripts calibrate a Double Sphere camera with no detector and no extra install. They read
`anns.json`, write JSON results, and render a reprojection check.

```bash
python calibrate.py    # reads anns.json -> writes results/calibration_params.json
python validate.py     # reprojection check -> results/visualizations/
```

On the bundled data this converges to:

```text
[REAL] Optimization success: True
[REAL] RMS reprojection error: 0.6367 px
```

with recovered intrinsics `fx≈711.6, fy≈711.2, cx≈949.2, cy≈518.8, ξ≈0.183, α≈0.809`
(written to `results/calibration_params.json`).

> **Note** The optimizer constrains Double Sphere distortion to the well-posed range
> `α ∈ [0, 1]`, `ξ ∈ [-1, 1]` (matching Basalt/Kalibr). Outside it the projection folds back and
> unprojection can't invert it. Real fisheye lenses sit roughly in `ξ ∈ [-0.2, 0.6]`. Full reasoning:
> [Projection validity and FOV](../explain/projection_validity_and_fov.md).

## Try it yourself

Re-run the fit with `loss="linear"` (plain L2) instead of `"cauchy"`. Predict first: with a few
mis-localized peripheral corners, does `rms_px` go up or down? The robust kernel keeps every corner
but down-weights large residuals, so L2 typically reports a *worse* fit when outliers are present.

You can run this on recipe 1 if you have the TUM-VI footage, or on recipe 2's bundled
`calibrate.py` path, which takes the same `loss` argument over the shipped data.

## Next steps

- **Walk it through with diagnostics:** [calibration capstone](../learn/capstone_calibrating_a_real_camera.md).
- **Understand the parameter domain:** [Projection validity and FOV](../explain/projection_validity_and_fov.md).
- **Convert the fitted model** to another representation, or undistort images with it — see the
  other [how-to recipes](README.md).
