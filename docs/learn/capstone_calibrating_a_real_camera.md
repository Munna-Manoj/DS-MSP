# 🏆 Capstone — calibrate a real fisheye camera, and match the published numbers

> **Run alongside this:** `pip install -e .[calib]` then
> `python examples/03_calibrate_tumvi_aprilgrid.py` (after the [setup](README.md#setup-once)).

Everything else in this curriculum is practice for one claim: **give me raw footage of a
camera looking at a calibration board, and I will recover its lens parameters from scratch
— and they will agree with the numbers the experts published.** No `convert()`, no loading
someone's answer. Detect the corners, bundle-adjust the geometry, compare to truth.

We use TUM-VI's `cam0` calibration sequence — 436 frames of someone waving a 6×6 AprilGrid
in front of a 195° fisheye — and check our result against the Kannala-Brandt calibration
the dataset authors published.

<div class="ds-stats">
  <div class="ds-stat"><span class="ds-stat__value">0.080 px</span><span class="ds-stat__label">median reprojection, all 14,460 corners</span></div>
  <div class="ds-stat"><span class="ds-stat__value">~0.02%</span><span class="ds-stat__label">focal-length agreement vs. published KB</span></div>
  <div class="ds-stat"><span class="ds-stat__value">14,460</span><span class="ds-stat__label">self-detected corners, none discarded</span></div>
</div>

**You'll learn**
- Calibrate a Kannala-Brandt camera from scratch — detect real AprilGrid corners,
  bundle-adjust intrinsics and a per-frame pose with `ds_msp.calib.calibrate`, and compare
  the result to TUM-VI's published reference number-for-number.
- Land on **0.080 px median reprojection** (0.160 px inlier RMS) over all 14,460
  self-detected corners, with focal length matching the published value to **~0.02%**.
- Diagnose a real "detector returns 0 tags on every frame" bug — a 2-cell vs. 1-cell
  AprilTag border mismatch between Kalibr-style boards and stock AprilTag-3 — and fix it.
- See why the Double Sphere model fits the very same corners just as tightly (~0.08 px)
  even though its raw focal number (`fx≈249`) looks nothing like KB's (`fx≈191`) — until you
  correct for DS's own focal gauge.

**Prerequisites**
- Finish [Chapter 2](02_double_sphere_model.md) — the capstone is runnable right after it;
  Chapters 3+ are optional theory, not required.
- Install the calibration extra and the dataset:
  ```bash
  uv pip install -e ".[calib]"
  bash scripts/download_datasets.sh tumvi
  ```
  (see the [project setup](README.md#setup-once) for details).

![AprilGrid detection on real TUM-VI frames](https://raw.githubusercontent.com/Munna-Manoj/DS-MSP/main/assets/learn/aprilgrid_detection.gif)

*Step one, on real data: the AprilGrid detector finds the board's tags (here ~35 of 36) and
their corners in each fisheye frame. Those corners are the measurements we calibrate from.*

## The pipeline (all library code)

```mermaid
graph LR
    A["raw frames"] --> B["detect AprilGrid<br/>detect_aprilgrid()"]
    B --> C["3D↔2D correspondences<br/>AprilGridTarget"]
    C --> D["bundle-adjust<br/>calibrate()"]
    D --> E["compare to<br/>published reference"]
```

1. **Detect** the board corners — `ds_msp.calib.detect_aprilgrid` (the one piece that needs
   a tag backend; see the war-story below).
2. **Correspond** — `ds_msp.calib.AprilGridTarget` knows where every tag corner sits in 3D
   board metres, so a detected `tag_id` becomes a `(X,Y,Z) ↔ (u,v)` pair.
3. **Calibrate** — `ds_msp.calib.calibrate` jointly refines the 6/8 intrinsics and a pose
   per frame by Levenberg-Marquardt, using each model's **analytic** projection Jacobian
   (Chapter 4's payoff: no finite differences, no autodiff).
4. **Compare** — line our `fx, fy, cx, cy, k1..k4` up against the published reference.

We calibrate a **Kannala-Brandt** model precisely because TUM-VI's published reference is
KB — so the comparison is number-for-number, not hand-waving.

Here are those four steps as real, runnable code — the actual pipeline
[`examples/03_calibrate_tumvi_aprilgrid.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/examples/03_calibrate_tumvi_aprilgrid.py)
runs, trimmed of argument parsing and print formatting:

```python
import glob, os
from ds_msp.calib import AprilGridTarget, calibrate, detect_aprilgrid
from ds_msp.io.kalibr import load_kalibr_with_resolution
from ds_msp.models import KannalaBrandtModel

CALIB_DIR = "datasets/tumvi/dataset-calib-cam1_512_16/mav0/cam0/data"
CAMCHAIN = "datasets/tumvi/dataset-room1_512_16/dso/camchain.yaml"

# 1. detect -- multi-scale by default (scales=(1, 2, 3)), so peripheral tags the
#    fisheye shrinks below a single-scale detector's size gate are still recovered.
paths = sorted(glob.glob(os.path.join(CALIB_DIR, "*.png")))
detections = detect_aprilgrid(paths, family="t36h11", min_tags=6, refine=True)

# 2. correspond -- board geometry -> (X,Y,Z world) <-> (u,v pixel) pairs.
target = AprilGridTarget(tag_rows=6, tag_cols=6, tag_size=0.088, tag_spacing=0.3)
Xs, UVs, VIs = target.build_correspondences(detections, min_corners=8)

# 3. calibrate -- a generic seed, NOT the published answer; Cauchy keeps every
#    corner but down-weights outliers instead of discarding them.
seed = KannalaBrandtModel(fx=180.0, fy=180.0, cx=256.0, cy=256.0)
result = calibrate(seed, Xs, UVs, VIs, loss="cauchy", f_scale=0.5)
model = result["model"]

# 4. compare -- against TUM-VI's own published Kalibr camchain for this camera.
published, (w, h) = load_kalibr_with_resolution(CAMCHAIN, cam="cam0")
print(model)                    # KannalaBrandtModel(fx=191.007, fy=190.994, ...)
print(published)                # KannalaBrandtModel(fx=190.978, fy=190.973, ...)
print(f"median reprojection: {result['median_px']:.3f} px")   # 0.080 px
```

Every number quoted in this chapter comes straight from running this (the full script also
prints the per-parameter delta table and repeats the fit with Double Sphere — see below).

## The result

```
            fx        fy        cx        cy        k1        k2        k3        k4
published  190.978   190.973   254.932   256.897   0.00348   0.00072  -0.00205   0.00020
mine       191.019   191.006   254.949   256.860   0.00641  -0.00491   0.00171  -0.00059
|Δ|          0.041     0.033     0.018     0.037

median reprojection 0.080 px, inlier RMS 0.160 px — over all 14460 corners we
detected ourselves, none discarded.
```

![Reprojected corners vs detected corners](https://raw.githubusercontent.com/Munna-Manoj/DS-MSP/main/assets/learn/calibration_reprojection.gif)

*What "0.1 px" looks like: green = the corners we **detected**, red = where the **calibrated
model predicts** they should be. They sit on top of each other in every frame — the model
reproduces the measurements to a tenth of a pixel.*

- **Principal point to ~0.04 px.** `cx` lands within 0.018 px; `cy` within 0.037 px.
- **Focal length to ~0.02%.** `fx` agrees to 0.041 px — from a single camera, against a
  reference fit with the full Basalt/Kalibr pipeline (both cameras + IMU). The higher-order
  `k`'s differ more — they're weakly constrained and trade off against each other, which is
  why we judge the camera by reprojection error, not by staring at `k4`.
- **0.080 px median reprojection** is Kalibr-grade. That number is the proof the
  calibration is real. (We report median + inlier RMS rather than a single RMS: under a
  robust loss the plain all-corner RMS is inflated by the few outliers the loss correctly
  *ignored*, so it would understate the fit.)

> This near-exact agreement is recent: it took **multi-scale detection** to recover the
> wide-FOV corners that pin down focal length and distortion — see the deep-dive
> **[detecting every AprilGrid tag](robust_aprilgrid_detection.md)**. With single-scale
> detection the focal agreed only to ~0.7%; the periphery is where the constraint lives.

The library's flagship **Double Sphere** model fits the very same corners just as tightly
(0.080 px median — run the snippet above with `DoubleSphereModel(fx=180, fy=180, cx=256,
cy=256, xi=0.0, alpha=0.5)` as the seed to see it yourself). Its focal lands elsewhere
(`fx≈249.1, ξ≈0.30`) — not a bug: `fx` is model-relative (the true paraxial focal is
`fx_DS/(1+ξ) ≈ 191.1`, matching KB to 0.02%), and on a *planar* target DS additionally has a
focal↔(`xi`,`alpha`) gauge freedom — a different seed can land on a different `(fx, ξ)` pair
that computes to the *same* paraxial focal. A full proof that the DS and KB calibrations are
the *same camera* — and where they stop being — is in
**[are two models the same camera?](../explain/are_two_models_the_same_camera.md)**. Judge a model by
reprojection error, not by its raw focal.

## War-story: why the detector returned **zero** tags (and the real fix)

This is the part worth internalizing, because it's the kind of bug that wastes a day.

Our first attempts with OpenCV's `aruco` AprilTag detector **and** `pupil-apriltags` (the
standard AprilTag-3 binding) returned **0 tags on every frame** — even on frames where the
grid is large and sharp. The tags were obviously *there*. So what gives?

The detectors weren't broken: a *synthetic* `tag36h11` decoded fine. The difference is the
board. **Kalibr-style AprilGrids print each tag with a 2-cell-wide black border; stock
AprilTag-3 / aruco assume a 1-cell border.** The detector finds the tag's outer quad either
way, but then samples the 6×6 code bits at the wrong grid positions → every tag fails its
checksum → silently dropped. Zero detections, no error message.

The fix is a detector that matches the board: the pure-Python
[`aprilgrid`](https://github.com/powei-lin/aprilgrid) package, which *defaults to Kalibr's
2-cell border*. One line changes, and detection jumps from 0 to ~15–33 tags per frame.

But that's only half the battle — **off-centre boards still lose most of their tags** (a
fully-visible corner board can drop to 4 of 36), because the fisheye shrinks peripheral tags
below the detector's size gate. Those are the wide-FOV corners that constrain the distortion,
so losing them is exactly what kept the focal at ~0.7% instead of ~0.02%. The fix —
**multi-scale detection** plus two subpixel/pixel-centre subtleties that decide whether the
recovered corners *help or hurt* — is its own deep-dive:
**[detecting every AprilGrid tag](robust_aprilgrid_detection.md)**. It's why this capstone now
uses `scales=(1, 2, 3)` by default and lands on the published numbers near-exactly.

The lesson isn't "use library X." It's: **when a detector returns nothing on data you can
see is valid, suspect a layout/convention mismatch, not your eyes** — and confirm it the way
we did, by checking that the *same* detector succeeds on a synthetic tag and fails on a
2-border one. (`blackTagBorder = 2` is the default in Kalibr's own
`GridCalibrationTargetAprilgrid` — this is exactly how the official benchmark detected the
same board.)

Two more details turn a mediocre fit into a Kalibr-grade one, both things Kalibr also does:

- **Subpixel refinement.** The raw detector localizes corners to ~1 px; `cv2.cornerSubPix`
  sharpens them to the underlying intensity edge (median reprojection ~0.6 px → ~0.08 px) —
  done *at the scale each tag was detected*, per the [detection deep-dive](robust_aprilgrid_detection.md).
- **A robust loss, not a hard cut.** A few peripheral corners are still mis-localized (curved-lens
  tags where `cornerSubPix` grabs the wrong edge) and would drag a plain L2 fit. We don't
  *drop* them — we calibrate with a **Cauchy loss** that keeps every corner but down-weights
  large residuals continuously (`calibrate(..., loss="cauchy", f_scale=0.5)`). It beats hard
  rejection on focal accuracy *and* discards no data. This has its own
  **[learning-by-doing page](robust_losses_and_evaluation.md)** with the IRLS math and a
  runnable hard-reject-vs-robust comparison (`examples/04`) — including why you must score a
  robust fit by **median / inlier RMS**, never by RMS over all corners.

## How it's built (the codebase's layering, applied)

The capstone adds two small modules that mirror the rest of the library — terse core,
heavy dependencies isolated at the edge:

| Module | Depends on | Role |
|---|---|---|
| [`ds_msp/calib/targets.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/calib/targets.py) | numpy only | `AprilGridTarget`: board geometry + correspondence assembly. Pure, unit-tested without any image. |
| [`ds_msp/calib/detect.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/calib/detect.py) | OpenCV + `aprilgrid` (optional) | the *only* place that touches a tag backend; lazily imported so `import ds_msp` never needs it. |
| [`ds_msp/calib/bundle.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/calib/bundle.py) | scipy + the model contract | the model-agnostic LM optimizer; calibrates *any* `CameraModel`, with `loss=`/`f_scale=` for robust kernels. |

`detect_aprilgrid` is exposed through a lazy `__getattr__`, and `aprilgrid` lives in the
`[calib]` optional extra. Install the core lean; opt into the detector only when you
calibrate from real images. That's the "layer it, don't blend it" rule the whole repo
follows.

## Try it yourself
1. Run with `--stride 2` (more frames). Does `fx` tighten toward 191? More wide-angle
   coverage constrains focal length better.
2. Calibrate `cam1` instead of `cam0` and compare to that camera's published row in the
   camchain — the two stereo lenses differ slightly.
3. Turn off subpixel refinement (`refine=False` in `detect_aprilgrid`) and watch the median
   climb. Then switch the Cauchy loss back to plain `loss="linear"` and watch `fx` drift away
   from 191. Each guard earns its place by a number — `examples/04` makes that A/B explicit.

## Beyond one camera: the stereo rig

A rig has a second number that matters just as much as the intrinsics — the rigid transform
**between** the two cameras. Because TUM-VI's `cam0` and `cam1` are hardware-synced, each
instant sees the *same* board from both:

![Synchronized stereo views of the AprilGrid](https://raw.githubusercontent.com/Munna-Manoj/DS-MSP/main/assets/learn/stereo_pair.gif)

Calibrate each camera, compose the per-frame board poses, and you recover the stereo extrinsic
`T_cam1_cam0` — matching TUM-VI's published transform to **0.062° rotation and 0.25 mm
baseline** (~0.2%). That's the next step up: see the
[stereo extrinsics chapter](stereo_extrinsics_calibration.md) and run
`python examples/06_stereo_extrinsics_tumvi.py --stride 8`.

**Back to the path:** the theory chapters explain *why* each step here works —
[Ch.3](03_projection_validity.md) (validity & the >180° cone) is ready; Ch.4 (Jacobians) and
Ch.5 (the LM math) are coming (see [`../ROADMAP.md`](../ROADMAP.md)).
