# DS-MSP

**Double Sphere & Multi-model Spherical-camera Platform** — one interface to calibrate, convert, and reason about every wide-FOV camera model, from a single lens to a full rig.

[![PyPI](https://img.shields.io/pypi/v/ds-msp)](https://pypi.org/project/ds-msp/)
[![CI](https://github.com/Munna-Manoj/DS-MSP/actions/workflows/ci.yml/badge.svg)](https://github.com/Munna-Manoj/DS-MSP/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/ds-msp/)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Munna-Manoj/DS-MSP/blob/main/LICENSE)
![Tests](https://img.shields.io/badge/tests-608%20passing-brightgreen)
[![Live demo](https://img.shields.io/badge/%E2%96%B6%20live%20demo-interactive%20studio-6e8bff)](https://munna-manoj.github.io/DS-MSP/studio/)

A clean, tested, OpenCV-compatible Python platform for wide-FOV (fisheye / omnidirectional) cameras.

One interface covers calibration, model conversion, 3D geometry, multi-camera rigs, and hardware export — so you can swap camera models in a single line and convert between them without re-shooting images.

![Double Sphere image formation — 3D points traced through two spheres onto the z=1 plane and the physical sensor](https://raw.githubusercontent.com/Munna-Manoj/DS-MSP/main/assets/learn/double_sphere_pipeline.gif)

> *The namesake **Double Sphere** model in motion: each 3D point is traced through two spheres and projected to the fisheye image. Cross-checked against the library itself (`std = 2e-16`).* **[Drive it live →](https://munna-manoj.github.io/DS-MSP/studio/)**

---

## Install

```bash
pip install ds-msp                 # core library
pip install "ds-msp[calib]"        # + AprilGrid detector
```

---

## What do you want to do?

| Task | Recipe |
|---|---|
| Project & unproject 3D points | [→ Project & unproject](#project--unproject) |
| Rectify a fisheye to a pinhole image (+ new `K`) | [→ Pinhole image & new K](#pinhole-image--new-k) |
| Estimate 2D/3D pose (PnP & two-view) | [→ 2D/3D pose](#2d3d-pose) |
| Calibrate any camera from scratch | [→ Calibrate from scratch](#calibrate-from-scratch) |
| Convert one model to another (no images, no data) | [→ Convert between models](#convert-between-models-no-images-no-data) |
| Calibrate a multi-camera rig (any FOV) for extrinsics | [→ Multi-camera rig](#multi-camera-rig-extrinsics-any-fov) |
| Run stereo depth on raw fisheye | [→ Stereo depth](#stereo-depth) |
| Export to TI hardware | [→ Hardware LDC export](#hardware-ldc-export) |
| Pick the right model | [→ Choosing a model](#choosing-a-model) |
| Learn the geometry | [→ Learn](#learn) |

---

## Recipes

Every recipe below is a copy-paste starting point. Each links to a fuller guide; the code uses the real API.

### Project & unproject

Build any model from its intrinsics, then `project` 3D points to pixels and `unproject` pixels back to unit bearing rays — exact inverses.

Every model also exposes analytic point and parameter Jacobians (no autodiff, no finite differences), so these drop straight into bundle adjustment.

```python
import numpy as np
from ds_msp import DoubleSphereCamera

cam = DoubleSphereCamera(fx=711.57, fy=711.24, cx=949.18, cy=518.81,
                         xi=0.183, alpha=0.809, width=1920, height=1080)

pts_3d = np.array([[0., 0., 1.], [1., 1., 2.]])
px,   ok = cam.project(pts_3d)     # (N,2) pixels + validity mask
rays, ok = cam.unproject(px)       # (N,3) unit rays (exact inverse of project)
```

Round-trips to ~1e-13 px. *The recipes below reuse this `cam`.* → [Multi-model guide](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/MULTI_MODEL.md)

### Pinhole image & new K

Rectify a raw fisheye frame to an undistorted **pinhole image** and get the **new pinhole intrinsics** `K_new` for it.

The `balance` knob trades field of view against crop:
- `0.0` keeps the widest FOV (more black border).
- `1.0` is the tightest crop (no border).

```python
import cv2, ds_msp.cv as ds_cv

img = cv2.imread("frame.png")      # your raw fisheye frame

# object API — rectified pinhole image AND its new pinhole intrinsics
img_undist, K_new = cam.undistort_image(img)

# OpenCV-style equivalent, with explicit size + balance (0 = widest FOV … 1 = tightest crop)
K_new      = ds_cv.estimateNewCameraMatrixForUndistortRectify(cam.K, cam.D, (1920, 1080), balance=0.0)
img_undist = ds_cv.undistortImage(img, cam.K, cam.D, Knew=K_new)
```

![Fisheye rectified to a pinhole view, sweeping balance from widest-FOV to tightest-crop](https://raw.githubusercontent.com/Munna-Manoj/DS-MSP/main/assets/undistort_demo.gif)

Inverse-projection error across all undistort modes is < 0.00003 px. → [Multi-model guide](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/MULTI_MODEL.md)

### 2D/3D pose

`cam.solve_pnp` is a **model-agnostic convenience, not a new algorithm**: it unprojects your pixels through whichever of the 7 models you use, then calls the standard `cv2.solvePnP` on the undistorted points — literally `unproject → cv2.solvePnP(K=I, dist=None)`.

`cv2.solvePnP` itself is correct here. OpenCV already ships `cv2.fisheye.solvePnP` for **Kannala-Brandt** lenses — the value of `cam.solve_pnp` is that the *same one call* also covers DS / UCM / EUCM / OCam / DS+.

> **Warning**
> `cv2.solvePnP` only goes wrong if you hand it **raw** fisheye pixels with a pinhole `K` — the wrong distortion model for a wide-FOV lens. `cam.solve_pnp` avoids that by unprojecting first.

For noisy matches, `cam.solve_pnp_ransac` adds RANSAC outlier rejection and returns an inlier mask.

> **Note**
> `solve_pnp` and `solve_pnp_ransac` keep only front-facing (<90°) correspondences — fine for calibration boards. The two-view snippet below has no such limit: it works directly on bearing rays.

```python
from ds_msp import relative_pose, triangulate_rays

# 3D pose from 2D–3D correspondences (PnP), straight on raw fisheye
points_3d = ...   # (N,3) world points (from your board / detector)
points_2d = ...   # (N,2) matching fisheye pixels
ok, rvec, tvec = cam.solve_pnp(points_3d, points_2d)

# noisy matches? RANSAC variant rejects gross outliers + returns an inlier mask
ok, rvec, tvec, inliers = cam.solve_pnp_ransac(points_3d, points_2d)

# 2D–2D → relative pose + triangulated 3D points, on bearing rays
px1 = ...         # (N,2) pixels in image 1
px2 = ...         # (N,2) matching pixels in image 2
f1, _ = cam.unproject(px1)
f2, _ = cam.unproject(px2)
R, t         = relative_pose(f1, f2)           # camera-2 pose relative to camera-1
pts_3d, _, _ = triangulate_rays(f1, f2, R, t)  # (N,3) triangulated points
```

Measured PnP + reprojection RMS on bundled real images: 0.43 px / 0.85 px. → [Two-view geometry](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/explain/two_view_geometry.md)

### Calibrate from scratch

`calibrate()` is **model-agnostic** — the *same* call fits any of the 7 models from a generic seed; only the seed class changes.

It runs manifold Levenberg–Marquardt with analytic Jacobians and an optional robust loss (`huber` / `cauchy`); per-view poses are re-seeded internally, so a rough focal is fine.

```python
from ds_msp.calib import calibrate
from ds_msp.models import KannalaBrandtModel        # or DoubleSphereModel, DSPlusModel, EUCMModel, …

# your detected correspondences, per frame (e.g. from cv2 / AprilGrid):
X_world    = ...   # list of (M,3) board points
keypoints  = ...   # list of (M,2) detected pixels
visibility = ...   # list of (M,) bool masks
w, h = 1920, 1080  # image size

seed = KannalaBrandtModel(fx=0.5 * w, fy=0.5 * w, cx=0.5 * w, cy=0.5 * h)   # generic seed
result = calibrate(seed, X_world, keypoints, visibility, loss="huber", f_scale=1.0)
print(result["model"], result["rms_px"])           # calibrated model + reprojection RMS
```

On the capstone (real TUM-VI fisheye + AprilGrid) this matches the *published* intrinsics to **~0.02% focal** at **0.08 px median**. → [Calibration capstone](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/learn/capstone_calibrating_a_real_camera.md)

**Just have a folder of images, no correspondences yet?** `ds-msp-calibrate` is a real console
command from `pip install ds-msp` alone — detection (checkerboard / ChArUco / AprilGrid) +
`calibrate()` above, config-driven, same pattern as the rig CLI below:

```bash
ds-msp-calibrate --init-config calib_config.yml   # write a starter config
ds-msp-calibrate --config calib_config.yml        # detect -> bundle-adjust -> report -> save
```

Prints the full reprojection-error distribution (not just one RMS number) and a PASS/WARN/FAIL
verdict; `ds_msp.calib.load_camera("camchain.yaml")` loads the result straight back into a
ready `CameraModel` instance. → [Single-camera calibration guide](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/CALIBRATE_GUIDE.md)

### Convert between models (no images, no data)

Translate a calibration between **any** two models with no images and no recalibration: DS-MSP samples the image grid, unprojects through the source model, builds a linear seed for the target, then refines on pixel reprojection.

The result agrees to sub-pixel accuracy; the design follows Fisheye-Calib-Adapter (see [Credits](#credits)).

```python
from ds_msp import convert
from ds_msp.models import DoubleSphereModel, KannalaBrandtModel

# convert() takes a model (not a Camera); same DS intrinsics as `cam` above
ds = DoubleSphereModel(fx=711.57, fy=711.24, cx=949.18, cy=518.81, xi=0.183, alpha=0.809)
kb, report = convert(ds, KannalaBrandtModel, width=1920, height=1080)   # DS → OpenCV fisheye
print(report["rms_px"])            # sub-pixel agreement; no images, no recalibration
```

→ [Multi-model guide](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/MULTI_MODEL.md)

### Multi-camera rig extrinsics (any FOV)

> **Note**
> Shipped in `pip install ds-msp` (`ds_msp.rig`) — validated on a real 8-camera rig and the MC-Calib Blender datasets (extrinsics within ~0.16% of ground truth).
>
> `pip install ds-msp` alone gives you the `ds-msp-calibrate-rig` command; no repo clone needed. A git clone can use `python scripts/calibrate_rig.py` instead — same CLI either way.

A drop-in for **MC-Calib**: same `calib_param.yml` config schema, same output files (`calibrated_cameras_data.yml`, `calibrated_objects_data.yml`, …), interoperable both directions.

It calibrates the **extrinsics** (where each camera sits relative to the others) plus per-camera intrinsics from a folder of ChArUco images.

It adds one extension over MC-Calib: a **different camera model per camera**, so one bench can mix `kb` fisheye, `radtan` pinhole, and `dsplus` at any FOV.

```bash
pip install ds-msp
ds-msp-calibrate-rig --init-config calib_param.yml   # write a starter config
ds-msp-calibrate-rig --config calib_param.yml        # detect → reconstruct → bundle-adjust → MC-Calib output
```

#### Config options

- Point it at a `camera_parameters` YAML with `fix_intrinsic=true` to **hold intrinsics fixed** and solve extrinsics only.
- Point it at the same YAML without that flag to refine intrinsics too.
- Omit the YAML to calibrate everything from scratch.
- If the chosen model differs from the one in the YAML, the same lens is carried into it via `convert()` plus a warning.

Detected corners are saved to `detected_keypoints_data.yml` and reused via `keypoints_path`, so re-calibrating with different models takes seconds.

#### Measured accuracy

On the MC-Calib Blender datasets, extrinsics land within **0.16% of ground truth** at sub-pixel reprojection, matching MC-Calib's own published per-camera RMS (0.92–1.07×). → [Rig guide](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/RIG_CALIBRATION_GUIDE.md) · [Blender evaluation](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/RIG_BLENDER_EVALUATION.md)

### Stereo depth

Estimate metric depth directly on raw fisheye with a sphere sweep — no rectification to pinhole, so the full wide FOV is kept.

Give the reference camera + image and the source views (each as `(camera, image, R, t)`), sweep a set of inverse-depth planes, and back-project to a point cloud.

```python
from ds_msp.stereo import sphere_sweep, sweep_to_points, inverse_depth_samples

ref_cam, ref_img = cam, ...        # reference camera + its image  (... = your image)
sources = ...                      # list of (camera, image, R, t) for the other posed views
depths  = inverse_depth_samples(near=0.5, far=20.0, n=64)

depth, cost, valid = sphere_sweep(ref_cam, ref_img, sources, depths)
points = sweep_to_points(ref_cam, depth, valid)        # (N,3) metric point cloud
```

→ [Stereo & extrinsics](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/learn/stereo_extrinsics_calibration.md)

### Hardware LDC export

Generate a TI Jacinto LDC displacement-mesh lookup table (J7 / TDA4) directly from a calibrated model:

```python
from ds_msp.ldc import TI_LDC_MeshGenerator

res = TI_LDC_MeshGenerator(cam).generate_mesh_and_intrinsics(1920, 1080, downsample_factor=4, balance=0.5)
mesh_lut, K_new = res["mesh_lut"], res["K_new"]
```

---

## Choosing a model

All 7 models live behind one `CameraModel` contract with analytic Jacobians, `convert`, and Kalibr + MC-Calib I/O.

| model | params | role |
|---|---|---|
| `radtan` | 9 | OpenCV pinhole (Brown-Conrady); mild distortion, not a fisheye model |
| `kb` | 8 | OpenCV fisheye (Kannala-Brandt); dependable baseline, iterative unproject |
| `ucm` | 5 | Unified Camera Model |
| `eucm` | 6 | Enhanced UCM |
| `ds` | 6 | **Double Sphere** (Usenko 2018); exact >180° half-space validity |
| `dsplus` | 9 | **DS+** — best accuracy, closed-form-invertible, differentiable |
| `ocam` | 10 | OCamCalib (Scaramuzza) |

#### Rule of thumb (directional, not a guarantee)

- **≥170° FOV** → start with Double Sphere (the cheapest closed form that fits).
- **~120–150°** → the spherical family (UCM/EUCM/DS) can *collapse* at these angles; if it does, use KB, or DS+ when you need a closed-form inverse.
- **Always confirm with the median reprojection error**, not the model class.

#### About DS+

- DS+ is the best-accuracy model tested and the only closed-form-invertible model that matches KB on hard lenses (~21% win on a demanding 158° full-FOV lens: 0.224 vs KB 0.285 px). The trade-off: its unproject is **~2× slower than KB** (Ferrari quartic). Pick it when you need a differentiable / closed-form unproject with deterministic latency — not for raw throughput.
- KB stays the right default when unproject throughput rules and an iterative inverse is acceptable.

→ [Choosing a model](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/explain/choosing_a_camera_model.md) · [DS+/EUCM+/KB case study (historical)](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/explain/case_study_eucmplus_dsplus_kb.md)

---

## Verify it

Every claim is backed by a number you can reproduce — **608 tests passing**, CI green.

- Inverse projection (all undistort modes): mean error **< 0.00003 px**.
- 3D reconstruction of checkerboard corners: **0.835 mm** mean; recovered **20.00 cm** square (target 20.00).
- PnP + reprojection RMS on real test images: **0.43 px / 0.85 px**.
- KB / RadTan vs OpenCV: agree to **~1e-13**.
- Bundled DS calibration (both test views) converges to fx≈733.9, fy≈733.0, cx≈951.7, cy≈517.9, xi≈0.230, alpha≈0.817 at **0.57 px** RMS. The reference intrinsics for this bundled data are fx≈711.6, xi≈0.183.

> **Note**
> A single-view or two-view planar fit doesn't fully pin the DS focal/xi/alpha gauge (see [are two models the same camera?](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/explain/are_two_models_the_same_camera.md)). Judge the fit above by reprojection RMS, not by how closely the raw parameters match the reference.

```bash
pytest
bash verify_all.sh
python benchmarks/benchmark.py
```

---

## Learn

DS-MSP doubles as a **runnable curriculum** in wide-FOV geometry. Every chapter in [`docs/learn/`](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/learn/README.md) prints a verifiable number — from the projection models through real calibration, model conversion, and the DS+/KB case study.

That case study openly documents a hypothesis the authors had to retract. Start at the [Learn index](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/learn/README.md).

---

## Credits

DS-MSP stands on prior work, and credit stays prominent:

- **[Fisheye-Calib-Adapter](https://github.com/eowjd0512/fisheye-calib-adapter)** — Sangjun Lee, arXiv:2407.12405. The conversion design and model set follow it.
- **[MC-Calib](https://github.com/rameau-fr/MC-Calib)** — Rameau, Park, Bailo, Kweon, CVIU 2022. The rig pipeline follows MC-Calib's design (config schema, result format, workflow); files interoperate both ways. **Full credit to the MC-Calib authors.**
- **Double Sphere** — Usenko, Demmel, Cremers, 3DV 2018 (arXiv:1807.08957; basalt-headers).
- **Kannala-Brandt** (2006), **RadTan / Brown-Conrady** (1966), **OCamCalib** (Scaramuzza), **EUCM** (Khomutenko et al. 2016), **UCM** (Geyer & Daniilidis / Mei & Rives).
- **Kalibr** (Furgale et al.), **dscamera**, and **Muhammadjon Boboev** (the original Python DS calibration this project grew from).
- **[Termynal](https://github.com/fastapi/typer)** — Ines Montani, MIT. Powers the animated terminal blocks in these docs; vendored from Typer, whose docs also inspired this site's style.

---

## License

MIT-licensed (see [LICENSING.md](https://github.com/Munna-Manoj/DS-MSP/blob/main/LICENSING.md)):
- **MIT** for the whole project — the generic library, the DS+ model, and the robust calibration/conversion engine.

SPDX: `MIT`. Please cite via [CITATION.cff](https://github.com/Munna-Manoj/DS-MSP/blob/main/CITATION.cff).
