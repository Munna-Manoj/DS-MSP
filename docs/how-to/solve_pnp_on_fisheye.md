# Solve PnP on raw fisheye points

Recover camera pose from 3D-to-2D correspondences on a wide-FOV fisheye image, where
`cv2.solvePnP` returns a wrong answer.

This is a task recipe. For *why* a pinhole PnP breaks down on fisheye points — the cheirality
(`z > 0`) and field-of-view geometry behind it — see
[Projection validity and FOV](../explain/projection_validity_and_fov.md).

> **Prerequisites**
>
> - `ds_msp` installed, plus `opencv-python` and `numpy` (both come with it).
> - A calibrated camera — here a Double Sphere model with known intrinsics. If you still need
>   to calibrate, start from the [README usage](https://github.com/Munna-Manoj/DS-MSP#readme).
> - At least **4 correspondences** whose 3D points land *in front of* the camera. Fewer than 4
>   front-facing points and the solve cannot run (see [Common failures](#common-failures)).

## Why `cv2.solvePnP` fails here

`cv2.solvePnP` assumes a pinhole projection: a 3D point maps to a pixel through one focal
length and an optional polynomial distortion. A fisheye lens does not project that way. Past
~90° the pinhole math has no valid pixel at all. Feed raw fisheye pixels to `cv2.solvePnP` and
it silently fits the wrong model, returning a pose that is degrees off.

`ds_msp` solves the right problem in three steps:

1. **Unproject** each pixel to a 3D unit bearing ray (a direction the lens sees) with the
   fisheye model, in closed form.
2. **Keep** only the valid, front-facing rays — those the model marks `valid` with ray
   component `z > 0`.
3. **Solve PnP in the normalized plane** (`x/z`, `y/z`) with an identity intrinsic. The rays
   are already metric, so no distortion model is needed downstream.

You get the same `(success, rvec, tvec)` triple as OpenCV, correct on fisheye data.

## The two entry points

Pick one of two equivalent calls. Use the object API when you already hold a camera:

> **API shape, not a runnable block.** The snippets here show the call signatures. The
> `points_3d` and `points_2d` arrays are filled in by the runnable end-to-end example in
> [Verify it on a synthetic scene](#verify-it-on-a-synthetic-scene) below — run that one.

```python
# points_3d: (N, 3) world points, metres ; points_2d: (N, 2) distorted fisheye pixels
success, rvec, tvec = cam.solve_pnp(points_3d, points_2d)
# rvec: (3,) Rodrigues rotation vector ; tvec: (3,) translation, metres
```

Or use the OpenCV-style functional wrapper. It takes `K` and `D`, so it drops into existing
`cv2.solvePnP` call sites:

```python
import ds_msp.cv as ds_cv

# cam.K is the pinhole matrix; cam.D = [xi, alpha] are the DS distortion coefficients.
success, rvec, tvec = ds_cv.solvePnP(points_3d, points_2d, cam.K, cam.D)
```

Both return `success=True` with squeezed `rvec`/`tvec`. Both return `(False, ...)` if fewer
than 4 points survive the front-facing filter.

## Verify it on a synthetic scene

Run this block end to end; the contrast section below continues from it. It generates a known
pose, projects 3D points through the fisheye model to make 2D correspondences, asks `solve_pnp`
to recover the pose, then measures the error.

```python
import numpy as np
import cv2
from ds_msp import DoubleSphereCamera

cam = DoubleSphereCamera(fx=711.57, fy=711.24, cx=949.18, cy=518.81,
                         xi=0.183, alpha=0.809, width=1920, height=1080)

# 1. A ground-truth pose (what we want to recover).
rvec_gt = np.array([0.05, -0.10, 0.02])      # Rodrigues vector, rad
tvec_gt = np.array([0.30, -0.20, 1.00])      # translation, metres
R_gt, _ = cv2.Rodrigues(rvec_gt)

# 2. 40 world points spread in front of the camera.
rng = np.random.default_rng(0)
points_3d = rng.uniform([-2, -2, 4], [2, 2, 8], size=(40, 3))   # (40, 3) metres

# 3. Project them through the fisheye to get 2D correspondences.
P_cam = (R_gt @ points_3d.T + tvec_gt[:, None]).T               # (40, 3) camera frame
uv, valid = cam.project(P_cam)                                  # uv: (40, 2) pixels
points_2d = uv[valid]
points_3d = points_3d[valid]

# 4. Recover the pose from the 3D<->2D correspondences.
success, rvec, tvec = cam.solve_pnp(points_3d, points_2d)
print(success, len(points_3d))                                 # -> True 40

# 5. Measure the error against ground truth.
R, _ = cv2.Rodrigues(rvec)
rot_err_deg = np.degrees(np.arccos(np.clip((np.trace(R @ R_gt.T) - 1) / 2, -1, 1)))
t_err_m = np.linalg.norm(tvec - tvec_gt)
print(f"rotation error: {rot_err_deg:.2e} deg")                # -> ~1.21e-06 deg
print(f"translation error: {t_err_m:.2e} m")                   # -> ~2.72e-16 m
```

Notice the error sizes: the recovered pose matches ground truth to `~1e-06°` and `~3e-16 m`.
That is numerical noise, not residual model error. The solve is exact on clean correspondences.

> **Note** — the numbers above are from a clean synthetic run (seed `0`). On real detections
> with pixel noise, expect a sub-pixel reprojection RMS, not machine epsilon.

### Contrast: pinhole PnP on the same points

Hand the *same* fisheye pixels to `cv2.solvePnP` with the camera's pinhole `K`. It fits the
wrong model:

```python
# Continues from the "Verify it on a synthetic scene" block above
# (cam, points_3d, points_2d, R_gt, tvec_gt).
ok, rv, tv = cv2.solvePnP(points_3d.astype(np.float64),
                          points_2d.astype(np.float64),
                          cam.K, np.zeros(5))                   # pinhole assumption
R_bad, _ = cv2.Rodrigues(rv)
bad_rot = np.degrees(np.arccos(np.clip((np.trace(R_bad @ R_gt.T) - 1) / 2, -1, 1)))
print(f"cv2 rotation error: {bad_rot:.2f} deg")                # -> ~0.57 deg
print(f"cv2 translation error: {np.linalg.norm(tv.squeeze() - tvec_gt):.2f} m")  # -> ~1.37 m
```

A `0.57°` rotation and `1.37 m` translation error from the *same* data — that gap is the
fisheye distortion that `cv2.solvePnP` cannot model.

## Common failures

| Symptom | Cause | Fix |
| :-- | :-- | :-- |
| Pose is degrees off, no error raised | Used `cv2.solvePnP` with pinhole `K` on raw fisheye pixels | Switch to `cam.solve_pnp` / `ds_cv.solvePnP` |
| `solve_pnp` returns `(False, None, None)` | Fewer than 4 points are in front of the camera after unprojection | Add correspondences, or check that your 3D points are actually in view |
| Recovered pose flips sign | Points behind the camera (`z <= 0`) leaked in | The `z > 1e-6` ray check filters these. Confirm your ground-truth pose puts every point in front: `((R_gt @ P.T).T + t)[:, 2] > 0` should be all `True` |

The solver drops any pixel that unprojects to an invalid or behind-camera ray (`z <= 1e-6`)
before it solves. If that leaves fewer than 4 points, it returns `(False, None, None)` rather
than guess.

## Next steps

- **Two views instead of one** — to recover the *relative* pose between two fisheye cameras
  from matched points (no known 3D), the ray-based cousin of this recipe is
  [Two-view geometry on rays](../learn/08_two_view_geometry_on_rays.md).
- **The geometry behind the filter** — why `z <= 0` rays are invalid and how FOV bounds the
  valid set: [Projection validity and FOV](../explain/projection_validity_and_fov.md).

**Recap:** on fisheye data, unproject pixels to rays, keep the front-facing valid ones, then
solve PnP in the normalized plane — `cam.solve_pnp` does all three and recovers pose to
numerical precision.

---

*Source:*
[`ds_msp/ops/pose.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/ops/pose.py) ·
[`DoubleSphereCamera.solve_pnp`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/model.py) ·
[`ds_msp.cv.solvePnP`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/cv.py)
