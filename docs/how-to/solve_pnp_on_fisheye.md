# Solve PnP on raw fisheye points

Recover camera pose from 3D-to-2D correspondences on a wide-<abbr title="Field Of View — the angular extent of the scene a lens captures.">FOV</abbr>
fisheye image, where `cv2.solvePnP` returns a wrong answer.

This is a task recipe. A naive pinhole
<abbr title="Perspective-n-Point — solving for camera pose from n known 3D points and their 2D projections.">PnP</abbr>
only ever considers points with `z > 0`.

It has no concept of the wider region a fisheye actually sees.

For *why* the fisheye model's real validity boundary is a tilted half-space rather than
`z > 0`, and how much further it reaches, see
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
length and an optional polynomial distortion. A fisheye lens does not project that way.

Past ~90° the pinhole math has no valid pixel at all. Feed raw fisheye pixels to
`cv2.solvePnP` and it silently fits the wrong model, returning a pose that is degrees off.

`ds_msp` solves the right problem in three steps:

1. **Unproject** each pixel to a 3D unit bearing ray (a direction the lens sees) with the
   fisheye model, in closed form.
2. **Keep** only the usable rays — those the model marks `valid` (unprojection succeeded)
   *and* whose ray component `z > 0` (in front of the camera, required by the next step).
3. **Solve PnP in the normalized plane** (`x/z`, `y/z`) with an identity intrinsic. The rays
   are already metric, so no distortion model is needed downstream.

You get the same `(success, rvec, tvec)` triple as OpenCV, correct on fisheye data.

## The two entry points

Two equivalent calls do the same solve. Pick the object API when you already hold a camera,
or the functional wrapper when you're dropping this into existing `cv2.solvePnP` call sites.

Both examples below build the same synthetic scene: a known ground-truth pose, 40 world
points projected through a Double Sphere model into fisheye pixels, then recovered.

### Object API: `cam.solve_pnp`

{* docs_src/how_to/solve_pnp_on_fisheye/solve_pnp_basic.py hl[33,34] *}

<div class="termy">

```console
$ python -m docs_src.how_to.solve_pnp_on_fisheye.solve_pnp_basic
True 40
rotation error: 0.00e+00 deg
translation error: 7.77e-16 m
```

</div>

All 40 points survive the front-facing filter, and the recovered pose matches ground truth to
the float64 round-off floor.

/// note | Why is the error exactly zero, not just small?
`cv2.SOLVEPNP_ITERATIVE` is an iterative Levenberg-Marquardt refine, not a closed-form solve.
Here the data is noiseless and the model exactly invertible, so it converges all the way to
machine round-off (`0.00e+00°`, `7.77e-16 m`) rather than stopping early.

On real detections with pixel noise, expect a sub-pixel reprojection RMS instead — this
measurement is a correctness check, not a noise-robustness one.
///

### Functional wrapper: `ds_cv.solvePnP`

`ds_cv.solvePnP` takes `K` and `D` instead of a camera object, so it drops into an existing
`cv2.solvePnP` call site with minimal changes:

{* docs_src/how_to/solve_pnp_on_fisheye/solve_pnp_cv_wrapper.py hl[31,32] *}

<div class="termy">

```console
$ python -m docs_src.how_to.solve_pnp_on_fisheye.solve_pnp_cv_wrapper
True (3, 1) (3, 1)
rotation error: 0.00e+00 deg
translation error: 7.77e-16 m
```

</div>

Same scene, same solve, identical error — the wrapper is a thin shim over `cam.solve_pnp`,
not a different algorithm.

### Return-shape differences

The two entry points differ only in the shape of what comes back:

- `cam.solve_pnp` returns squeezed `(3,)` `rvec`/`tvec`.
- `ds_cv.solvePnP` returns `(3, 1)` column vectors, matching `cv2.solvePnP`'s native shape.
- Both return `(False, ...)` if fewer than 4 points survive the front-facing filter.

## Contrast: pinhole PnP on the same points

Hand the *same* fisheye pixels to `cv2.solvePnP` with the camera's pinhole `K`. It fits the
wrong model:

{* docs_src/how_to/solve_pnp_on_fisheye/pinhole_contrast.py hl[29,31] *}

<div class="termy">

```console
$ python -m docs_src.how_to.solve_pnp_on_fisheye.pinhole_contrast
cv2 rotation error: 0.57 deg
cv2 translation error: 1.37 m
```

</div>

A `0.57°` rotation and `1.37 m` translation error from the *same* data — that gap is the
fisheye distortion that `cv2.solvePnP` cannot model.

## Common failures

Three symptoms account for nearly every PnP failure on fisheye data:

| Symptom | Cause | Fix |
| :-- | :-- | :-- |
| Pose is degrees off, no error raised | Used `cv2.solvePnP` with pinhole `K` on raw fisheye pixels | Switch to `cam.solve_pnp` / `ds_cv.solvePnP` |
| `solve_pnp` returns `(False, None, None)` | Fewer than 4 points are in front of the camera after unprojection | Add correspondences, or check that your 3D points are actually in view |
| Recovered pose flips sign | Points behind the camera (`z <= 1e-6`) leaked in | The `z > 1e-6` ray check filters these. Confirm your ground-truth pose puts every point in front: `((R_gt @ P.T).T + t)[:, 2] > 0` should be all `True` |

The solver drops any pixel that unprojects to an invalid or behind-camera ray (`z <= 1e-6`)
before it solves.

If that leaves fewer than 4 points, it returns `(False, None, None)` rather than guess.

## Next steps

- **Two views instead of one** — to recover the *relative* pose between two fisheye cameras
  from matched points (no known 3D), the ray-based cousin of this recipe is
  [Two-view geometry on rays](../learn/08_two_view_geometry_on_rays.md).
- **The geometry behind the filter** — the real (tilted half-space, not `z > 0`) validity
  boundary and how far it reaches: [Projection validity and FOV](../explain/projection_validity_and_fov.md).

**Recap:** on fisheye data, unproject pixels to rays, keep the front-facing valid ones, then
solve PnP in the normalized plane — `cam.solve_pnp` does all three and recovers pose to the
float64 round-off floor (`0.00e+00°` rotation error, on this synthetic scene).

---

*Source:*
[`ds_msp/ops/pose.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/ops/pose.py) ·
[`DoubleSphereCamera.solve_pnp`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/model.py) ·
[`ds_msp.cv.solvePnP`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/cv.py)
