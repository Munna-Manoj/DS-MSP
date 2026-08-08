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
> - At least **4 model-valid correspondences** for a coplanar target. A non-coplanar
>   full-sphere solve needs at least 6 (see [Common failures](#common-failures)).

## Why `cv2.solvePnP` fails here

`cv2.solvePnP` assumes a pinhole projection: a 3D point maps to a pixel through one focal
length and an optional polynomial distortion. A fisheye lens does not project that way.

Past ~90° the pinhole math has no valid pixel at all. Feed raw fisheye pixels to
`cv2.solvePnP` and it silently fits the wrong model, returning a pose that is degrees off.

`ds_msp` solves the right problem in three steps:

1. **Unproject** each pixel to a 3D unit bearing ray (a direction the lens sees) with the
   fisheye model, in closed form.
2. **Keep** every ray the model marks `valid`; a negative ray `z` is not discarded merely for
   lying past 90° off-axis.
3. **Select by geometry.** Forward-only observations use the established normalized-plane
   solve. If peripheral rays are present, non-coplanar targets use a bearing DLT (ADR-0018)
   and coplanar boards use a bearing homography (ADR-0019).

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
translation error: 0.00e+00 m
```

</div>

All 40 points are model-valid, and the recovered pose matches ground truth to
the float64 round-off floor.

/// note | Why is the error vanishingly small, not exactly zero?
`cv2.SOLVEPNP_ITERATIVE` is an iterative Levenberg-Marquardt refine, not a closed-form solve.
Here the data is noiseless and the model exactly invertible, so it converges essentially to
machine round-off rather than stopping early. The example normalizes display-only noise below
`1e-05°` and `1e-12 m` to zero; any larger error remains visible and fails its mirrored test.

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
translation error: 0.00e+00 m
```

</div>

Same scene, same solve, identical error — the wrapper is a thin shim over `cam.solve_pnp`,
not a different algorithm.

### Return-shape differences

The two entry points differ only in the shape of what comes back:

- `cam.solve_pnp` returns squeezed `(3,)` `rvec`/`tvec`.
- `ds_cv.solvePnP` returns `(3, 1)` column vectors, matching `cv2.solvePnP`'s native shape.
- Both return `(False, ...)` if fewer than 4 usable points remain after unprojection.

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
| `solve_pnp` returns `(False, None, None)` | Too few model-valid points: fewer than 4 for a coplanar target, or fewer than 6 for a non-coplanar full-sphere solve | Add valid correspondences and inspect `model.unproject(...)[1]` |
| Recovered pose flips or is unstable | Degenerate/near-degenerate point layout, incorrect 3D↔2D ordering, or too many outliers | Verify correspondence ordering; use `solve_pnp_ransac`; add spatially diverse points |

The solver drops any pixel that unprojects to an invalid ray before it solves.

/// note | This page's scene is forward-only — the wide-FOV case is handled too
This how-to's synthetic scene keeps every point in the forward hemisphere (`z > 0`), so it
uses the classic normalized-plane solve. When valid correspondences extend past 90°,
`solve_pnp` keeps them: a **non-coplanar** target uses the bearing DLT from ADR-0018, while a
**coplanar** board uses the bearing homography from ADR-0019.
///

If the selected solver lacks its minimum usable set (4 coplanar, 6 non-coplanar
full-sphere), it returns failure rather than guess.

## Next steps

- **Two views instead of one** — to recover the *relative* pose between two fisheye cameras
  from matched points (no known 3D), the ray-based cousin of this recipe is
  [Two-view geometry on rays](../learn/08_two_view_geometry_on_rays.md).
- **The geometry behind the filter** — the real (tilted half-space, not `z > 0`) validity
  boundary and how far it reaches: [Projection validity and FOV](../explain/projection_validity_and_fov.md).

**Recap:** on fisheye data, unproject pixels to rays, then solve PnP — on the normalized plane
for the forward-hemisphere case (this page's scene), or directly on bearing vectors for
peripheral data (ADR-0018 for non-coplanar targets, ADR-0019 for coplanar boards).
`cam.solve_pnp` does all of this and
recovers pose to the float64 round-off floor (displayed as zero on this synthetic scene).

---

*Source:*
[`ds_msp/ops/pose.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/ops/pose.py) ·
[`DoubleSphereCamera.solve_pnp`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/model.py) ·
[`ds_msp.cv.solvePnP`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/cv.py)
