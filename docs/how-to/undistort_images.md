# Undistort a fisheye image

Turn a distorted fisheye frame into a flat pinhole image, and control the
field-of-view-vs-black-border trade-off with one knob.

This is a task recipe. For *why* a wide fisheye can't fit into a pinhole image without either
cropping or leaving black borders, see
[Projection validity and FOV](../explain/projection_validity_and_fov.md).

> **Prerequisites**
>
> - `ds_msp` installed, plus `opencv-python` and `numpy` (both come with it).
> - A calibrated camera — here a Double Sphere model with known intrinsics. If you still need
>   to calibrate, start from the [README usage](https://github.com/Munna-Manoj/DS-MSP#readme).
> - The snippets read `assets/test_image.jpg`, which ships in the repo. Run them from the repo
>   root, or point the path at your own fisheye frame.

## Undistort an image in three lines

Build the camera, ask for a new pinhole intrinsic matrix, then remap. `ds_msp.cv` mirrors the
[`cv2.fisheye`](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html) function
signatures, so it drops into existing OpenCV pipelines.

```python
import cv2
import numpy as np
from ds_msp import DoubleSphereCamera
import ds_msp.cv as ds_cv

# A calibrated Double Sphere camera (1920x1080 fisheye).
cam = DoubleSphereCamera(fx=711.57, fy=711.24, cx=949.18, cy=518.81,
                         xi=0.183, alpha=0.809, width=1920, height=1080)

img = cv2.imread("assets/test_image.jpg")     # (1080, 1920, 3) BGR
K, D = cam.K, cam.D                            # K: (3,3); D = [xi, alpha] = [0.183, 0.809]

# balance=0.0 -> widest FOV (keeps the most scene; leaves black borders)
K_new = ds_cv.estimateNewCameraMatrixForUndistortRectify(K, D, (1920, 1080), balance=0.0)
img_undist = ds_cv.undistortImage(img, K, D, Knew=K_new)   # (1080, 1920, 3)

cv2.imwrite("undistorted.jpg", img_undist)
print(img_undist.shape)          # -> (1080, 1920, 3)
print(round(K_new[0, 0], 2))     # -> 284.56  (new focal length, px)
```

You get a straight-line pinhole image: edges that curved in the fisheye are now straight. The
new focal length (`284.56 px`) is shorter than the original (`711.57 px`) because `balance=0.0`
zooms out to keep the widest possible view.

> **Note** — `estimateNewCameraMatrixForUndistortRectify` returns a single new matrix `K_new`
> with `fx_new == fy_new` (it uses the average of `fx` and `fy` so nothing is stretched).
> Pass that exact `K_new` to `undistortImage` via `Knew=` so the map and the matrix agree.

## Or use the object API

Hold a `DoubleSphereCamera` already? `undistort_image` does the same job in one call and hands
back the matrix it chose. Called with `K_new=None`, it builds a balanced matrix at `balance=0.5`.

```python
# continues from the setup above (cam, img)
img_undist, K_new = cam.undistort_image(img)   # K_new=None -> built at balance=0.5

print(img_undist.shape)          # -> (1080, 1920, 3)
print(round(K_new[0, 0], 2))     # -> 426.84  (between the widest and tightest focals)
```

`undistort_image` takes no `balance` argument — `cam.undistort_image(img, balance=0.3)` raises
`TypeError`. To pick a different balance, build the matrix with
`estimateNewCameraMatrixForUndistortRectify(..., balance=...)` and pass it as `K_new=`:

```python
# continues from the setup above (cam, img)
K_tight = ds_cv.estimateNewCameraMatrixForUndistortRectify(
    cam.K, cam.D, (1920, 1080), balance=1.0)
img_tight, _ = cam.undistort_image(img, K_new=K_tight)
print(round(K_tight[0, 0], 2))   # -> 569.12  (tightest crop, no borders)
```

The two APIs are equivalent. Use the OpenCV-style functions to slot into a `cv2.fisheye`
pipeline, or the object method when you already have the camera.

## Control the FOV-vs-border trade-off with `balance`

`balance` slides between two extremes of the same image. Lower keeps more of the scene at the
cost of black corners; higher crops in until the borders are gone.

| `balance` | New focal `fx_new` | Black-border fraction | What you get |
| :-- | :-- | :-- | :-- |
| `0.0` | `284.56 px` | `0.075` | Widest FOV — the most scene, with black corners |
| `0.5` | `426.84 px` | between the two | Compromise (object-API default) |
| `1.0` | `569.12 px` | `0.000` | Tightest crop — no borders, least scene |

Measure the trade-off yourself. The "black-border fraction" is the share of output pixels that
fell outside the fisheye's coverage and were filled with black:

```python
# continues from the setup above (cam, img)
def black_fraction(im):
    """Fraction of pixels that are pure black (outside the fisheye coverage)."""
    return float(np.mean(np.all(im == 0, axis=2)))

K0 = ds_cv.estimateNewCameraMatrixForUndistortRectify(cam.K, cam.D, (1920, 1080), balance=0.0)
K1 = ds_cv.estimateNewCameraMatrixForUndistortRectify(cam.K, cam.D, (1920, 1080), balance=1.0)
und0 = ds_cv.undistortImage(img, cam.K, cam.D, Knew=K0)
und1 = ds_cv.undistortImage(img, cam.K, cam.D, Knew=K1)

print(round(black_fraction(und0), 3))   # -> 0.075   (balance=0.0: visible borders)
print(round(black_fraction(und1), 3))   # -> 0.000   (balance=1.0: no borders)
```

Notice that going from `balance=0.0` to `balance=1.0` drops the black-border fraction from
`0.075` to `0.000` while the focal length roughly doubles (`284.56 px` -> `569.12 px`). You
trade visible scene for a clean frame.

## Troubleshooting: my undistorted image has black borders

Black borders are expected, not a bug — tune `balance` to control them. Raise it toward `1.0`
to crop the empty corners away, or lower it toward `0.0` to keep more scene. For why the
trade-off is geometric and no value removes the borders without losing FOV, see
[Projection validity and FOV](../explain/projection_validity_and_fov.md).

## Try it yourself

Set `balance=0.3` in `estimateNewCameraMatrixForUndistortRectify`. Before you run it, predict:
will the black-border fraction be closer to `0.075` or to `0.000`, and will `fx_new` land
between `284.56` and `569.12`? Then check your guess with `black_fraction(...)`.

If `cv2.imread` returns `None` and `black_fraction` then errors, the image path didn't
resolve — run from the repo root so `assets/test_image.jpg` is found.

## Next steps

- **Why this works** — [Projection validity and FOV](../explain/projection_validity_and_fov.md):
  why a > 180° FOV can't fit a pinhole and black borders are geometric, not a defect.
- **The functions used here** — source on GitHub:
  [`ds_msp/cv.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/cv.py)
  (`estimateNewCameraMatrixForUndistortRectify`, `undistortImage`) and
  [`ds_msp/ops/undistort.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/ops/undistort.py)
  (the model-agnostic `Undistorter` behind `cam.undistort_image`).
- **Other recipes** — back to the [How-to guides](README.md).
