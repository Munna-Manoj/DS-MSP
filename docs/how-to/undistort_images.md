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

## Undistort an image in three calls

Build the camera, ask for a new pinhole intrinsic matrix, then remap. `ds_msp.cv` mirrors the
[`cv2.fisheye`](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html) function
signatures, so it drops into existing OpenCV pipelines.

{* docs_src/how_to/undistort_images/estimate_and_undistort.py hl[20,21,28,29] *}

<!-- termynal -->
```
$ python3 -m docs_src.how_to.undistort_images.estimate_and_undistort
D = [0.183, 0.809]
(1080, 1920, 3)
284.56
```

You get a straight-line pinhole image: edges that curved in the fisheye are now straight.

The new focal length (`284.56 px`) is shorter than the original (`711.57 px`) because
`balance=0.0` zooms out to keep the widest possible view.

/// note
`estimateNewCameraMatrixForUndistortRectify` returns a single new matrix `K_new` with
`fx_new == fy_new` (it uses the average of `fx` and `fy` so nothing is stretched). Pass that
exact `K_new` to `undistortImage` via `Knew=` so the map and the matrix agree.
///

## Or use the object API

Hold a `DoubleSphereCamera` already? `undistort_image` does the same job in one call and hands
back the matrix it chose. Called with `K_new=None`, it builds a balanced matrix at `balance=0.5`.

`undistort_image` takes no `balance` argument — passing one raises `TypeError`. To pick a
different balance instead, build the matrix with
`estimateNewCameraMatrixForUndistortRectify(..., balance=...)` and pass it as `K_new=`:

{* docs_src/how_to/undistort_images/object_api.py hl[21,27,32:34] *}

<!-- termynal -->
```
$ python3 -m docs_src.how_to.undistort_images.object_api
(1080, 1920, 3)
426.84
TypeError: DoubleSphereCamera.undistort_image() got an unexpected keyword argument 'balance'
569.12
```

The two APIs are equivalent:

- OpenCV-style functions slot straight into an existing `cv2.fisheye` pipeline.
- The object method is the shorter path when you already hold the camera.

## Control the FOV-vs-border trade-off with `balance`

`balance` slides between two extremes of the same image:

- **Lower** `balance` keeps more of the scene, at the cost of black corners.
- **Higher** `balance` crops in until the borders are gone.

| `balance` | New focal `fx_new` | Black-border fraction | What you get |
| :-- | :-- | :-- | :-- |
| `0.0` | `284.56 px` | `0.075` | Widest <abbr title="Field of View">FOV</abbr> — the most scene, with black corners |
| `0.5` | `426.84 px` | `0.001` | Compromise (object-API default) |
| `1.0` | `569.12 px` | `0.000` | Tightest crop — no borders, least scene |

#### Measure the trade-off yourself

The "black-border fraction" is the share of output pixels that fell outside the fisheye's
coverage and were filled with black. Measure it directly instead of trusting the table:

{* docs_src/how_to/undistort_images/balance_tradeoff.py hl[27:34] *}

<!-- termynal -->
```
$ python3 -m docs_src.how_to.undistort_images.balance_tradeoff
balance=0.0  fx_new=284.56 px  black_fraction=0.075
balance=0.5  fx_new=426.84 px  black_fraction=0.001
balance=1.0  fx_new=569.12 px  black_fraction=0.000
fx_new(1.0) / fx_new(0.0) = 2.00
midpoint(0.0, 1.0) = 426.84  (balance=0.5 gives 426.84)
```

Going from `balance=0.0` to `balance=1.0` drops the black-border fraction from `0.075` to
`0.000`. Over the same range the focal length exactly doubles: `284.56 px` to `569.12 px`.

/// tip | `balance` interpolates linearly
`balance=0.5`'s `426.84 px` is exactly the midpoint of `284.56` and `569.12`. `balance`
interpolates the focal linearly, so you can predict where any intermediate value lands
without computing it.

You trade visible scene for a clean frame — there's no value of `balance` that gives you both.
///

## Troubleshooting: my undistorted image has black borders

Black borders are expected, not a bug. Tune `balance` to control them:

- Raise it toward `1.0` to crop the empty corners away.
- Lower it toward `0.0` to keep more scene.

For why the trade-off is geometric — no value removes the borders without losing FOV — see
[Projection validity and FOV](../explain/projection_validity_and_fov.md).

/// warning | A missing image fails loudly, not silently
If `cv2.imread` returns `None`, `undistortImage` errors with `AttributeError: 'NoneType'
object has no attribute 'shape'` before you ever get to `black_fraction`. The image path
didn't resolve — run from the repo root so `assets/test_image.jpg` is found.
///

## Try it yourself

Set `balance=0.3` in `estimateNewCameraMatrixForUndistortRectify`. Before you run it, predict:

- Will the black-border fraction be closer to `0.075` or to `0.000`?
- Will `fx_new` land between `284.56` and `569.12`?

Then wire the resulting `K_new` through `undistortImage(img, cam.K, cam.D, Knew=K_new)` (same
pattern as the setup snippet) and check your guess with `black_fraction(...)` on that result.

## Next steps

- **Why this works** — [Projection validity and FOV](../explain/projection_validity_and_fov.md):
  why a > 180° FOV can't fit a pinhole and black borders are geometric, not a defect.
- **The functions used here** — source on GitHub:
  [`ds_msp/cv.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/cv.py)
  (`estimateNewCameraMatrixForUndistortRectify`, `undistortImage`) and
  [`ds_msp/model.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/model.py)
  (`DoubleSphereCamera.undistort_image`, `compute_K_new`). The same algorithm, generalized to
  any `CameraModel`, lives in
  [`ds_msp/ops/undistort.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/ops/undistort.py)'s
  `Undistorter` class — use it when you're working with the newer contract-based model classes
  in `ds_msp.models` instead of `DoubleSphereCamera`.
- **Other recipes** — back to the [How-to guides](README.md).
