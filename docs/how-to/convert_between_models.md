# Convert a calibration between camera models

Translate an already-calibrated camera from one model to another **without images
and without recalibrating** — e.g. take a Double Sphere calibration and hand a
downstream tool an OpenCV-ready Kannala-Brandt model instead.

> **Prerequisites**
> - DS-MSP installed (`pip install ds-msp`).
> - A *source* model you have already calibrated. This recipe uses the bundled
>   sample calibration so it runs with no files; [load your own](#load-your-own-calibration) at the end.
> - You want a different model's parameters, not new measurements. Conversion fits
>   the target model to reproduce the source's geometry; it adds no new information.

## Convert in one call

`convert(source, target_class, width=..., height=...)` returns the fitted target
model and a quality report. Pass the source *instance* and the target *class*:

```python
from ds_msp import DoubleSphereModel, KannalaBrandtModel, convert

ds = DoubleSphereModel.sample()        # the bundled DS calibration (no file needed)

kb, report = convert(ds, KannalaBrandtModel, width=1920, height=1080)
print(report["rms_px"])                # -> 0.00021  (pixels of disagreement, RMS)
print(report["converged"])             # -> True
```

`kb` is a `KannalaBrandtModel` whose `project`/`unproject` reproduce the original
Double Sphere camera. `report["rms_px"]` is the RMS pixel disagreement between the
two models, sampled across the image. Here it reads **0.00021 px**, far below one
pixel, so the conversion is effectively lossless.

> **Notice:** you pass `KannalaBrandtModel` (the class), not an instance. The
> converter constructs and fits the target itself.

## Read the quality report

The report tells you whether the conversion is faithful and over what field of
view. Always check it. Some conversions are lossy, and the report is how you catch
that — a conversion never fails silently.

```python
# continues from the setup above
print(report["median_px"])         # 0.00012   typical pixel error (half the samples below this)
print(report["max_px"])            # 0.00099   worst-case pixel error
print(report["fov_covered_deg"])   # 179.9     FOV the fit actually covered
print(report["source_model"])      # 'ds'
print(report["target_model"])      # 'kb'
```

| Field | Meaning |
| :--- | :--- |
| `rms_px` | RMS pixel disagreement between source and target across the image. Your headline accuracy number. |
| `max_px` | Worst single-sample error — catches edge-of-image blow-ups. |
| `median_px` | Median pixel error. |
| `fov_covered_deg` | Full-angle FOV the fit covered. A narrow target shrinks this. |
| `converged` | `True` if the Levenberg-Marquardt refine converged. |
| `source_model` / `target_model` | The two models' names. |

A conversion is trustworthy when `converged` is `True` **and** `rms_px` is small
relative to a pixel over the `fov_covered_deg` you care about.

## Pick a target model

Any model in the library is a valid target. How well it reproduces the source
depends on how expressive the target is. These numbers are from converting the
bundled Double Sphere calibration (1920×1080):

| Target model | Class | `rms_px` | Verdict |
| :--- | :--- | :--- | :--- |
| Kannala-Brandt | `KannalaBrandtModel` | 0.00021 | near-exact; OpenCV `cv2.fisheye`-ready |
| EUCM | `EUCMModel` | 0.014 | near-exact |
| UCM | `UCMModel` | 0.334 | lossy — UCM has one fewer shape parameter |
| RadTan @ 120° | `RadTanModel` | 0.768 | lossy — pinhole cannot hold a wide FOV |

Supported models: **UCM, EUCM, Kannala-Brandt, RadTan, OCamCalib, Double Sphere**.
All implement the same contract, so any can be a source *or* a target. (OCamCalib,
class `OCamModel`, is supported but left out of the table above; it converts well —
`rms_px` ≈ 0.54 — but the four rows shown are enough to make the expressiveness point.)

```python
from ds_msp import DoubleSphereModel, EUCMModel, convert
ds = DoubleSphereModel.sample()

eucm, report = convert(ds, EUCMModel, width=1920, height=1080)
print(report["rms_px"])            # -> 0.014
```

## Restrict the FOV for narrow targets

Pinhole-style models (RadTan) cannot represent a >180° fisheye. Convert one without
limiting the field of view and rays the target can never reproduce will drag the
fit. Pass `max_fov_deg` to fit and report only the representable region:

```python
from ds_msp import DoubleSphereModel, RadTanModel, convert
ds = DoubleSphereModel.sample()

rt, report = convert(ds, RadTanModel, width=1920, height=1080, max_fov_deg=120.0)
print(report["rms_px"])            # -> 0.768  over the covered 120°
print(report["fov_covered_deg"])   # -> 119.9
```

`max_fov_deg` is the **full** angle. The report now reflects the 120° cone the
pinhole model is meant to cover, instead of being dominated by unrepresentable
peripheral rays.

Notice that `fov_covered_deg` reads 119.9, just under the 120.0 you requested. The
fit samples a discrete pixel grid, so the widest sampled ray lands a fraction of a
degree inside the cap rather than exactly on it.

> **Warning:** a low `rms_px` over a restricted FOV does not mean the conversion is
> lossless — it means it is faithful *within that FOV*. Check `fov_covered_deg`
> against the field of view your application actually uses.

## Use the converted model everywhere

The converted model is a first-class camera model. Every service in DS-MSP depends
only on the model contract, so pose estimation, undistortion, and the rest work on
it unchanged. Converting is a one-line swap in your pipeline:

```python
import numpy as np
from ds_msp import DoubleSphereModel, KannalaBrandtModel, convert, solve_pnp

ds = DoubleSphereModel.sample()
kb, _ = convert(ds, KannalaBrandtModel, width=1920, height=1080)   # DS -> OpenCV fisheye

object_points = np.array([[0, 0, 0], [0.1, 0, 0],          # (N, 3) known 3D points, metres
                          [0, 0.1, 0], [0.1, 0.1, 0]], dtype=float)
image_points = np.array([[610, 480], [720, 470],           # (N, 2) their pixels
                         [600, 590], [715, 580]], dtype=float)

ok, rvec, tvec = solve_pnp(kb, object_points, image_points)   # same call as for the DS model
print(ok)                                                    # -> True   the pose solve succeeded
```

Because KB and RadTan use exactly OpenCV's distortion conventions, the converted
model also plugs straight into OpenCV: `kb.K` and `kb.distortion` (shape `(4,)` for
KB) go directly into `cv2.fisheye.*`.

## Load your own calibration

The recipe above used `DoubleSphereModel.sample()` so it runs with no files. To
convert a calibration you produced, load it with `from_dict` and pass your real
image size:

```python
import json
from ds_msp import DoubleSphereModel, KannalaBrandtModel, convert

# point this at your own file; the JSON needs keys: fx, fy, cx, cy, xi, alpha
with open("results/calibration_params.json") as f:
    ds = DoubleSphereModel.from_dict(json.load(f))   # dict with fx, fy, cx, cy, xi, alpha

# use your camera's real sensor width/height, not the 1920x1080 placeholder
kb, report = convert(ds, KannalaBrandtModel, width=1920, height=1080)
print(report["rms_px"])
```

`from_dict` expects the model's parameter names as keys (for Double Sphere:
`fx, fy, cx, cy, xi, alpha`). Each model exposes a matching `to_dict` to save back.
Pass your camera's actual `width`/`height` so the sampling and report cover the
real sensor.

## Next steps

- **Why two different models can describe the same camera** — the intuition behind
  why conversion works at all:
  [Are two models the same camera?](../learn/are_two_models_the_same_camera.md)
- **The `adapt` API** (`convert`, the report fields, sampling):
  [API reference](../reference/index.md).

---

*Recap:* `convert(source, target_class, width, height)` fits a target model to an
existing calibration with no images, returns it plus a `report`, and the converted
model works with every DS-MSP feature. Check `report["rms_px"]`, `converged`, and
`fov_covered_deg`; use `max_fov_deg` for narrow targets.

The sample→unproject→linear-seed→refine conversion design follows the public
[Fisheye-Calib-Adapter](https://github.com/eowjd0512/fisheye-calib-adapter) work
(credited in the project README). Source:
[`ds_msp/adapt/convert.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/adapt/convert.py),
[`ds_msp/adapt/evaluate.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/adapt/evaluate.py).
