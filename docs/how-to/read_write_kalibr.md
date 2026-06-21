# Read and write Kalibr camchain YAML

Load a Kalibr `camchain` YAML into a DS-MSP camera model, and write a DS-MSP model
back out as Kalibr YAML — so you can move calibrations in and out of the Kalibr
ecosystem without retyping intrinsics.

> **Prerequisites**
> - DS-MSP installed (`pip install ds-msp`).
> - A Kalibr `camchain*.yaml` file to read, *or* a DS-MSP model to write out. The
>   first recipe below uses a bundled TUM-VI camchain so it runs with no extra
>   files; the [write recipe](#write-a-model-to-kalibr-yaml) needs no files at all.
> - **Run from the repo root.** Snippets that read the TUM-VI fixture use the
>   relative path `datasets/tumvi/…`. Run them with your working directory set to
>   the repository root (`cd /path/to/DS-MSP`), or replace the path with the
>   absolute path to the file on your system.
> - This is a **format I/O** recipe — it moves parameters between files and model
>   objects. It does not run a calibration.

## Read a camchain in one call

`load_kalibr(path, cam="cam0")` reads one camera's stanza and returns the matching
DS-MSP model. The function picks the model class from the YAML's `camera_model` /
`distortion_model` fields, so you get back the right type automatically:

```python
from ds_msp.io.kalibr import load_kalibr

path = "datasets/tumvi/dataset-room1_512_16/dso/camchain.yaml"
model = load_kalibr(path, cam="cam0")

print(type(model).__name__)   # -> KannalaBrandtModel
print(model.name)             # -> 'kb'
```

The TUM-VI camchain declares `camera_model: pinhole` with
`distortion_model: equidistant` — Kalibr's name for the Kannala-Brandt fisheye
model. So `load_kalibr` returns a `KannalaBrandtModel`. The file picks the class,
not you.

> **Notice:** `cam="cam0"` selects which camera in the chain to load. A stereo
> camchain has `cam0`, `cam1`, …; pass `cam="cam1"` for the second camera. If the
> requested name is absent, the loader falls back to the first `cam*` key it finds.

## Get the resolution too

`load_kalibr` returns only the model. When you also need the sensor size — and you do, to
write the file back out — call `load_kalibr_with_resolution`. It returns
`(model, (width, height))`:

> This snippet continues from the [read recipe](#read-a-camchain-in-one-call) above and
> reuses its `path`. To run it on its own, set
> `path = "datasets/tumvi/dataset-room1_512_16/dso/camchain.yaml"` first.

```python
from ds_msp.io.kalibr import load_kalibr_with_resolution

model, (width, height) = load_kalibr_with_resolution(path, cam="cam0")

print((width, height))                 # -> (512, 512)
print(round(model.K[0, 0], 4))         # fx -> 190.9785
print(round(model.K[1, 1], 4))         # fy -> 190.9733
print(round(model.K[0, 2], 4))         # cx -> 254.9317
print(model.distortion.round(6).tolist())
# -> [0.003482, 0.000715, -0.002053, 0.000203]   (k1, k2, k3, k4)
```

`model.K` is the 3×3 intrinsic matrix. `model.distortion` is the `(4,)` Kannala-Brandt
coefficient vector `[k1, k2, k3, k4]` — the four `distortion_coeffs` from the YAML, in
order.

## Which models the Kalibr I/O supports

DS-MSP maps five model families to and from the Kalibr camchain format. On read, the
`camera_model` / `distortion_model` pair in the YAML decides the DS-MSP class. On
save, the model's type decides the fields written:

| DS-MSP model | Kalibr `camera_model` | `distortion_model` | `intrinsics` order | `distortion_coeffs` |
| :--- | :--- | :--- | :--- | :--- |
| `DoubleSphereModel` | `ds` | `none` | `[xi, alpha, fx, fy, cx, cy]` | `[]` |
| `EUCMModel` | `eucm` | `none` | `[alpha, beta, fx, fy, cx, cy]` | `[]` |
| `KannalaBrandtModel` | `pinhole` | `equidistant` | `[fx, fy, cx, cy]` | `[k1, k2, k3, k4]` |
| `RadTanModel` | `pinhole` | `radtan` | `[fx, fy, cx, cy]` | `[k1, k2, p1, p2]` |
| `UCMModel` | `omni` | `none` | `[xi_mei, fx, fy, cx, cy]` | `[]` |

Two mappings need care:

- **UCM `xi`** — Kalibr's `omni` model stores the Mei mirror parameter
  `xi_mei = alpha / (1 - alpha)`, not DS-MSP's unified `alpha`. The I/O converts
  between them automatically on both read and write.
- **RadTan `k3`** — Kalibr's `radtan` has only four coefficients (`k1, k2, p1, p2`).
  A non-zero `k3` cannot be stored, so it is **dropped on export with a warning**.
  Keep `k3 = 0` if you need an exact RadTan round-trip.

A plain `pinhole` with `distortion_model: none` loads as a `RadTanModel` with zero
distortion. An `omni` model carrying distortion raises `NotImplementedError` — the
combination is not representable.

This page shows two families end to end — a Kannala-Brandt read and a Double Sphere
write — as representative of all five. The read and write calls are identical across
families: `load_kalibr` reads whatever class the file declares, and `save_kalibr`
emits whatever class you hand it. To write a EUCM, UCM, or RadTan file, pass that
model to the same `save_kalibr` call below; it places the fields from the row above.

## Write a model to Kalibr YAML

`save_kalibr(model, path, width, height, cam="cam0")` writes one model as a
single-camera camchain. The function reads the model's type and emits the matching
`camera_model` / `distortion_model` / `intrinsics` fields from the table above:

```python
import tempfile, os
from ds_msp.io.kalibr import save_kalibr
from ds_msp.models.double_sphere import DoubleSphereModel

ds = DoubleSphereModel.sample()          # bundled DS calibration, no file needed
out = os.path.join(tempfile.gettempdir(), "camchain.yaml")

save_kalibr(ds, out, width=1920, height=1080, cam="cam0")
print(open(out).read())
```

```yaml
cam0:
  camera_model: ds
  intrinsics: [0.183, 0.809, 711.57, 711.24, 949.18, 518.81]   # xi, alpha, fx, fy, cx, cy
  distortion_model: none
  distortion_coeffs: []
  resolution: [1920, 1080]
```

The `intrinsics` list leads with `xi, alpha` — the Double Sphere ordering Kalibr
expects — then `fx, fy, cx, cy`. `save_kalibr` placed every field in the right slot,
so you never format the YAML by hand.

## Confirm the round-trip

A save followed by a load must return the same model with the same parameters. Run
this check to verify interop before you ship a file downstream:

```python
import tempfile, os
import numpy as np
from ds_msp.io.kalibr import save_kalibr, load_kalibr
from ds_msp.models.double_sphere import DoubleSphereModel

ds = DoubleSphereModel.sample()
out = os.path.join(tempfile.gettempdir(), "rt.yaml")

save_kalibr(ds, out, width=1920, height=1080, cam="cam0")
back = load_kalibr(out, cam="cam0")

print(type(back) is type(ds))                       # -> True
print(np.allclose(back.params, ds.params, atol=1e-9))  # -> True
print(np.max(np.abs(back.params - ds.params)))      # -> 0.0
```

The reloaded model is the same class with identical `params` — max difference
**0.0** here, exact to machine precision. (The same exact round-trip holds for EUCM,
KB, UCM, and RadTan with `k3 = 0`; RadTan with `k3 ≠ 0` loses only `k3`, per the note
above.)

> **Notice:** `model.params` is each model's flat parameter vector — for Double
> Sphere, `[fx, fy, cx, cy, xi, alpha]`. Comparing `params` is the quickest way to
> assert two models are equal.

## Read stereo extrinsics

A multi-camera camchain stores each camera's pose relative to the previous one in
`T_cn_cnm1`, a 4×4 transform. Call `load_kalibr_extrinsics(path, cam="cam1")` to read
that matrix:

```python
import numpy as np
from ds_msp.io.kalibr import load_kalibr_extrinsics

path = "datasets/tumvi/dataset-room1_512_16/dso/camchain.yaml"
T_cam1_cam0 = load_kalibr_extrinsics(path, cam="cam1")   # 4x4, cam0 -> cam1
print(T_cam1_cam0.shape)                                 # -> (4, 4)
print(round(float(np.linalg.norm(T_cam1_cam0[:3, 3])), 5))  # baseline -> 0.10109 m
```

`T_cn_cnm1` maps points from the *previous* camera's frame into this one, so
`cam1`'s `T_cn_cnm1` is `T_cam1_cam0`. The TUM-VI stereo baseline read here is
**0.10109 m** (~10 cm).

> **Notice:** only `cam1` and later cameras carry `T_cn_cnm1`. `cam0` is the chain
> origin and has no pose relative to a previous camera, so calling
> `load_kalibr_extrinsics(path, cam="cam0")` raises `KeyError: 'T_cn_cnm1'`. Pass
> `cam="cam1"` (or higher) to read a transform; for `cam0` the relative pose is the
> identity by definition.

## Real-world usage

[`examples/09_monocular_vo_tumvi.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/examples/09_monocular_vo_tumvi.py)
loads the same TUM-VI camchain with `load_kalibr` to drive monocular visual
odometry on real data — the recipe above is exactly the front end of that pipeline.

## Try it yourself

Change `cam="cam0"` to `cam="cam1"` in the [read recipe](#get-the-resolution-too)
and reload from the TUM-VI camchain. Predict: same model class (`KannalaBrandtModel`),
but slightly different `fx` and distortion (it is the other physical lens). Confirm
the new `fx` rounds to **190.4424**.

## Next steps

- **Convert the loaded model to another model** before exporting — e.g. read a Kalibr
  KB file and write back a Double Sphere or OpenCV-ready model:
  [Convert a calibration between camera models](convert_between_models.md).
- **Calibrate from scratch, then export** to Kalibr for the rest of your toolchain:
  [Calibrate any camera model](calibrate_any_model.md).

---

*Recap:* `load_kalibr(path, cam)` reads a camchain stanza into the DS-MSP model the
file declares; `load_kalibr_with_resolution` adds `(width, height)`;
`save_kalibr(model, path, width, height, cam)` writes a model back out with the
right Kalibr field ordering. Five model families are supported (DS, EUCM, KB,
RadTan, UCM), `omni` uses the Mei `xi_mei` parameter, and RadTan `k3` cannot be
stored. Round-trips are exact to machine precision (`0.0` max param diff). Source:
[`ds_msp/io/kalibr.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/io/kalibr.py).
