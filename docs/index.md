# DS-MSP

A tested wide-<abbr title="Field of View">FOV</abbr> (fisheye / Double Sphere) camera library —
project and unproject through 8 camera models to **~1e-13 px** round-trip.

It doubles as a guided course in the geometry behind
<abbr title="Simultaneous Localization And Mapping">SLAM</abbr> and
<abbr title="Structure from Motion">SfM</abbr>: every chapter proves a number.

## Quickstart

<!-- termynal -->
```
$ pip install ds-msp
```

{* docs_src/guides/index/quickstart.py hl[12,13,19,20,21] *}

<!-- termynal -->
```
$ python -m docs_src.guides.index.quickstart
500/500 points valid through the round trip
max round-trip error: 2.27e-13 px
```

Every model behind that `cam` exposes the same `project`/`unproject` pair, plus analytic
Jacobians for bundle adjustment. Swap the camera, not the code.

## Pick a door

<div class="grid cards" markdown>

- :material-school: **[Learn](learn/README.md)** — the ordered tutorial path, calibration first.
- :material-toolbox: **[How-to](how-to/README.md)** — task recipes: calibrate, convert,
  undistort, solve <abbr title="Perspective-n-Point — solving for camera pose from n known 3D points and their 2D projections.">PnP</abbr>, export an <abbr title="Lens Distortion Correction -- hardware that undistorts a fisheye frame from a stored per-pixel displacement mesh, instead of a CPU/GPU remap.">LDC</abbr> mesh.
- :material-book-open-variant: **[Explanation](explain/README.md)** — the math and the *why*
  behind the library's design choices.
- :material-api: **[API Reference](reference/index.md)** — complete signatures, generated from
  the library's own docstrings.
- :material-camera-iris: **[Calibrate a camera](CALIBRATE_GUIDE.md)** — config-driven,
  `ds-msp-calibrate`.
- :material-camera-burst: **[Calibrate a rig](RIG_CALIBRATION_GUIDE.md)** — multi-camera,
  `ds-msp-calibrate-rig`.
- :material-swap-horizontal: **[Multi-model library & conversion](MULTI_MODEL.md)** — calibrate
  in one model, convert to any other, no re-shooting.
- :material-cube-scan: **[Interactive studio →](studio/)** — drive the Double Sphere model live
  in 3D, in your browser.

</div>

New here? Start with **[Learn](learn/README.md)**. Want the full recipe cookbook (undistort,
PnP, rig calibration, hardware export)? See the
[README quick start](https://github.com/Munna-Manoj/DS-MSP#readme).
