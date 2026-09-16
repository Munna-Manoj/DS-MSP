# Export a calibrated camera to Isaac Sim

Use this recipe when an Isaac Sim camera must reproduce the same projection as a
camera calibrated by DS-MSP. The exporter works through the common
`CameraModel.project`/`unproject` interface, so there is no model-specific renderer code.

## Generate the LUT during calibration

Add the optional block to `calib_config.yml`:

```yaml
isaac_lut:
  enabled: true
  output_dir: isaac_lut
  texture_width: 0     # 0 means the calibrated image width
  texture_height: 0    # 0 means the calibrated image height
  overwrite: false
```

Then run the normal calibration command:

```bash
ds-msp-calibrate --config calib_config.yml
```

For a one-off run, use flags instead:

```bash
ds-msp-calibrate ./images \
  --board charuco --rows 5 --cols 6 --square-size 0.025 --model ds \
  --save-dir ./results --isaac-lut
```

Use `--lut-texture-size WIDTH HEIGHT` for a quick lower-resolution test. For final
rendering, use a LUT at least as large as the render product; a smaller texture adds
interpolation error.

## Generate from an existing artifact

Kalibr/DS-MSP camchains, MC-Calib `calibrated_cameras_data.yml`, model JSON/YAML, and a
previous DS-MSP LUT manifest are accepted:

```bash
ds-msp lut --path ./results/camchain.yaml --camera cam0
ds-msp-lut --path ./Results/calibrated_cameras_data.yml --camera camera_3
```

`ds-msp --lut ...` is an alias for `ds-msp lut ...`.

Older MC-Calib artifacts may not state their exact camera family. Supply it explicitly:

```bash
ds-msp --lut --path ./camera.yml --model ds --camera 0 --resolution 2592 1800
```

## Generate directly from parameters

List the exact parameter order for every installed model:

```bash
ds-msp lut --list-models
```

Named parameters are easiest to audit:

```bash
ds-msp lut \
  --model ds --resolution 2592 1800 \
  --params fx=999.98 fy=999.17 cx=1271.98 cy=878.03 xi=0.18 alpha=0.62 \
  --output-dir ./isaac_lut
```

Ordered values are also accepted in the order printed by `--list-models`:

```bash
ds-msp lut --model ds --resolution 2592 1800 \
  --params 999.98 999.17 1271.98 878.03 0.18 0.62
```

`--param` is accepted as a singular alias for `--params`.

The same commands support `radtan`, `kb`, `ucm`, `eucm`, `ds`, `ocam`, and `dsplus`.

## What the output contains

Each content-addressed bundle contains:

- `*_ray_enter_direction.exr`: RGB float32, image NDC to RTX camera-local XYZ ray;
- `*_ray_exit_position.exr`: RGB float32, with RG holding ray-to-image NDC and B unused;
- `*_isaac_lut.json`: model parameters, dimensions, checksums, validation numbers, and the
  exact USD attribute names;
- `*_camera.usda`: a Camera prim with `OmniLensDistortionLutAPI` already authored.

DS-MSP writes the EXRs with its own scanline OpenEXR writer (`ds_msp.isaac_sim.exr`, ZIP
compressed, verified against the reference OpenEXR library), so no OpenEXR-enabled OpenCV
build is required.

The calibration principal point is baked into both textures. Therefore the generated USD
sets `opticalCenter` to half the nominal width/height, as NVIDIA requires, rather than
applying `cx,cy` a second time.

## Load it in Isaac Sim

The quickest no-code path is to add/reference the generated `*_camera.usda` in the stage,
then position that Camera prim.

To assign a manifest to an existing camera from Isaac's Script Editor:

```python
import omni.usd
from ds_msp.isaac_sim.usd import apply_lut_manifest

stage = omni.usd.get_context().get_stage()
apply_lut_manifest(
    stage,
    "/World/Robot/Camera",
    "/absolute/server/path/ds_2592x1800_..._isaac_lut.json",
)
```

The manifest and EXRs must be on the machine running Isaac Sim. A WebRTC client displays
the remote application but does not make local client paths visible to the server.

## Validate in the target Isaac Sim build

LUT support is an RTX renderer capability, so validate both schema loading and rendered
distortion in the exact Isaac/Kit build used for an experiment. DS-MSP includes a bounded
headless validator that loads every manifest in a directory, checks both assets, renders an
asymmetric scene, and compares each result with a pinhole baseline:

```bash
DS_MSP_ISAAC_LUT_MANIFEST_DIR=/absolute/server/path/to/luts \
DS_MSP_ISAAC_LUT_OUTPUT_DIR=/tmp/ds_msp_lut_test \
  isaacsim isaacsim.exp.full --no-window \
  --exec tools/verify_isaac_lut_in_kit.py
```

Inspect `report.json` and the PNG captures in the output directory. Every model shipped
with DS-MSP has also passed this validator on Isaac Sim 6.0.1.0 / Kit 110.1.2.

## Python API

```python
from ds_msp.isaac_sim import export_lut

bundle = export_lut(camera_model, 2592, 1800, "./isaac_lut")
print(bundle.manifest)
```

Generation checks `project(unproject(pixel))` on the valid sensor domain and refuses to
write a bundle when the maximum error exceeds `1e-5` pixels. Invalid input pixels use RTX's
documented `(0, 0, +1)` ray sentinel; non-projectable directions in the exit map are placed
off-screen rather than incorrectly mapped to pixel `(0, 0)`.
