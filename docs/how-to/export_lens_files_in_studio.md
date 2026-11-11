# Export LDC meshes and Isaac Sim LUTs

Write lens-correction files for each camera: a TI LDC mesh for the lens-correction engine on a
TI Jacinto board, and an Isaac Sim lens LUT for rendering through the calibrated lens.

There are two ways to get them:

- **While calibrating.** Turn them on in the calibration's settings. The files land in the
  result folder.
- **For camera values you already have.** Use the **Export** page in DS-MSP Studio, or
  `ds-msp export` in a terminal. Both write the same files.

Either way the folder looks the same:

| Path | Holds |
|---|---|
| `ldc/<camera>_ldc_mesh.bin` | The mesh |
| `ldc/<camera>_ldc.json` | Its record: sizes, settings, the camera, the measured error, a checksum |
| `isaac_lut/<camera>_*` | Two EXR textures, a manifest and a USD camera |
| `exports.json` | Each camera's files, its measured error, or why it could not be written |

## While calibrating

**In Studio:** in step 5 of a new calibration (**More settings**), open **Files for a device or
a simulator**. Turn on **TI LDC mesh for each camera**, **Isaac Sim lens LUT for each camera**,
or both.

**In `calib_param.yml`:**

```yaml
export_ldc: "true"
ldc_downsample_factor: 4          # a grid point every 2^4 = 16 output pixels
ldc_balance: 0.5                  # 0 keeps all of the view, 1 crops the edges
# ldc_output_size: [ 1280, 720 ]  # corrected image size; leave out for each camera's own
export_isaac_lut: "true"
# isaac_lut_texture_size: [ 1024, 768 ]
```

**For one camera with `ds-msp-calibrate`:** add `--ldc`, `--isaac-lut`, or both. The same
settings are also available as the `ldc:` and `isaac_lut:` blocks of `calib_config.yml`.

A camera whose files cannot be written is named in `exports.json`, with the reason. The
calibration itself is still saved.

## For camera values you already have

### In Studio

Start Studio with `ds-msp studio` and open **Export**. The first page asks where the camera
values come from:

- **A calibration file:** MC-Calib or DS-MSP rig YAML, a Kalibr or DS-MSP camchain, COLMAP
  `cameras.txt`, nerfstudio `transforms.json`, or a DS-MSP camera model file. Studio lists every
  camera it finds. A camera it cannot read is listed with the reason.
- **A Studio result:** any result under **Results**. A result page also has **Export LDC or LUT
  files**, which starts here with that result chosen.
- **I'll type the values in:** one card per camera, with the lens type, its values and the
  image size.

Then four steps: the source, the cameras, what to write, and a name. While you change the LDC
settings, step 3 measures each mesh and shows the error, as in the table below.

Below the table, **See it, one camera at a time** draws the selected camera. Pick a camera from
the row of camera chips (each with its status colour), with **Previous camera** and **Next
camera**, or by clicking its row in the table:

- **On the TI board (LDC):** what the camera sees, beside the same picture after the board's
  correction. The picture is a test room (a box with a grid on every wall and a red cross ahead)
  or, when the source is a Studio result, one of the camera's photos. After a correct correction
  the room's lines are straight; black is where the corrected image shows nothing. **Mark where
  the mesh is less accurate** tints the picture amber where the mesh misplaces it by more than
  0.5 px, and red beyond 2 px.
- **In Isaac Sim (LUT):** the test room rendered through the ray directions the LUT stores.
  Floor at the bottom and the cross ahead mean the axes are right.

The pictures are drawn by Studio's server with the same interpolation the board uses: the
corrected test room matches an ideal pinhole view of the room except on the edges of grid lines
(under 2 % of pixels, tested on DS, KB and EUCM lenses). They are a simulation, not a check on
the board.

The job page lists each camera's files with what was measured, and draws them the same way from
the files written. A result with exports shows them too, with its own photos. It also shows the same export as
one `ds-msp export` command, which writes the same bytes.

### In a terminal

```bash
ds-msp export --input results/calibrated_cameras_data.yml --ldc --isaac-lut --output-dir exports
```

Without `--camera`, every camera in the file is exported. Use `--camera camera_2` (repeatable)
to choose cameras. For one camera from typed values, give `--model`, `--params` and
`--resolution` instead of `--input`:

```bash
ds-msp export --model ds --params fx=380 fy=380 cx=639.5 cy=399.5 xi=-0.2 alpha=0.6 \
  --resolution 1280 800 --ldc --isaac-lut --lut-texture-size 320 200 --output-dir exports
```

```text
LDC mesh cam0: worst mesh error 0.303 px -> exports/ldc/cam0_ldc.json
Isaac LUT cam0: /abs/path/exports/isaac_lut/cam0_isaac_lut.json
export summary: exports/exports.json
```

`ds-msp ldc` and `ds-msp lut` write one kind of file each, with the same camera choices.
`ds-msp ldc --input FILE --list-cameras` lists what a file holds.

## Read the LDC error

The error is how far the stored mesh sends a corrected pixel from where the lens values say it
belongs, in pixels of the camera image. It covers the grid spacing and the rounding to eighths of
a pixel that the file format uses. The page shows the typical (median) and the worst value.

**Shows the scene** is the share of the corrected image whose source lies on the sensor. The
rest is black.

A mesh is marked **Do not use** when some shifts were too large for the file (`q3_overflow` in
the record). Those shifts were cut off, so the edges are corrected wrongly.

Measured on the 184 calibrated cameras of the seven MC-Calib sequences and six rendered
25-camera rigs:

| How much of the view to keep | Grid every | Cameras cut off | Typical error px | Worst camera's worst px |
|---|---|---|---|---|
| 0 (all) | 16 px | 59 of 184, all standard lenses | 0.065 | 15.9 (not cut off) |
| 0.5 (usual) | 8 px | 0 | 0.033 | 0.83 |
| 0.5 (usual) | 16 px | 0 | 0.051 | 3.10 |
| 0.5 (usual) | 32 px | 0 | 0.145 | 11.3 |
| 1 (crop) | 16 px | 0 | 0.044 | 0.33 |

The worst values come from standard (RadTan) lenses at the edge of the image. On the fisheye
types (DS, EUCM, KB, UCM) the worst error at the usual settings was 0.29 px.

/// warning | Not yet checked on a TI board
The byte layout of the mesh file is written into its record (`layout`), and the record says
`"checked_on_board": false`. Check one mesh on your board before relying on it.
///

## Check a mesh by correcting real photos

The error above compares the mesh with the lens values directly. As a second check, board photos
from the same 13 captures were corrected through each mesh, as the board would, and through the
exact lens values. The corners were then found again in both images.

- **The two sets of corners agreed.** In each capture, the camera with the largest difference
  had a median of 0.03 to 0.14 px on the MC-Calib sequences, and 0.12 to 0.13 px on the
  rendered rigs.
- **Single corners differed by up to 11 px.** This happened at the steep edge of 190° lenses,
  where the corner finder settled differently in the two images. The mesh error measured
  directly for those cameras is at most 0.21 px.
- **No half-pixel offset.** Corners found in the exact correction lay a median of 0.05 to
  0.10 px from where the lens values put them. A half-pixel offset between the mesh and the
  lens values would show here as about 0.5 px.

## Related

- [Export a TI Jacinto LDC displacement mesh](export_ldc_mesh.md): the Python API.
- [Export an Isaac Sim camera LUT](export_isaac_sim_lut.md): the LUT files in detail.
- [Calibrate a rig in DS-MSP Studio](calibrate_a_rig_in_studio.md)
