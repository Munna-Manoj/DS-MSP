# Export a TI Jacinto LDC displacement mesh

Generate a displacement-mesh lookup table (LUT) that the on-chip
<abbr title="Lens Distortion Correction -- hardware that undistorts a fisheye frame from a stored per-pixel displacement mesh, instead of a CPU/GPU remap.">LDC</abbr>
engine on a TI Jacinto J7 / TDA4 <abbr title="System on Chip">SoC</abbr> can read, so the
hardware undistorts each fisheye frame for you.

This is a task recipe — no theory. If you want to undistort on the CPU/GPU instead of on the
SoC, see [Undistort a fisheye image](undistort_images.md).

**Prerequisites**

- `ds_msp` installed (`numpy` comes with it).
- A calibrated `DoubleSphereCamera`. `width`/`height` on the model are **not** required by
  the mesh generator — it uses the `output_width`/`output_height` arguments you pass to
  `generate_mesh_and_intrinsics`. If you still need to calibrate, start from the
  [README usage](https://github.com/Munna-Manoj/DS-MSP#readme).
- The output of this recipe is a NumPy array you flash to the SoC; this page does not cover
  the board-side flashing toolchain.

## Generate the mesh in five lines

Build the camera, wrap it in `TI_LDC_MeshGenerator`, and ask for a mesh at the output
resolution you want on-chip. You get back the quantized mesh and the rectified intrinsics
`K_new` that describe the undistorted image the mesh produces.

{* docs_src/how_to/export_ldc_mesh/mesh_pipeline.py ln[10:30] hl[20:21,23:24,29:30] *}

<div class="termy">

```console
$ python3 -m docs_src.how_to.export_ldc_mesh.mesh_pipeline
# (excerpt -- this stage's prints; the full pipeline output is at the end of this page)
(69, 121, 2) int16
426.84
```

</div>

You now have everything the SoC needs: the displacement mesh and the matrix `K_new` that
defines the undistorted image it will output.

The grid is `(69, 121, 2)` because the generator samples one mesh node every
`2**downsample_factor = 16` output pixels, plus a one-node border, across the `1920x1080`
frame.

/// note
`generate_mesh_and_intrinsics(output_width, output_height, ...)` takes the *output*
(undistorted) resolution. It can differ from the sensor resolution — the dimensions on `cam`
describe the input fisheye, the arguments describe the on-chip output.
///

## What the dictionary contains

`generate_mesh_and_intrinsics` returns one dict. These are the keys and their shapes for the
call above.

| Key | Type / shape | What it is |
| :-- | :-- | :-- |
| `mesh_lut` | `(69, 121, 2)` `int16` | Q3 fixed-point `(h, v)` displacements — the array you flash to the LDC. |
| `mesh_lut_float` | `(69, 121, 2)` `float64` | The same displacements before quantization (for verification on the host). |
| `K_new` | `(3, 3)` `float64` | Rectified pinhole intrinsics of the undistorted output image. |
| `config` | `dict` | The call parameters, the resulting `mesh_size`, and the source DS intrinsics — a self-describing record to flash alongside the mesh. |

{* docs_src/how_to/export_ldc_mesh/mesh_pipeline.py ln[10:35] hl[33:35] *}

<div class="termy">

```console
$ python3 -m docs_src.how_to.export_ldc_mesh.mesh_pipeline
# (excerpt -- this stage's prints)
['mesh_lut', 'mesh_lut_float', 'K_new', 'config']
(69, 121, 2)
4
```

</div>

## Read the Q3 fixed-point format

Each mesh node holds two `int16` values — the horizontal and vertical displacement — in
<abbr title="Q3 fixed point: a real value stored as an integer equal to the value times 2^3 (8), rounded; the hardware divides by 8 to recover it.">Q3</abbr>
fixed point.

The LDC hardware reads these integers and divides by `8` internally.

To recover a node's displacement in pixels, divide by `8`. Read a displacement this way: to
fill output pixel `p`, sample the input fisheye at `p + delta`. Displacements grow toward the
corners.

{* docs_src/how_to/export_ldc_mesh/mesh_pipeline.py ln[10:43] hl[39:41,43] *}

<div class="termy">

```console
$ python3 -m docs_src.how_to.export_ldc_mesh.mesh_pipeline
# (excerpt -- this stage's prints)
[ -87 -156]
[-10.875 -19.5  ]
-3046 2873
```

</div>

The integer range of this mesh runs from `-3046` to `2873` Q3 units — roughly `-381 px` to
`+359 px`.

/// warning
A much wider FOV produces larger displacements, which can push Q3 values past the `int16`
range (`-32768..32767`) and wrap silently to the wrong sign. After generating a mesh for an
aggressive FOV, check `mesh_lut.min()` and `mesh_lut.max()` stay inside that range. If they
sit near the limits, raise `balance` to crop the periphery before flashing.
///

## Trade mesh size against accuracy with `downsample_factor`

`downsample_factor` is the power-of-two spacing between mesh nodes: the generator samples one
node every `2**downsample_factor` output pixels.

- A smaller factor stores more nodes — a denser, more accurate mesh.
- A larger factor stores fewer — a smaller LUT the hardware bilinearly interpolates between.

| `downsample_factor` | Node spacing | Mesh shape (for `1920x1080`) | Nodes |
| :-- | :-- | :-- | :-- |
| `3` | 8 px | `(136, 241, 2)` | denser, larger LUT |
| `4` | 16 px | `(69, 121, 2)` | balanced (the default) |
| `5` | 32 px | `(35, 61, 2)` | coarser, smaller LUT |

{* docs_src/how_to/export_ldc_mesh/mesh_pipeline.py ln[10:48] hl[46:48] *}

<div class="termy">

```console
$ python3 -m docs_src.how_to.export_ldc_mesh.mesh_pipeline
# (excerpt -- this stage's prints)
3 8 (136, 241, 2)
4 16 (69, 121, 2)
5 32 (35, 61, 2)
```

</div>

`balance` is the same field-of-view knob as in CPU undistortion:

- `balance=0.0` keeps the widest scene, with black corners.
- `balance=1.0` crops in until the borders are gone.

It sets `K_new` — at `balance=0.5` here, `fx_new = 426.84 px`. See
[Undistort a fisheye image](undistort_images.md) for how `balance` trades FOV against borders.

## Undistort keypoints with the closed form, not the mesh

Use the mesh for the **picture**. Undistort **keypoints** with the closed-form
`cam.undistort_points(pts, K_new)` at the same `K_new` — that is, the same `balance`.

The mesh's point-inverse is exact at the center and diverges toward the periphery. Sharing
`K_new` keeps the image pipeline and the keypoint pipeline on the same rectified frame.

{* docs_src/how_to/export_ldc_mesh/mesh_pipeline.py ln[10:57] hl[52:57] *}

<div class="termy">

```console
$ python3 -m docs_src.how_to.export_ldc_mesh.mesh_pipeline
(69, 121, 2) int16
426.84
['mesh_lut', 'mesh_lut_float', 'K_new', 'config']
(69, 121, 2)
4
[ -87 -156]
[-10.875 -19.5  ]
-3046 2873
3 8 (136, 241, 2)
4 16 (69, 121, 2)
5 32 (35, 61, 2)
[[ 967.68  555.05]
 [1427.98  832.03]
 [ 657.04  350.07]]
[ True  True  True]
```

</div>

That last block is the full pipeline output — the same command as every stage above, run once,
start to finish.

### Why share `K_new`

Measured against the closed-form result over keypoints spread across the frame:

- the mesh point-inverse agrees to a **median of ~0.08 px**,
- and to **~0.05 px in the central region** (radius `< 300 px`).

It diverges sharply toward the periphery. Out at the corners (here, roughly `r > 600 px` from
the principal point) the disagreement reached **~80 px** in this configuration.

So use the mesh to render the image, and the closed form for any coordinate you need
precisely: PnP, feature tracks, reprojection. Both must use the same `K_new`.

/// warning
Do not undistort keypoints by inverting the displacement mesh. It is accurate only near the
center. `cam.undistort_points` is exact everywhere a ray is recoverable, and its second return
value flags points that are not.
///

## Troubleshooting: a camera method raises about image dimensions

`width`/`height` on the `DoubleSphereCamera` are **not** used by `TI_LDC_MeshGenerator` — the
mesh is sized from the explicit `output_width`/`output_height` arguments. The camera's own
image-level helpers do need them: `cam.compute_K_new()` and `cam.get_undistortion_maps()` both
raise `ValueError` without them.

{* docs_src/how_to/export_ldc_mesh/troubleshooting.py hl[15,23:26,29:31] *}

<div class="termy">

```console
$ python3 -m docs_src.how_to.export_ldc_mesh.troubleshooting
(69, 121, 2)
ValueError: compute_K_new requires image dimensions; construct with width=... and height=... (only needed for image-level ops).
426.84
```

</div>

## Try it yourself

Re-run the generator with `downsample_factor=5`. Before you run it, predict two things: will
the mesh shape have more or fewer nodes than `(69, 121, 2)`, and will `K_new` change?

Run it, then open the answer.

/// details | Answer
The node count drops to `(35, 61, 2)` — a coarser grid stores fewer nodes. `K_new` is
unchanged: it depends on `balance`, not on the node spacing. So you can shrink the LUT without
re-deriving the rectified frame your keypoint pipeline shares.
///

## Next steps

- **Undistort on the CPU/GPU instead** —
  [Undistort a fisheye image](undistort_images.md): the software path with the same `balance`
  knob, for hosts without an LDC engine.
- **The code used here** — source on GitHub:
  [`ds_msp/ldc.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/ldc.py)
  (`TI_LDC_MeshGenerator.generate_mesh_and_intrinsics`) and
  [`ds_msp/model.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/model.py)
  (`DoubleSphereCamera.undistort_points`).
- **Other recipes** — back to the [How-to guides](README.md).
