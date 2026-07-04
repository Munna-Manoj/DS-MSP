# DS-MSP[rig] — Multi-Camera Rig Calibration Guide

Calibrate a multi-camera rig (the **extrinsics** — where each camera sits relative to the
others — plus per-camera **intrinsics**) from a folder of ChArUco images, driven by a single
MC-Calib-style `calib_param.yml`. This is the DS-MSP analogue of MC-Calib's `./calibrate
calib_param.yml`: same config schema, same output files, with one extension — you may choose a
*different camera model per camera* (`radtan`, `kb`, `ucm`, `eucm`, `ds`, `ocam`, `dsplus`,
`eucmplus`).

> **TL;DR**
> ```bash
> pip install ds-msp                                       # or: git clone + pip install -e .
> # 1. write yourself a starter config
> ds-msp-calibrate-rig --init-config calib_param.yml
> # 2. edit it (number_camera, board geometry, root_path, save_path, models) — see below
> # 3. run
> ds-msp-calibrate-rig --config calib_param.yml
> ```
> `ds-msp-calibrate-rig` is a real console command from `pip install ds-msp` alone — no repo
> clone needed. (A git-clone checkout can equivalently run `python scripts/calibrate_rig.py`;
> it's the exact same CLI either way.)

---

## 1. What you get out

The run writes MC-Calib's exact result set into `save_path/`:

| File | Contents |
|------|----------|
| `calibrated_cameras_data.yml` | per camera: `camera_matrix` (K), `distortion_vector`, `camera_model`, `camera_pose_matrix` (**the extrinsics**), `img_width/height`, `camera_group` |
| `calibrated_objects_data.yml` | the fused 3D calibration object (board corners in object frame) |
| `calibrated_objects_pose_data.yml` | object pose per frame |
| `reprojection_error_data.yml` | per-camera / per-point reprojection residuals |
| `detected_keypoints_data.yml` | **the detected 2D corners — save this to skip detection next time** (see §6) |
| `Detection/`, `Reprojection/` | (optional) overlay images per camera/frame |

The console prints per-camera reprojection RMS and, if a `GroundTruth.yml` / MC-Calib `Results/`
is found next to the data, the worst baseline error vs those references.

**Loading a camera back into a ready instance.** `calibrated_cameras_data.yml` holds every
camera in the rig, indexed 0-based (`camera_0`, `camera_1`, …) in write order. Load any one of
them straight into a [`CameraModel`](../ds_msp/core/contracts.py) — no manual K/distortion-array
handling:

```python
import ds_msp.rig as rig

cam0 = rig.load_camera("calibrated_cameras_data.yml", 0)
print(cam0)                       # e.g. KannalaBrandtModel(fx=..., fy=..., ...)
uv, valid = cam0.project(points_3d)
```

This is the MC-Calib-format analogue of `ds_msp.calib.load_camera` (the single-camera
`ds-msp-calibrate` output loader) — same one-liner ergonomics, different file shape, since
`calibrated_cameras_data.yml` splits `camera_matrix` (fx/fy/cx/cy) and `distortion_vector`
(model-specific length and order) into separate fields rather than Kalibr's single combined
`intrinsics` array. `load_camera` needs the file's `camera_model` (or the legacy
`distortion_type` int) to know which of the 8 models to reconstruct — always present in
DS-MSP's own output.

---

## 2. Prepare your data (folder & file naming)

The pipeline discovers images by a **strict folder convention**. Get this right and everything
else is automatic.

```
my_capture/                     <- this path is your root_path
├── Cam_001/                    <- camera 0   (folder index = camID + 1, zero-padded to 3 digits)
│   ├── 00000.png               <- frame 0
│   ├── 00001.png               <- frame 1
│   ├── 00002.png
│   └── ...
├── Cam_002/                    <- camera 1
│   ├── 00000.png
│   └── ...
├── Cam_003/                    <- camera 2
└── ...                         <- one folder per camera
```

Rules:

- **Folder name** = `<cam_prefix><camID+1:03d>`. With the default `cam_prefix: "Cam_"`, camera 0
  is `Cam_001`, camera 1 is `Cam_002`, … Cameras are **0-indexed internally** but the **folders
  are 1-indexed** — this matches MC-Calib. Change the prefix with the `cam_prefix` config key
  (e.g. `cam_`, `camera`) but keep the `+1`, 3-digit padding.
- **Frame correspondence is by filename, across cameras.** A frame is identified by the **digits
  in the filename** (`00007.png` → frame 7; `img_000007.png` → frame 7). The *same* board view
  seen by several cameras at the *same instant* **must share the same frame number** in each
  camera's folder. Frame numbers do not have to be contiguous or start at 0 (they are rebased
  internally), but they must be **consistent across cameras**.
- **Image formats:** `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`. All cameras can differ in
  resolution (a mixed-resolution rig is fine — intrinsics are per camera).
- **A camera need not see every frame.** Overlap is what links cameras: each pair that should be
  related must co-observe the board in enough shared frames. Cameras with no shared views cannot
  be tied into the rig.

### The calibration target

A printed **ChArUco** board (OpenCV ChArUco). You can use **one board** (enough when all cameras
overlap on it) or a **multi-board rigid object** (several boards fixed rigidly together — better
coverage for cameras that rarely see the same board; DS-MSP reconstructs the fused object
geometry automatically). You must know the board geometry (squares in X/Y, square and marker
lengths) and put it in the config.

---

## 3. Make a config

Generate a fully-commented starter and edit it:

```bash
ds-msp-calibrate-rig --init-config calib_param.yml
```

This copies [`ds_msp/rig/configs/calib_param.template.yml`](../ds_msp/rig/configs/calib_param.template.yml). Below is
every field that matters.

### Board geometry — *must match your printed board*
| Key | Meaning |
|-----|---------|
| `number_x_square`, `number_y_square` | squares along X / Y of the ChArUco board |
| `length_square` | full ChArUco square length, in board-generation units (the 100% size) |
| `length_marker` | inner ArUco marker length (< `length_square`; the 75% size) |
| `number_board` | number of physical boards in the rigid object (`1` for a single board) |
| `square_size` | **physical** square size in your unit of choice (m / cm / mm). **This sets the metric scale** of all 3D output and the camera baselines. Use the real measured size. |
| `*_per_board` (`number_x_square_per_board`, …) | only if boards differ in size; else leave `[]` |

### Camera model selection
| Key | Meaning |
|-----|---------|
| `number_camera` | number of cameras in the rig |
| `distortion_model` | global default: `0` = Brown→`radtan` (pinhole), `1` = Kannala→`kb` (fisheye) |
| `distortion_per_camera` | per-camera `0/1` list overriding the global, length = `number_camera` |
| `camera_models` *(DS-MSP extension, highest precedence)* | per-camera model **name** — `[ kb, kb, kb, kb, radtan, radtan, kb, kb ]`. Choose from `radtan, ucm, eucm, ds, kb, ocam, dsplus, eucmplus`. Overrides the two keys above. |

> **Which model?** Pinhole / low-distortion lens → `radtan`. Fisheye → `kb` is the safe default;
> `ds`/`ucm`/`eucm` are compact sphere models; **`dsplus` (DS+)** is the most expressive for very
> wide (≳170°) lenses. See [Choosing a model by FOV](../README.md#choosing-a-model-by-fov-from-experience)
> and the real-data comparison in §7.

### Intrinsics (optional prior)
| Key | Meaning |
|-----|---------|
| `cam_params_path` | path to an initial-intrinsics yml, or `"None"` to estimate from scratch. Schema = MC-Calib `calibrated_cameras_data`, extended with a per-camera `camera_model`. See [`ds_msp/rig/configs/camera_intrinsics.template.yml`](../ds_msp/rig/configs/camera_intrinsics.template.yml). |
| `fix_intrinsic` | `false` = estimate & refine intrinsics; `true` = **hold intrinsics fixed**, solve extrinsics only (requires `cam_params_path`) |

Behaviour (verified on real data, §7):
- **No file** + `fix_intrinsic=false` → every camera initializes from scratch.
- **File given**, stated model **matches** the chosen model → used natively (held if `fix_intrinsic=true`).
- **File given**, model **differs** + `fix_intrinsic=false` → a **warning** is printed and `convert()`
  carries the *same physical lens* into the chosen model, then the bundle adjustment refines it.
- **File given**, model **differs** + `fix_intrinsic=true` → **error** (you cannot hold a camera
  fixed in a model it was not provided in).

### Inputs
| Key | Meaning |
|-----|---------|
| `root_path` | folder containing the `Cam_00N/` subfolders (raw images) |
| `cam_prefix` | folder prefix (default `Cam_`) |
| `keypoints_path` | a pre-detected `detected_keypoints_data.yml`; `"None"` ⇒ detect from images. **Set this to skip detection** (§6). |

### Optimization / output
| Key | Meaning |
|-----|---------|
| `ransac_threshold` | reprojection threshold (px) for gross-outlier rejection — keep generous (e.g. 10) |
| `number_iterations` | max non-linear refinement iterations |
| `he_approach` | `0` bootstrapped hand-eye / `1` traditional (extrinsics init strategy) |
| `save_path` | output directory |
| `save_detection` / `save_reprojection` | `true` to write overlay images (needs raw images present) |
| `camera_params_file_name` | output cameras filename (`""` ⇒ `calibrated_cameras_data.yml`) |
| `webviewer` | `true` (default) to launch the live browser 3D view during the run; `false` to skip it. Independent of `verbose` (terminal progress). Needs the optional `ds-msp[webviewer]` extra. |

Relative paths resolve against the **config file's** directory. Override any value on the CLI
without editing the file:

```bash
ds-msp-calibrate-rig --config calib_param.yml \
    --set root_path=/abs/my_capture --set save_path=/abs/out \
    --set camera_models=kb,kb,kb,kb,radtan,radtan,kb,kb
```

---

## 4. Run it

```bash
ds-msp-calibrate-rig --config calib_param.yml
```

What happens internally:
1. **Detect** ChArUco corners in every image (or load `keypoints_path`).
2. **Reconstruct** the fused multi-board object from the detections (single board: built from
   config). MC-Calib's `calibrate3DObjects` analogue.
3. **Initialize** per-camera intrinsics (from scratch, or from `cam_params_path`) and the
   relative extrinsics from camera-group covisibility.
4. **Bundle-adjust** intrinsics + extrinsics + object poses jointly (staged), holding intrinsics
   if `fix_intrinsic=true`.
5. **Write** the MC-Calib result set to `save_path`.

The extrinsics you want are the `camera_pose_matrix` per camera in
`calibrated_cameras_data.yml` (camera-from-world 4×4, MC-Calib convention).

---

## 5. Extrinsics-only calibration (intrinsics already known)

If you already have trusted intrinsics (from a prior per-camera calibration), hold them fixed and
solve only the rig geometry:

1. Put the intrinsics in a yml (see [`ds_msp/rig/configs/camera_intrinsics.template.yml`](../ds_msp/rig/configs/camera_intrinsics.template.yml)
   — one entry per model is documented; `camera_model` per camera must match what you set in
   `camera_models`).
2. In the config: `cam_params_path: /abs/intrinsics.yml`, `fix_intrinsic: true`, and `camera_models`
   matching the stated models.
3. Run as usual. The bundle adjustment optimizes extrinsics + object poses only.

Emit a starter intrinsics file with:
```bash
ds-msp-calibrate-rig --init-intrinsics camera_intrinsics.yml
```

---

## 6. Detect once, calibrate many (keypoints reuse)

Corner detection over a whole rig is the slow part. Do it **once**, then re-run calibration in
seconds with different models / intrinsics / options on the *same* detections.

- **Save:** any run with a `save_path` writes `detected_keypoints_data.yml` (and
  `calibrated_objects_data.yml`) into it automatically.
- **Reuse:** point `keypoints_path` at that saved file and set `root_path: "None"`. No image
  detection runs — only the rig math. For multi-board rigs also pass `object_path` so the fused
  object geometry is identical across runs.

A ready-made reuse config: [`ds_msp/rig/configs/calib_param.keypoints.template.yml`](../ds_msp/rig/configs/calib_param.keypoints.template.yml)
(save a local copy — e.g. `reuse.yml` — from that link, or copy it out of your installed
package with `python -c "import importlib.resources,shutil; shutil.copyfile(importlib.resources.files('ds_msp.rig')/'configs/calib_param.keypoints.template.yml', 'reuse.yml')"`).

```bash
# run 1 — detect from images, save keypoints (uses the normal template)
ds-msp-calibrate-rig --config calib_param.yml \
    --set root_path=/abs/my_capture --set save_path=/abs/out

# run 2+ — reuse the saved keypoints, try a different model, fast
ds-msp-calibrate-rig --config reuse.yml \
    --set keypoints_path=/abs/out/detected_keypoints_data.yml \
    --set object_path=/abs/out/calibrated_objects_data.yml \
    --set save_path=/abs/out_dsplus \
    --set camera_models=dsplus,dsplus,dsplus,dsplus,radtan,radtan,dsplus,dsplus
```

The keypoints file is MC-Calib's exact `detected_keypoints_data.yml` schema, so files produced by
MC-Calib are accepted here and vice-versa.

---

## 7. Worked example & robustness (real 8-camera rig)

On a real 8-camera capture (cam 0–3 & 6–7 fisheye, cam 4–5 pinhole), reusing one detection set
across intrinsics scenarios — varying **only** how intrinsics are provided — gives:

| Scenario | mean RMS (px) | extrinsics vs from-scratch |
|----------|---------------|----------------------------|
| from scratch (`kb`/`radtan`) | **0.5665** | — |
| provided intrinsics + refine | 0.5665 | Δrot 0.000°, Δt 0.0 mm |
| provided intrinsics + **fixed** (extrinsics-only) | 0.5665 | Δrot 0.047°, Δt 2.0 mm |
| convert `kb`→`dsplus` (warn + convert) | 0.5623 | matches from-scratch dsplus |

Takeaways (these are *measured*, not asserted):
- The pipeline converges to the **same extrinsics and the same reprojection error** whether
  intrinsics are estimated from scratch, provided and refined, or provided and held fixed —
  robust to how you supply intrinsics.
- `convert()` is faithful: seeding a model by conversion lands in the **same optimum** the bundle
  adjustment reaches from scratch in that model.
- **Model choice still matters.** Plain `ds` cannot represent these ≳170° lenses (it saturates at
  ~16 px); switching the *same run* to **`dsplus`** drops it to **0.56 px**, matching/beating the
  `kb`+`radtan` baseline. When a model under-fits, the residual shows it honestly rather than
  hiding the limitation.

See [Choosing a model by FOV](../README.md#choosing-a-model-by-fov-from-experience) for how to pick a
model for *your* lens before you calibrate.

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| "config has neither keypoints_path nor root_path" | both are `"None"` — set one |
| A camera is missing from the output | its folder wasn't found (`<cam_prefix><id+1:03d>`) or it shares no frames with the rest |
| Frames don't line up across cameras | filenames must encode the **same frame number** for the same instant in every camera |
| `fix_intrinsic=true` error about a missing/mismatched model | provide intrinsics for every camera with `camera_model` matching `camera_models`, or set `fix_intrinsic=false` to convert+refine |
| Large reprojection on wide lenses | the model under-fits — try `kb` or `dsplus` for those cameras (§7) |
| Want overlays | set `save_detection: true` / `save_reprojection: true` **and** keep `root_path` pointing at the images |
| Browser window keeps popping open | set `webviewer: false` in the config (or `--no-webviewer` on the CLI) |
| Wrong real-world scale | `square_size` must be the **physically measured** square size |

---

### See also
- [`ds_msp/rig/configs/calib_param.template.yml`](../ds_msp/rig/configs/calib_param.template.yml) — annotated base config
- [`ds_msp/rig/configs/calib_param.keypoints.template.yml`](../ds_msp/rig/configs/calib_param.keypoints.template.yml) — keypoints-reuse config
- [`ds_msp/rig/configs/camera_intrinsics.template.yml`](../ds_msp/rig/configs/camera_intrinsics.template.yml) — initial-intrinsics schema (all models)
- [`docs/learn/`](learn/README.md) — the geometry curriculum (camera models, robust detection, evaluation)
