# Calibrate a multi-camera rig

Recover a multi-camera rig's **extrinsics** — where each camera sits relative to the others —
plus per-camera **intrinsics**, from a folder of <abbr title="A chessboard with ArUco markers
embedded in each square, so corners stay identifiable even partially occluded.">ChArUco</abbr>
images, driven by a single config.

This is the multi-camera analogue of [`ds_msp[calib]`](CALIBRATE_GUIDE.md)'s config-driven
pattern: the same schema style as MC-Calib's own `./calibrate calib_param.yml`, the same output
files, with one extension — pick a *different camera model per camera*
(`radtan`, `kb`, `ucm`, `eucm`, `ds`).

**Prerequisites**

- `ds-msp` installed: `pip install ds-msp` (or `git clone` + `pip install -e .`).
- A folder of ChArUco images, one subfolder per camera — see
  [§2](#2-prepare-your-data-folder-file-naming) for the exact naming convention.
- A printed ChArUco board (or a multi-board rigid object) whose geometry you know.

## Quick start

Write yourself a starter config, edit it, and run it:

<div class="termy">

```console
$ pip install ds-msp
$ ds-msp-calibrate-rig --init-config calib_param.yml
wrote base calib_param template to: calib_param.yml
```

</div>

Edit the file — `number_camera`, board geometry, `root_path`, `save_path`, `camera_models` —
see [§3](#3-make-a-config) — then run it:

<div class="termy">

```console
$ ds-msp-calibrate-rig --config calib_param.yml
```

</div>

/// tip
`ds-msp-calibrate-rig` is a real console command from `pip install ds-msp` alone — no repo
clone needed. A git-clone checkout can equivalently run `python scripts/calibrate_rig.py`;
it's the exact same CLI either way.
///

---

## 1. What you get out

The run writes MC-Calib's exact result set into `save_path/`:

| File | Contents |
|------|----------|
| `calibrated_cameras_data.yml` | per camera: `camera_matrix` (K), `distortion_vector`, `camera_model`, `camera_pose_matrix` (the extrinsics), `img_width/height`, `camera_group` |
| `calibrated_objects_data.yml` | the fused 3D calibration object (board corners in object frame) |
| `calibrated_objects_pose_data.yml` | object pose per frame |
| `reprojection_error_data.yml` | per-camera / per-point reprojection residuals |
| `detected_keypoints_data.yml` | the detected 2D corners — save this to skip detection next time (see [§6](#6-detect-once-calibrate-many-keypoints-reuse)) |
| `calibration_report.txt`, `calibration_report.json` | human- and machine-readable result generated from the same report data |
| `calibration_manifest.json` | code/data/config/environment provenance, effective-input digests, and output hashes |
| `calibration_acceptance.json` | every check and gate the run applied; `diagnostics` holds per-camera outlier shares and per-board flatness |
| `diagnostics/residuals.npz` | one row per observed corner: where it was detected and its signed residual (px) |
| `diagnostics/lens_sigma.npz` | each camera's lens 1-sigma over a 24 x 18 pixel grid (free intrinsics only) |
| `Detection/`, `Reprojection/` | (optional) overlay images per camera/frame |

The two `diagnostics/` files are the evidence behind the verdicts, for plotting and inspection
tools. They take about 2 MB for a 25-camera rig with 100,000 corners.

```python
import numpy as np

r = np.load("save_path/diagnostics/residuals.npz")       # your save_path
cam0 = (r["cam"] == 0) & r["valid"]
print(np.sqrt(np.mean(r["dx"][cam0] ** 2 + r["dy"][cam0] ** 2)))   # camera 0 RMS, px
```

The console labels reprojection PASS/WARN/FAIL as a **training-fit verdict**. When an independent
`GroundTruth.yml` is available it also reports full all-pair pose accuracy. MC-Calib `Results/` is
shown separately as an implementation comparison, not as physical truth.

### Evaluate physical accuracy over your operating volume

Generate the shipped profile template:

```bash
ds-msp-calibrate-rig --init-accuracy-profile accuracy_profile.json
```

Before using it, edit all three deployment assumptions:

```json
{
  "schema": "ds-msp-rig-accuracy-profile",
  "schema_version": 1,
  "name": "warehouse-10mm-zone",
  "length_unit": "m",
  "roi": {
    "frame": "pair_source_camera",
    "min": [-2.0, -1.0, 1.0],
    "max": [2.0, 1.0, 10.0],
    "grid_shape": [5, 3, 10]
  },
  "required_pairs": [[0, 1], [0, 2], [1, 2]],
  "limits": {"task_space_p95_max": 0.01}
}
```

- `length_unit` must match `square_size`, the reference poses, and every ROI coordinate. With
  metres, `0.01` means 10 mm.
- ROI points live in each pair's **source-camera frame**. The box above samples 150 known 3D
  positions from 1–10 m range. You can use an explicit `"points": [[x,y,z], ...]` array instead.
- List every pair required by the application as `[source_camera, target_camera]`; order matters
  because the ROI is expressed in the source frame. `[]` evaluates all common unordered pairs in
  ascending camera order. A named pair that is missing makes the result `INCOMPLETE`; it never
  silently passes.
- Limits are optional. Other supported limits include rotation, camera-center, relative-translation,
  translation-direction, auxiliary baseline-length, and task-space maximum.

Reference the profile from `calib_param.yml`:

```yaml
accuracy_profile_path: "accuracy_profile.json"
```

The profile declares *how* to evaluate; it is not truth itself. DS-MSP discovers
`GroundTruth.yml` from the scenario hierarchy anchored by `root_path` (or `keypoints_path`). If
you reuse keypoints from a separate results directory, keep `root_path` pointing at the original
scenario's `Images/` directory; keypoint reuse still wins, while ground truth remains discoverable.
Otherwise the report correctly says `NOT_EVALUATED`.

For scenario-style CLI runs, pass `--accuracy-profile accuracy_profile.json` instead. The report's
`metrics.accuracy` section is schema-versioned and contains, for every comparable pair:

- geodesic rotation error in degrees;
- full camera-center vector error and relative translation percentage;
- translation-direction error in degrees;
- baseline-length error, explicitly labeled **auxiliary**;
- exact p50/p95/max 3D displacement across the declared ROI, plus the worst point/pair.

Interpret statuses literally: `EVALUATED` means a reference comparison ran; `INCOMPLETE` means a
required pair was absent; `NOT_EVALUATED` means truth or an ROI was unavailable. A low training-fit
residual does not replace independent truth, and the MC-Calib comparison does not become ground
truth merely because both tools agree.

### Reproduce an attributed run

A saved run can be verified and replayed from its manifest without overwriting the original:

```bash
MANIFEST=/absolute/path/to/your/results/calibration_manifest.json
REPLAY_OUTPUT=/absolute/path/to/a/new/replay-directory
python tools/replay_rig_manifest.py "$MANIFEST" --output "$REPLAY_OUTPUT"
```

The command first verifies every recorded source-file SHA-256 hash. The replay report then checks
that the effective 2D observations, target geometry, and deterministic MC-Calib outputs match. Use
`--verify-only` when you only want to check whether the recorded inputs are still intact. The paths
above are examples: the manifest is written inside the `save_path` of the original calibration run,
and the replay output must be a new or empty directory.

Follow [Reproduce and audit a rig calibration](how-to/reproduce_rig_calibration.md) for the complete
operator and developer workflow: reference-environment setup, manifest interpretation, exit statuses,
troubleshooting, golden baselines, retention, sharing/privacy, and the boundary between reproducibility
and physical calibration accuracy.

For release qualification and golden baselines, create a Python 3.10 environment and install the
exact numerical runtime before installing DS-MSP:

```bash
python3.10 -m venv .venv-production
.venv-production/bin/python -m pip install -r requirements-production.txt
.venv-production/bin/python -m pip install -e .
```

The public `pyproject.toml` intentionally retains compatibility ranges; the production requirements
file defines the one reference environment, not the only environment in which DS-MSP can run.

### Loading a camera back into a ready instance

`calibrated_cameras_data.yml` holds every camera in the rig, indexed 0-based (`camera_0`,
`camera_1`, …) in write order. Load any one straight into a
[`CameraModel`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/core/contracts.py) — no
manual K/distortion-array handling:

{* docs_src/guides/rig_calibration_guide/load_camera.py hl[36,37,40,41] *}

<div class="termy">

```console
$ python -m docs_src.guides.rig_calibration_guide.load_camera
KannalaBrandtModel(fx=900.000, fy=900.000, cx=960.000, cy=540.000, k=[0.01000, -0.02000, 0.00000, 0.00000])
uv[0] = (960.000, 540.000)   valid=True
uv[1] = (1136.736, 422.176)   valid=True
```

</div>

This is the MC-Calib-format analogue of `ds_msp.calib.load_camera` — the single-camera
`ds-msp-calibrate` output loader — with the same one-liner ergonomics but a different file
shape.

`calibrated_cameras_data.yml` splits `camera_matrix` (fx/fy/cx/cy) and `distortion_vector`
(model-specific length and order) into separate fields, rather than Kalibr's single combined
`intrinsics` array.

`load_camera` needs the file's `camera_model` (or the legacy `distortion_type` int) to know
which of the 7 models to reconstruct — always present in DS-MSP's own output.

---

## 2. Prepare your data (folder & file naming)

The pipeline discovers images by a **strict folder convention**. Get this right and everything
else is automatic.

```text
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

A printed ChArUco board (OpenCV ChArUco) works for most rigs.

Use a **multi-board rigid object** instead — several boards fixed rigidly together — for
cameras that rarely share a view of one board; DS-MSP reconstructs the fused object geometry
automatically.

You must know the board geometry (squares in X/Y, square and marker lengths) and put it in the
config.

---

## 3. Make a config

Generate a fully-commented starter and edit it:

<div class="termy">

```console
$ ds-msp-calibrate-rig --init-config calib_param.yml
wrote base calib_param template to: calib_param.yml
```

</div>

This copies [`ds_msp/rig/configs/calib_param.template.yml`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/rig/configs/calib_param.template.yml). Below is
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
| `camera_models` *(DS-MSP extension, highest precedence)* | per-camera model **name** — `[ kb, kb, kb, kb, radtan, radtan, kb, kb ]`. Choose from `radtan, ucm, eucm, ds, kb`. Overrides the two keys above. |

/// tip | Which model?
Pinhole / low-distortion lens → `radtan`. Fisheye → `kb` is the safe default; `ds`/`ucm`/`eucm`
are compact sphere models. (DS+ and OCam were removed: their extra intrinsic terms traded off
against camera rotation, [ADR-0031](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/process/architecture/decisions/ADR-0031-remove-dsplus-and-ocam.md).) See [Choosing a model by FOV](https://github.com/Munna-Manoj/DS-MSP/blob/main/README.md#choosing-a-model)
and the real-data comparison in [§7](#7-worked-example-robustness-real-8-camera-rig).
///

### Intrinsics (optional prior)
| Key | Meaning |
|-----|---------|
| `cam_params_path` | path to an initial-intrinsics yml, or `"None"` to estimate from scratch. Schema = MC-Calib `calibrated_cameras_data`, extended with a per-camera `camera_model`. See [`ds_msp/rig/configs/camera_intrinsics.template.yml`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/rig/configs/camera_intrinsics.template.yml). |
| `fix_intrinsic` | `false` = estimate & refine intrinsics; `true` = **hold intrinsics fixed**, solve extrinsics only (requires `cam_params_path`) |

Behaviour (verified on real data, [§7](#7-worked-example-robustness-real-8-camera-rig)):

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
| `keypoints_path` | a pre-detected `detected_keypoints_data.yml`; `"None"` ⇒ detect from images. **Set this to skip detection** ([§6](#6-detect-once-calibrate-many-keypoints-reuse)). |

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
| `accuracy_profile_path` *(DS-MSP extension)* | strict JSON deployment profile declaring physical unit, task-space ROI, required camera pairs, and optional accuracy limits. Relative paths resolve beside the config. |

/// note
Relative paths resolve against the **config file's** directory.
///

Override any value on the CLI without editing the file:

<div class="termy">

```console
$ ds-msp-calibrate-rig --config calib_param.yml \
    --set root_path=/abs/my_capture --set save_path=/abs/out \
    --set camera_models=kb,kb,kb,kb,radtan,radtan,kb,kb
```

</div>

---

## 4. Run it

<div class="termy">

```console
$ ds-msp-calibrate-rig --config calib_param.yml
```

</div>

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
`calibrated_cameras_data.yml` (**camera-to-world/group-reference** 4×4, MC-Calib convention).
DS-MSP's internal `RigState.T_c_g` uses the inverse world-to-camera convention; the accuracy module
normalizes this explicitly before comparison.

### Read the result in DS-MSP Studio

The report ends with the command that opens this result in your browser:

```text
view in DS-MSP Studio:  ds-msp studio --open "/path/to/rig/out"
```

Add `--studio` to open it as soon as the calibration is done. Studio needs
`pip install "ds-msp[ui]"`.

```console
$ ds-msp-calibrate-rig --config calib_param.yml --studio
```

Studio gives a verdict for the rig and each camera, at most three next steps, and per camera the
answers to "is there a pattern in the errors?" and "where can I trust this lens?". See
[View rig results in DS-MSP Studio](how-to/view_rig_results_in_studio.md).

Studio can also set up and start the calibration, with no config file to write: see
[Calibrate a rig in DS-MSP Studio](how-to/calibrate_a_rig_in_studio.md).

### Follow a run from another program

`--progress-file PATH` writes the run's progress to `PATH` as JSON, one object per line:

```console
$ ds-msp-calibrate-rig --config calib_param.yml --progress-file progress.jsonl
```

```text
{"t": 0.0, "event": "start"}
{"t": 0.61, "event": "detect", "cameras": {"0": [30, 30], "1": [30, 30]}}
{"t": 1.02, "event": "stage", "name": "(c) global joint BA + intrinsics", "index": 4}
{"t": 1.48, "event": "iter", "it": 3, "max_iter": 60, "rms": 0.066, "cost": 12.4}
{"t": 2.9, "event": "done", "rms": 0.066}
```

`t` is seconds since the start. `detect` and `save` map each camera to `[images done, images]`.
The log only listens: the calibration and its files are the same with or without it.

---

## 5. Extrinsics-only calibration (intrinsics already known)

If you already have trusted intrinsics (from a prior per-camera calibration), hold them fixed and
solve only the rig geometry:

1. Put the intrinsics in a yml (see [`ds_msp/rig/configs/camera_intrinsics.template.yml`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/rig/configs/camera_intrinsics.template.yml)
   — one entry per model is documented; `camera_model` per camera must match what you set in
   `camera_models`).
2. In the config: `cam_params_path: /abs/intrinsics.yml`, `fix_intrinsic: true`, and `camera_models`
   matching the stated models.
3. Run as usual. The bundle adjustment optimizes extrinsics + object poses only.

Emit a starter intrinsics file with:

<div class="termy">

```console
$ ds-msp-calibrate-rig --init-intrinsics camera_intrinsics.yml
wrote camera intrinsics template to: camera_intrinsics.yml
```

</div>

---

## 6. Detect once, calibrate many (keypoints reuse)

Corner detection over a whole rig is the slow part. Do it **once**, then re-run calibration in
seconds with different models / intrinsics / options on the *same* detections.

- **Save:** any run with a `save_path` writes `detected_keypoints_data.yml` (and
  `calibrated_objects_data.yml`) into it automatically.
- **Reuse:** point `keypoints_path` at that saved file and set `root_path: "None"`. No image
  detection runs — only the rig math. For multi-board rigs also pass `object_path` so the fused
  object geometry is identical across runs.
- **Reconstruct from the keypoints:** a `calibrated_objects_data.yml` in the same folder as
  `keypoints_path` is loaded even without `object_path`, and the target is then not rebuilt. The
  run prints a warning when this happens. To compare code versions or settings that change the
  target, copy the keypoints file to an empty folder first.

A ready-made reuse config: [`ds_msp/rig/configs/calib_param.keypoints.template.yml`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/rig/configs/calib_param.keypoints.template.yml)
(save a local copy — e.g. `reuse.yml` — from that link, or copy it out of your installed
package with `python -c "import importlib.resources,shutil; shutil.copyfile(importlib.resources.files('ds_msp.rig')/'configs/calib_param.keypoints.template.yml', 'reuse.yml')"`).

Run once from images, saving the keypoints (this uses the normal template):

<div class="termy">

```console
$ ds-msp-calibrate-rig --config calib_param.yml \
    --set root_path=/abs/my_capture --set save_path=/abs/out
```

</div>

Then reuse the saved keypoints for a second, much faster run — trying a different model for the
fisheye cameras:

<div class="termy">

```console
$ ds-msp-calibrate-rig --config reuse.yml \
    --set keypoints_path=/abs/out/detected_keypoints_data.yml \
    --set object_path=/abs/out/calibrated_objects_data.yml \
    --set save_path=/abs/out_eucm \
    --set camera_models=eucm,eucm,eucm,eucm,radtan,radtan,eucm,eucm
```

</div>

The keypoints file is MC-Calib's exact `detected_keypoints_data.yml` schema, so files produced by
MC-Calib are accepted here and vice-versa.

---

## 7. Worked example & robustness (real 8-camera rig)

On a real 8-camera capture (cam 0–3 & 6–7 fisheye, cam 4–5 pinhole), reusing one detection set
across intrinsics scenarios — varying **only** how intrinsics are provided — gives:

| Scenario | mean RMS (px) | extrinsics vs from-scratch |
|----------|---------------|----------------------------|
| from scratch (`kb`/`radtan`) | **0.5532** | — |
| provided intrinsics + refine | 0.5517 | Δrot 0.057°, Δt 2.8 mm |
| provided intrinsics + **fixed** (extrinsics-only) | 0.5641 | Δrot 0.989°, Δt 15.1 mm |

Takeaways (these are *measured*, not asserted):

- The pipeline reaches **close to the same reprojection error** (0.552–0.564 px across the three
  scenarios) whether intrinsics are estimated from scratch, provided and refined, or provided and
  held fixed — robust to how you supply intrinsics.
- Extrinsics are **close but not identical** across scenarios — a real, small basin difference,
  not a bug: refining from provided intrinsics stays nearest to from-scratch (Δrot 0.057°, Δt
  2.8 mm), while holding intrinsics **fixed** moves the extrinsics the most (Δrot 0.989°, Δt
  15.1 mm) since the bundle adjustment can no longer trade off intrinsics error against pose
  error.
- **Model choice still matters.** Plain `ds` cannot represent these ≳170° lenses (it saturates at
  ~16 px), while `kb` fits them at ~0.55 px. When a model under-fits, the residual shows it
  honestly rather than hiding the limitation.
- An earlier version of this table also had a row converting `kb` to the since-removed DS+ model
  (0.5470 px, Δrot 0.317°, Δt 6.7 mm from the from-scratch run). DS+ was removed because its tilt
  terms absorb camera rotation ([ADR-0031](https://github.com/Munna-Manoj/DS-MSP/blob/main/docs/process/architecture/decisions/ADR-0031-remove-dsplus-and-ocam.md)).

See [Choosing a model by FOV](https://github.com/Munna-Manoj/DS-MSP/blob/main/README.md#choosing-a-model) for how to pick a
model for *your* lens before you calibrate.

### Watch it converge

The replay below was recorded before DS+ was removed, from this run with the cameras converted
to DS+. That model no longer exists, so the command no longer runs as shown:

<div class="termy">

```console
$ ds-msp-calibrate-rig --config calib_param.seeded_dsplus_fisheye_radtan_pinhole.yml
[ds-msp][rig] WARNING: camera 0: config selects 'dsplus' but provided intrinsics are 'kb'.
Using convert() to seed 'dsplus' from the 'kb' lens, then refining it in the bundle adjustment.
[front-end] calibrated 8 cameras
[groups] 1 group(s): [[0, 1, 2, 3, 4, 5, 6, 7]]
  BA: rms 0.8202px iters=25 intr=fixed dense kernel=huber
  BA: rms 0.7131px iters=22 intr=fixed sparse kernel=huber
  BA: rms 0.6540px iters=19 intr=free sparse kernel=cauchy
  BA: rms 0.5929px iters=25 intr=free sparse kernel=cauchy
  BA: rms 0.5915px iters=25 intr=free sparse kernel=cauchy
```

</div>

Every run of `ds-msp-calibrate-rig` opens a live cinematic 3D view of exactly this — each camera's
depth in "the pond" driven by its own real reprojection error, rising as the bundle adjustment
converges, with a final reveal of the actual solved rig geometry and calibration board. Below is a
**replay of the run above**, frame-for-frame from its real recorded state — not a live server (this
page is static), and not synthetic — the same 429 real optimizer/detection snapshots this exact
command produced.

<iframe class="ds-wide-embed" src="../assets/rig_pond_replay/index.html" height="820" style="border: 1px solid var(--ds-border, #444); border-radius: 12px;" loading="lazy"></iframe>

/// tip
Drag to orbit, scroll to zoom — "take manual control" hands you the camera; "orient rig" lets you
match the scene's up-direction to your own setup, exactly like the live view.
///

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| "config has neither keypoints_path nor root_path" | both are `"None"` — set one |
| A camera is missing from the output | its folder wasn't found (`<cam_prefix><id+1:03d>`) or it shares no frames with the rest |
| Frames don't line up across cameras | filenames must encode the **same frame number** for the same instant in every camera |
| `fix_intrinsic=true` error about a missing/mismatched model | provide intrinsics for every camera with `camera_model` matching `camera_models`, or set `fix_intrinsic=false` to convert+refine |
| Large reprojection on wide lenses | the model under-fits — try `kb` for those cameras ([§7](#7-worked-example-robustness-real-8-camera-rig)) |
| Want overlays | set `save_detection: true` / `save_reprojection: true` **and** keep `root_path` pointing at the images |
| Browser window keeps popping open | set `webviewer: false` in the config (or `--no-webviewer` on the CLI) |
| `--studio` says it needs the ui extra | `pip install "ds-msp[ui]"`; the calibration itself already finished and saved |
| Wrong real-world scale | `square_size` must be the **physically measured** square size |

---

### See also
- [`ds_msp/rig/configs/calib_param.template.yml`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/rig/configs/calib_param.template.yml) — annotated base config
- [`ds_msp/rig/configs/calib_param.keypoints.template.yml`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/rig/configs/calib_param.keypoints.template.yml) — keypoints-reuse config
- [`ds_msp/rig/configs/camera_intrinsics.template.yml`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/rig/configs/camera_intrinsics.template.yml) — initial-intrinsics schema (all models)
- [`docs/learn/`](learn/README.md) — the geometry curriculum (camera models, robust detection, evaluation)


## Independent intrinsics and unsupported frames

Supply independently calibrated lenses through `cam_params_path`. Complete models
initialize frame poses directly; the front end calibrates only missing cameras.
Free mode may refine all lens parameters. Fixed mode requires a model for every
observed camera and preserves every parameter, including high-order distortion.
For recognized explicit model labels, malformed parameter vectors are rejected
rather than reinterpreted.

Capture varied board tilt, distance and peripheral coverage, and validate on held-out
views. Supplying a good seed alone does not preserve that sequence's information
when the rig observations constrain a smaller image region. See
[lens information and frame-pose recovery](learn/rig_intrinsics_and_pose_recovery.md)
for the fixed-intrinsic tradeoff and the assumptions behind joint information retention.

Before optional rejection and after target refinement, the pipeline attempts
model-aware resection of missing or poorly supported frame poses. Accepted poses
must improve whole-frame support and truncated cost without invalidating previously
valid projections. Recovery never drops detections and preserves the chosen
intrinsic policy. `solver.pose_recovery` in `calibration_acceptance.json` records
attempts; `solver.frame_support` independently audits the last returned solve.
An unresolved frame warns and needs inspection. These diagnostics measure training
support, not independent physical accuracy; see the
[interface contract](process/srs/interfaces.md#rig-initialization-and-support-diagnostics-ifc-03).
