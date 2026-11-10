# External Interfaces `[IFC]`

> The public API surface and external file formats DS-MSP commits to (ISO/IEC/IEEE 29148 §9.4). These
> are the seams other software depends on; changes to them follow the change & release process
> ([../management/CHANGE_RELEASE_MGMT.md](../management/CHANGE_RELEASE_MGMT.md)) and Conventional
> Commits (a breaking change → major version).

| ID | Interface | Kind | Surface | Realized by | Reqs |
|----|-----------|------|---------|-------------|------|
| IFC-01 | `CameraModel` protocol | Python API | `project`, `unproject`, `project_jacobian`, (de)serialize | `ds_msp/core/contracts.py`, `ds_msp/models/*` | FR-MODEL-001..005 |
| IFC-02 | Single-camera calibration | Python API | `calibrate(model_cls, dataset, …) → params, poses, residuals` | `ds_msp/calib` | FR-CALIB-001..003 |
| IFC-03 | Rig calibration | Python API | rig scenario → intrinsics + extrinsics + object poses | `ds_msp/rig` | FR-RIG-001 |
| IFC-04 | Data containers | Python API | `Observation`, `BoardObs`, `Object3D`, `RigState`, `CalibDataset` | `ds_msp/data` | FR-CALIB-*, FR-RIG-001 |
| IFC-05 | Model conversion / autoselect | Python API | convert params between any two models; pick best-fitting | `ds_msp/adapt` | FR-ADAPT-001..002 |
| IFC-06 | Interop file formats | File I/O | Kalibr camchain YAML, COLMAP, nerfstudio `transforms.json`, MC-Calib | `ds_msp/io` | FR-IO-001..004 |
| IFC-07 | OpenCV-compatible API | Python API | drop-in `cv`-style calls for wide-FOV models | `ds_msp/cv.py` | FR-INTEROP-001 |
| IFC-08 | TI Jacinto LDC export | File I/O | distortion-correction mesh for hardware rectification, from any `CameraModel` | `ds_msp/ldc.py` | FR-INTEROP-002 |
| IFC-09 | Model-agnostic pose operations | Python API | `solve_pnp`, recommended `solve_pnp_robust`, compatibility `solve_pnp_ransac` over any central `CameraModel` | `ds_msp/ops/pose.py`, `ds_msp/model.py` | FR-OPS-003 |
| IFC-10 | Isaac Sim camera LUT export | File I/O + CLI | `OmniLensDistortionLutAPI` EXRs, manifest, Camera USDA | `ds_msp/isaac_sim`, `ds-msp lut` | FR-INTEROP-003 |
| IFC-11 | Rig run manifest | File I/O + CLI | strict-JSON `calibration_manifest.json`; immutable-input verification and replay into a new output directory | `ds_msp/rig/provenance.py`, `tools/replay_rig_manifest.py` | NFR-REPRO-002 |
| IFC-12 | Rig accuracy profile and report | File I/O + Python API + CLI | strict versioned profile JSON; schema-versioned all-pair SE(3), exact ROI displacement, and explicit evaluation status in `calibration_report.json` | `ds_msp/rig/accuracy.py`, `ds_msp/rig/report.py`, `ds-msp-calibrate-rig --init-accuracy-profile` | FR-RIG-024 |
| IFC-13 | DS-MSP Studio | CLI + HTTP JSON API | `ds-msp studio --root DIR` / `--open DIR`, `ds-msp-calibrate-rig --studio`; `GET /api/runs`, `/api/run?id=`, `/api/run/diagnostics?id=&cam=`, `/api/run/file?id=&name=`, `/api/schema/rig-config`; run document schema `ds-msp-studio-run` v1 | `ds_msp/studio`, `ds_msp/rig/config_schema.py`, `ds_msp/rig/cli.py`, `web/` | FR-UI-001..006 |

## Stability & conventions

- **Arrays:** NumPy `float64`; points `(N,3)`, pixels `(N,2)`, bearings unit-norm `(N,3)`; a boolean
  `valid`/visibility mask accompanies projections. Shapes and dtypes are part of the contract and are
  CI-checked (`tests/contract/test_camera_model_contract.py`).
- **The contract seam (IFC-01)** is the central interface: every capability and pipeline depends on
  the protocol, never on a concrete model class ([ADR-0002](../architecture/decisions/ADR-0002-protocol-camera-models.md)).
- **File formats (IFC-06, IFC-08, IFC-10, IFC-11, IFC-12)** are external contracts with third-party tools; their round-trip
  fidelity is verified by the `io` test suite. Treat any change as potentially breaking.
- **Versioning:** the public surface follows SemVer via Conventional Commits + release-please; a
  breaking change to any interface above requires a major bump and a note in `CHANGELOG.md`.

## Internal (non-public) seams

`core`, `geometry`, `detect`, and the `*_math` modules are **internal**: useful to contributors but
not part of the stability promise. Depend on them at your own risk; they may change within a minor
release. The layering rules that keep these internal are enforced per
[ARCHITECTURE.md](../architecture/ARCHITECTURE.md) §5.

Calibration measurement extensions (FR-CALIB-009, FR-RIG-025): observation containers accept
optional finite symmetric positive-definite `(N,2,2)` pixel covariances. Solvers whiten both
residuals and Jacobians, while reports retain pixel units. MC-Calib keypoint caches carry these
in an optional parallel `pixel_covariances` sequence, and a top-level `pixel_convention`
(`pixel_center`; readers accept `keypoints_pixel_convention: opencv4_charuco` to migrate
unlabeled caches written by earlier DS-MSP on OpenCV 4.x). Rig configuration adds
`structure_mode` (`rigid` default, `bend`, `free`, `off`; legacy `refine_structure: true` means
`free`), `structure_prior_sigma` (target units, free mode), `frame_pattern` (regex with a
`frame` group), `accuracy_gate` (`off` or `refuse`), `uncertainty_gate` (`off`, `warn`, `refuse`)
with `max_rotation_sigma_deg` / `max_centre_sigma`, `convergence_gate` (`off`, `warn`,
`refuse`; judges the final BA stage), `coverage_gate` (`off`, `warn`, `refuse`) with
`min_camera_coverage` (default 0.75; the share of each camera's corners within the GNC-TLS inlier
band), `reject_outliers`, `lens_gate` and `crossval_gate` (`off`, `warn`, `refuse`; `crossval_gate` is `off`
by default and under production) with
`max_lens_sigma_px` (default `noise_bound`, else 1 px), and `profile`
(`default` or `production`; production sets the audit, uncertainty, convergence and lens gates to
`refuse`). MC-Calib `boards_index` sets marker-id offsets. Unknown keys warn
(production refuses them). Refusal gates run before calibration export, and
`calibration_acceptance.json` records the physical acceptance status, per-pair extrinsic 1σ,
each BA stage's solver termination status, front-end fallbacks, hand-eye links, the rejection
ledger, config warnings, the observability audit's findings (`observability`), each camera's lens
1σ in pixels with the parameters left at an optimizer bound (`lens`), and the block-split
frame cross-validation of each lens (`crossval`), plus `diagnostics`: per-camera outlier shares (`outliers`), per-board flatness in target units (`board_flatness`) and the paths of `diagnostics/residuals.npz` (signed residual per observed corner) and `diagnostics/lens_sigma.npz` (each camera's 24x18 lens 1σ grid; FR-RIG-031). A numerical failure in any BA stage always refuses export. Filename-based rig ingestion uses the trailing frame
number (or `frame_pattern`), rejects timestamp-like and trailing-camera-id names, collisions
within a camera, and cameras that share no frame index.


## Rig initialization and support diagnostics (IFC-03)

Complete `init_cameras` / `cam_params_path` models initialize PnP directly; only
missing cameras are calibrated from scratch. `fix_intrinsics=True` (configuration
`fix_intrinsic: true`) requires every observed camera and holds its full parameter
vector. Free mode initializes from the supplied models but may refine all their
parameters. A recognized explicit model label with a malformed parameter vector
raises `ValueError`; legacy unnamed files retain distortion-type/vector inference.
Unknown model strings still follow the importer's existing unnamed-model handling. Scenario CLI
`--fix-intrinsics` reads native models from `Results/calibrated_cameras_data.yml`
and refuses missing camera models.

The additive acceptance-file fields below appear under `solver` (FR-RIG-029):

- `pose_recovery`: list of attempted object-frame replacements, with `stage`,
  integer `object_id` / `frame_id`, `threshold_px`, `before` / `after` summaries,
  boolean `accepted`, `reason`, candidate/failure counts and optional `source`.
  Reasons are `recovered`, `no_supported_candidate`, or `too_few_corners`.
  Summaries contain `n_observed`, `n_valid`, `n_supported`, `support_fraction`,
  and `truncated_cost` (sum of squared pixel errors capped at the squared band).
- `frame_support`: final-state audit of every nonempty observed object-frame,
  with IDs, the same summary fields, `threshold_px`, `min_support`, and boolean
  `supported`. Missing poses/models and invalid projections count as unsupported.

Support pools corners across cameras. The pipeline uses 75% and the existing
`3.03 * noise_bound` band (3.03 pixels by default). Recovery does not discard
observations; the separate optional rejection stage can do so, and its removals
remain in the rejection ledger. Final support describes the observations retained
at the end. Unresolved frames warn even if an earlier recovery succeeded; these
fields are diagnostic and do not replace physical acceptance or uncertainty gates.

## DS-MSP Studio (IFC-13)

`ds-msp studio` needs the `ui` extra (`pip install "ds-msp[ui]"`). Options: `--root DIR`
(repeatable; default the current folder), `--open DIR` (serve that result folder's parent and
link to its page), `--host` (default `127.0.0.1`), `--port` (default 8765; the next free port is
used when it is taken), `--token`, `--no-browser`, `--max-depth` (default 6). On any host other
than localhost a token is generated when none is given, and the printed URL carries it.

`ds-msp-calibrate-rig` ends a saved run with `view in DS-MSP Studio:  ds-msp studio --open
"<save_path>"`. With `--studio` (and `--studio-port`, default 8765) it then runs that command in
a child process until Ctrl+C and exits with the calibration's own code; without the `ui` extra it
prints the install command instead. The library never imports `ds_msp.studio`.

`--read-only` leaves out everything that writes: Studio then only shows results.

The API is versioned by `api_version` in `GET /api/health`, which also reports `can_run`
(false under `--read-only`). Reading results:

- `GET /api/runs`: one summary per result folder (`id`, `path`, `created_utc`, `n_cameras`,
  `models`, `fit_level`, `median_px`, `p95_px`, `solver_status`, `n_weak`, `lens_failed`).
  A run id is `<root index>:<folder path relative to that root>`.
- `GET /api/run?id=`: the run document, `{"schema": "ds-msp-studio-run", "schema_version": 1,
  "summary", "report", "acceptance", "manifest", "files"}`. `report` is
  `calibration_report.json` unchanged. `acceptance` is `calibration_acceptance.json` without
  `solver.frame_support` (replaced by `frame_support_count`) and without the per-pair covariance
  matrices. Non-finite numbers are `null`.
- The run document also carries `verdict` (FR-UI-004: `level` of `accept`, `check`, `improve`
  or `reject`, `headline`, per-camera `cameras`, `rig_reasons`, worst-first `actions` with
  `cams` and `evidence`) and `has_diagnostics`.
- `GET /api/run/diagnostics?id=&cam=`: one camera's evidence from the FR-RIG-031 files
  (FR-UI-005): `residuals` (at most 4000 arrows, every out-of-band corner kept), `worst_frames`,
  `pattern` (answer, error by distance from the centre, 6 x 4 region means), `lens` (the 24 x 18
  grid and its `zones`), `outliers`. Files outside the result folder are ignored.
- `GET /api/run/file?id=&name=`: one of the known result files, by name only.
- `GET /api/schema/rig-config`: the JSON Schema of `calib_param.yml` (FR-UI-002).
\n
A result on its own capture (FR-UI-009):

- `GET /api/run/scene?id=`: the target's boards (points, outline) and its 4x4 pose in every
  frame, in the reference camera's frame; `photos` says how photos are matched to frames.
- `GET /api/run/frames?id=&cam=` and `GET /api/run/frame?id=&cam=&frame=`: per-photo error
  summaries, and one photo's corners with the step to their predicted position.
- `GET /api/run/photo?id=&cam=&frame=&w=`: that photo as a JPEG no wider than `w`; 404 unless
  it lies under a root.
- `GET /api/run/pair?id=&a=&b=`: distance, position and yaw, pitch, roll of camera `b` seen
  from `a`, with the recorded 1-sigma.
- `GET /api/run/task?id=&a=&b=&distance=&pixel_noise=`: 1-sigma uncertainty, sideways and in
  depth, of a point the two cameras triangulate, and the same over a range of distances.

Other starting points and repeats (FR-UI-010):

- `GET /api/source/config?path=`, `/api/source/keypoints?path=`, `/api/source/intrinsics?path=`:
  a `calib_param.yml` on disk as form values, and summaries of a saved-detections file and of
  a lens file. The path must lie under a root.
- `GET /api/run/fingerprint?id=`: what the run's manifest records. `GET
  /api/run/fingerprint/check?id=` re-hashes its inputs and lists those that changed.
- `POST /api/jobs/repeat` `{id, root, name}`: runs the recorded config and overrides into a new
  folder. The job's document then carries `repeat_of` and, once done, `repeat` (`same` and one
  row per compared digest or file).

Setting up and running a calibration (FR-UI-007, FR-UI-008; absent under `--read-only`):

- `GET /api/browse?ref=`: one folder under the roots, with its subfolders (image count, whether
  it is a result) and its `.yml`/`.yaml`/`.json` files; without `ref`, the roots.
- `POST /api/photos/scan` `{root_path, cam_prefix?, frame_pattern?}`: the cameras found in the
  layout the calibration reads, image sizes, shared frame count and `problems`.
- `POST /api/photos/check-board` `{values}`: detects the configured board in up to 4 photos per
  camera.
- `POST /api/config/check` `{values}`: checks form values against the schema, the roots and
  the CLI's own loader; returns `ok`, `problems`, `warnings` and the `yaml` that would be written.
- `POST /api/config/import` `{text}` and `GET /api/config?id=`: the schema keys set in a
  `calib_param.yml` (uploaded text, or the one in a result folder), with `unknown` keys listed.
- `POST /api/jobs` `{root, name, values}`: creates `<root>/<name>/`, writes `calib_param.yml`
  there and queues `python -m ds_msp.rig.cli --config <that file> --no-webviewer --progress-file
  ...`. `GET /api/jobs`, `GET /api/job?id=` (state, progress summary, last output lines) and
  `POST /api/job/cancel` `{id}` follow and stop it. States: `queued`, `running`, `done`,
  `failed`, `cancelled`, `interrupted`. Jobs run one at a time.

`values` holds only the keys the user set; a key left out keeps the loader's default, so a
`profile` can still raise it. Path values must lie under a root. A POST must carry the header
`x-dsmsp-studio: 1` and, when the browser sends `Origin`, come from the server's own address;
when Studio is bound to a local address it answers only requests addressed to `127.0.0.1`,
`localhost` or `::1`.
