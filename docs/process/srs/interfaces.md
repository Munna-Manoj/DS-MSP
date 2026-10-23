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
unlabeled caches written by earlier DS-MSP on OpenCV 4.x). Rig configuration adds `refine_structure`
(default false), `structure_prior_sigma` (target units), `frame_pattern` (regex with a `frame`
group), and `accuracy_gate` (`off` or `refuse`). Refusal gates run before calibration export, and
`calibration_acceptance.json` records the physical acceptance status. Filename-based rig
ingestion uses the trailing frame number (or `frame_pattern`), rejects timestamp-like and
trailing-camera-id names, collisions within a camera, and cameras that share no frame index.
