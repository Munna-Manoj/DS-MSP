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
| IFC-08 | TI Jacinto LDC export | File I/O | distortion-correction mesh for hardware rectification | `ds_msp/ldc.py` | FR-INTEROP-002 |
| IFC-09 | Model-agnostic pose operations | Python API | `solve_pnp`, recommended `solve_pnp_robust`, compatibility `solve_pnp_ransac` over any central `CameraModel` | `ds_msp/ops/pose.py`, `ds_msp/model.py` | FR-OPS-003 |

## Stability & conventions

- **Arrays:** NumPy `float64`; points `(N,3)`, pixels `(N,2)`, bearings unit-norm `(N,3)`; a boolean
  `valid`/visibility mask accompanies projections. Shapes and dtypes are part of the contract and are
  CI-checked (`tests/contract/test_camera_model_contract.py`).
- **The contract seam (IFC-01)** is the central interface: every capability and pipeline depends on
  the protocol, never on a concrete model class ([ADR-0002](../architecture/decisions/ADR-0002-protocol-camera-models.md)).
- **File formats (IFC-06, IFC-08)** are external contracts with third-party tools; their round-trip
  fidelity is verified by the `io` test suite. Treat any change as potentially breaking.
- **Versioning:** the public surface follows SemVer via Conventional Commits + release-please; a
  breaking change to any interface above requires a major bump and a note in `CHANGELOG.md`.

## Internal (non-public) seams

`core`, `geometry`, `detect`, and the `*_math` modules are **internal**: useful to contributors but
not part of the stability promise. Depend on them at your own risk; they may change within a minor
release. The layering rules that keep these internal are enforced per
[ARCHITECTURE.md](../architecture/ARCHITECTURE.md) §5.
