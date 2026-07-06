# ARCHITECTURE — DS-MSP layered stack `[ARC]`

> Architecture description for DS-MSP, in the spirit of ISO/IEC/IEEE 42010. It records the
> structure, the dependency rules that keep it acyclic, and *how those rules are enforced in
> CI*. Significant choices are captured as immutable decision records under
> [`decisions/`](decisions/INDEX.md). Requirements map to components via the `arc_ref` column
> of [`../srs/requirements.csv`](../srs/requirements.csv); components are enumerated in
> [`components.csv`](components.csv).

## 1. Purpose & driving forces

DS-MSP is a NumPy-native platform for wide-field-of-view (fisheye / spherical) camera geometry:
camera **models**, **calibration**, model **conversion**, and downstream **3D** (stereo, two-view
geometry, visual odometry). The architecture is shaped by four forces:

1. **Composability over a frozen monolith.** New camera models, robust kernels, IO formats and
   pipelines are added without editing existing ones — the way a tensor library lets new losses,
   optimizers and datasets interoperate. This is achieved with a shared math core and a single
   interchangeable model contract, *not* with automatic differentiation (see
   [ADR-0002](decisions/ADR-0002-protocol-camera-models.md),
   [ADR-0003](decisions/ADR-0003-analytic-jacobians.md)).
2. **One implementation of each primitive.** Lie-group exp/log, the LM/Schur optimizer, robust
   kernels, PnP/resection and bundle adjustment each live in exactly one place; every higher layer
   reuses them. No duplicated geometry across `calib`, `rig`, `vo`.
3. **Portable, dependency-light solver path.** The math foundation is pure NumPy — OpenCV and SciPy
   are confined to detection / IO / image services so the numerical core stays portable
   ([ADR-0004](decisions/ADR-0004-cv2-scipy-free-foundation.md)).
4. **Acyclic, machine-checked layering.** The dependency rules below are not a convention — they are
   enforced on every PR (§4).

## 2. The layered model (low → high)

A lower layer may never import a higher one. The services split into **two acyclic tiers** — a
deliberately tensor-library-like shape: single-purpose *capabilities* compose into *pipelines*.

```
                         ┌───────────────── pipelines ─────────────────┐
                         │   rig (→calib, geometry)     vo (→mvg)       │   compose capabilities
                         └──────────────────────────────────────────────┘
        ┌──────────────────────── capabilities (mutually independent) ───────────────────────┐
        │   ops      adapt      calib      mvg      stereo                                     │
        └──────────────────────────────────────────────────────────────────────────────────┘
                 ┌──────────── adapters ────────────┐
                 │   detect (OpenCV)     io           │
                 └────────────────────────────────────┘
        ┌──────────────────────── math foundation (cv2/scipy-free) ──────────────────────────┐
        │   models (7 models + *_math, registry)                                              │
        │   geometry (resection · averaging · graph · single-camera BA driver)                │
        │   data (Observation / BoardObs / Object3D / RigState / CalibDataset)                │
        │   core (CameraModel contract · lie · optimize · robust · pinhole)                   │
        └─────────────────────────────────────────────────────────────────────────────────────┘
                              interop: cv.py (OpenCV-compatible API) · ldc.py (TI Jacinto export)
```

- **core** depends on nothing in `ds_msp` and on NumPy + stdlib only.
- **data** imports core only; **geometry** imports core + data only.
- **models/`*_math`** modules are pure NumPy and import nothing from `ds_msp`.
- **capabilities** (`ops`, `adapt`, `calib`, `mvg`, `stereo`) are mutually independent.
- **pipelines** (`rig`, `vo`) may import capabilities downward (`rig → calib`, `vo → mvg`) but never
  each other; capabilities never import a pipeline (no upward edge ⇒ no cycle).

See [ADR-0001](decisions/ADR-0001-layered-capability-pipeline.md) for the rationale behind the
two-tier split.

## 3. Components `[ARC-*]`

Canonical list with dependencies in [`components.csv`](components.csv). Summary:

| ID | Layer | Package | Responsibility |
|----|-------|---------|----------------|
| ARC-CORE | core | `ds_msp/core` | `CameraModel` contract, Lie groups, LM/Schur optimizer, robust kernels, pinhole helper |
| ARC-DATA | data | `ds_msp/data` | Neutral observation / correspondence containers and `CalibDataset` |
| ARC-GEOMETRY | geometry | `ds_msp/geometry` | One resection/PnP, pose averaging, covisibility graph, single-camera BA driver |
| ARC-MODELS | models | `ds_msp/models` | Seven camera models + pure-NumPy math + registry implementing the contract |
| ARC-DETECT | detect | `ds_msp/detect` | OpenCV detection adapters (ChArUco / AprilGrid) → data records |
| ARC-IO | io | `ds_msp/io` | Interop read/write (Kalibr / COLMAP / nerfstudio / MC-Calib) |
| ARC-OPS | ops | `ds_msp/ops` | Model-agnostic services: undistort, multi-chart reproject, PnP |
| ARC-ADAPT | adapt | `ds_msp/adapt` | Model conversion and automatic model selection |
| ARC-CALIB | calib | `ds_msp/calib` | Single-camera intrinsic + stereo extrinsic calibration |
| ARC-MVG | mvg | `ds_msp/mvg` | Two-view geometry on bearing vectors |
| ARC-STEREO | stereo | `ds_msp/stereo` | Wide-FOV stereo depth and rectification |
| ARC-RIG | rig | `ds_msp/rig` | Multi-camera rig calibration pipeline (composes calib + geometry) |
| ARC-VO | vo | `ds_msp/vo` | Monocular visual odometry pipeline (composes mvg) |
| ARC-INTEROP | interop | `ds_msp/cv.py`, `ds_msp/ldc.py` | OpenCV-compatible API and TI Jacinto LDC export |

## 4. The contract seam

Every camera model — `DoubleSphere`, `UCM`, `EUCM`, `KannalaBrandt`, `RadTan`, `OCam`, and the
closed-form-invertible `DSPlus` ([ADR-0005](decisions/ADR-0005-dsplus-eucmplus.md)) —
satisfies one `CameraModel` protocol (`ds_msp/core/contracts.py`): `project`, `unproject`,
`project_jacobian`, parameter (de)serialization. Higher layers depend on the **protocol**, never on
a concrete model class, so any model is a drop-in for any other. This single seam is what makes the
platform composable without autodiff.

## 5. Enforcement (the rules are machine-checked)

The layering is verified on every PR, two independent ways:

- **import-linter** — six contracts in [`pyproject.toml`](../../../pyproject.toml) (`[tool.importlinter]`):
  core is dependency-free; data depends only on core; geometry only on core+data; capabilities are
  mutually independent; capabilities never import a pipeline; pipelines stay independent of each other.
  Run by the `lint + types + layering` CI job (`lint-imports`).
- **`tests/contract/test_independence.py`** — a pure-pytest AST gate mirroring those contracts, plus
  `test_math_foundation_is_cv2_and_scipy_free` (core/data/geometry/models import no `cv2`/`scipy`) and
  isolated-import smoke checks. Holds even if import-linter is not installed.

The model contract itself is gated by `tests/contract/test_camera_model_contract.py` (shapes, dtypes,
unit-norm bearings, round-trip serialization, protocol satisfaction) and the strict analytic-Jacobian
check `tests/contract/test_gradcheck.py` (§ [ADR-0003](decisions/ADR-0003-analytic-jacobians.md)).

These verify NFR-ARCH-001, NFR-ARCH-002, NFR-ARCH-003 (see the SRS).

## 6. Views

- **Module / dependency view:** §2–§3 and `components.csv` (the authoritative `depends_on` graph).
- **Decision view:** [`decisions/INDEX.md`](decisions/INDEX.md) — the accepted ADRs and their scope.
- **Requirements view:** `arc_ref` in [`../srs/requirements.csv`](../srs/requirements.csv) joins each
  FR/NFR to the component that realizes it; the generated
  [`../traceability/TRACEABILITY.md`](../traceability/TRACEABILITY.md) closes the loop to tests.

## 7. Evolution rules

- Adding a **model**, **kernel**, **IO format** or **capability** must not require editing a sibling;
  follow the matching playbook under [`../playbooks/`](../playbooks/).
- A new dependency edge that crosses layers requires an ADR and an update to the import-linter
  contracts + `test_independence.py` — otherwise CI fails by design.
- ADRs are immutable; a reversed decision is a *new* ADR that supersedes the old one.
