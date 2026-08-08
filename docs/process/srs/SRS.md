# Software Requirements Specification — DS-MSP `[SRS]`

> Standards-informed after ISO/IEC/IEEE 29148. The **canonical, machine-checked** requirements live
> in [`requirements.csv`](requirements.csv) (one row per FR/NFR with area, architecture ref, code
> module, verification method, status, release gate). This document gives the scope, stakeholders,
> constraints, and the verification narrative that the CSV cannot. The two are kept in sync by
> `tools/check_traceability.py` and the generated
> [`../traceability/TRACEABILITY.md`](../traceability/TRACEABILITY.md).

## 1. Introduction & scope

DS-MSP is a NumPy-native platform for **wide-field-of-view (fisheye / spherical) camera geometry**:
camera models, single-camera and multi-camera-rig calibration, model conversion, and downstream 3D
(two-view geometry, wide-FOV stereo, monocular visual odometry), plus interop with the SLAM/SfM
ecosystem (Kalibr, COLMAP, nerfstudio, MC-Calib) and an OpenCV-compatible drop-in API.

**In scope:** the camera-model contract and seven models; calibration and conversion; the 3D
capabilities and pipelines above; IO formats; embedded export (TI Jacinto LDC mesh).
**Out of scope:** dense multi-view reconstruction, learned/neural calibration, GUI tooling, and any
internal research-process tooling (kept local and out of the tracked tree — CON-06).

## 2. Stakeholders `[STK]`

Canonical list in [`stakeholders.csv`](stakeholders.csv). The driving stakeholders:

- **STK-01 Library users** (SLAM/SfM/robotics) — accuracy, correct wide-FOV geometry, interop.
- **STK-02 Embedded / robotics engineers** — TI LDC export, real-time pose, portability.
- **STK-03 CV practitioners** — IO fidelity; convert between models without re-shooting.
- **STK-04 Learning practitioners** — runnable curriculum on small public data.
- **STK-05 Contributors** (human and AI) — clear playbooks, traceability, CI gates, a Definition of Done.
- **STK-06 Maintainer / release owner** — no unverified release; no internal-process leakage; auditability.

## 3. Constraints `[CON]`

Canonical list in [`constraints.csv`](constraints.csv); each is verified, not assumed:

| ID | Constraint | Verified by |
|----|-----------|-------------|
| CON-01 | Math foundation depends only on NumPy + stdlib | `test_independence.py` |
| CON-02 | OpenCV and SciPy excluded from the math foundation | `test_math_foundation_is_cv2_and_scipy_free` |
| CON-03 | Support Python 3.10–3.12 | CI matrix |
| CON-04 | Analytic Jacobians only (no autodiff dependency) | `test_gradcheck.py` |
| CON-05 | Examples run on small (<10 GB) public data on a laptop | `docs/ROADMAP.md` |
| CON-06 | No internal R&D / process content in tracked files | `tools/check_tree_hygiene.py` |
| CON-07 | Releases only via release-please + PyPI OIDC | `.github/workflows/release.yml` |

These map onto the architecture decisions: CON-01/02/04 ↔ ADR-0004/ADR-0003; CON-07 ↔ ADR-0006.

## 4. External interfaces `[IFC]`

The public API surface and external file formats are specified in
[`interfaces.md`](interfaces.md) (IFC-01 … IFC-09): the `CameraModel` protocol, calibration and rig
APIs, data containers, conversion and camera-agnostic pose operations, IO formats, the
OpenCV-compatible API, and TI LDC export.

## 5. Functional requirements `[FR]`

Functional requirements are grouped by area; each canonical row in
[`requirements.csv`](requirements.csv) names its architecture component and verification method.
The registry, rather than a duplicated aggregate count here, is authoritative. Areas:

- **CORE** — reusable robust optimization and other pipeline-neutral capabilities.
- **MODEL** — project / unproject / analytic Jacobians / one contract / serialization.
- **CALIB** — model-agnostic bundle adjustment, bearing-native pose initialization, board
  detection, configuration, reporting, and stereo calibration.
- **RIG** — multi-camera calibration, non-overlapping-board fusion, robust estimation, diagnostics,
  trust reporting, configuration, packaging, and live visualization.
- **MVG** — two-view pose and triangulation, robust sampling, and full-sphere bundle refinement.
- **STEREO** — sphere-sweep depth and spherical epipolar rectification.
- **OPS** — undistortion, multi-chart reprojection, and clean/robust PnP on bearings.
- **ADAPT** — model conversion without images and automatic model selection.
- **IO** — Kalibr, COLMAP, nerfstudio, and MC-Calib interchange.
- **VO** — monocular trajectory estimation and Sim(3) ATE/RPE evaluation.
- **INTEROP** — OpenCV-compatible API and TI Jacinto LDC export.

## 6. Non-functional requirements `[NFR]`

Non-functional requirements are canonical rows in [`requirements.csv`](requirements.csv); the
registry is authoritative as the set grows:

- **NUM** — Jacobian accuracy, external numerical parity, round-trip and calibration accuracy,
  full-sphere validity, robust convergence, and deterministic conversion.
- **ARCH** — a strictly layered, acyclic system; a NumPy-native math foundation; camera-model
  contract conformance.
- **PORT** — the supported Python runtime matrix.
- **REPRO** — deterministic tests and validations with explicit seeds.
- **PRIV** — no internal R&D/process content in tracked files.
- **PERF** — measured, regression-tested performance and parallelism behavior.
- **DOCS** — strict site builds, executable source-backed examples, and the top-level docs allowlist.

## 7. Verification approach

Each requirement carries a **verify_method** (a test path / CI workflow) in the CSV. Test levels,
entry/exit criteria, and the synthetic→real-data gate are defined in the
[QA & V&V plan](../quality/QA_VV_PLAN.md). Requirements whose canonical row has
`release_gated=yes` are governed by ADR-0006: they must have *both* synthetic and `realdata`
coverage linked, and the linked real-data tests must actually execute and pass before a release.
`tools/check_traceability.py --release` enforces the structural coverage half of that gate.

## 8. Traceability

The full chain is `STK → FR/NFR ↔ ARC ↔ code module ↔ test (↔ ISS ↔ REL)`. Requirement→test links are
discovered from `@pytest.mark.req(...)` markers co-located with the tests, so they cannot silently
drift. `tools/check_traceability.py --check` fails CI on malformed/duplicate IDs, orphan requirements,
dangling links, or a matrix out of sync; the rendered matrix is
[`../traceability/TRACEABILITY.md`](../traceability/TRACEABILITY.md).
