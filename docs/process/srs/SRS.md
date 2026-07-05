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
| CON-02 | OpenCV confined to detection / IO adapters | `test_math_foundation_is_cv2_and_scipy_free` |
| CON-03 | Support Python 3.10–3.12 | CI matrix |
| CON-04 | Analytic Jacobians only (no autodiff dependency) | `test_gradcheck.py` |
| CON-05 | Examples run on small (<10 GB) public data on a laptop | `docs/ROADMAP.md` |
| CON-06 | No internal R&D / process content in tracked files | `tools/check_tree_hygiene.py` |
| CON-07 | Releases only via release-please + PyPI OIDC | `.github/workflows/release.yml` |

These map onto the architecture decisions: CON-01/02/04 ↔ ADR-0004/ADR-0003; CON-07 ↔ ADR-0006.

## 4. External interfaces `[IFC]`

The public API surface and external file formats are specified in
[`interfaces.md`](interfaces.md) (IFC-01 … IFC-08): the `CameraModel` protocol, the calibration and
rig APIs, the data containers, the conversion API, the IO formats, the OpenCV-compatible API, and the
TI LDC export.

## 5. Functional requirements `[FR]`

28 functional requirements, grouped by area; each row in [`requirements.csv`](requirements.csv) names
its architecture component and the test that verifies it. Areas:

- **MODEL** (FR-MODEL-001..005) — project / unproject / analytic Jacobians / one contract / serialize.
- **CALIB** (FR-CALIB-001..004) — bundle-adjustment calibration for any model; robust PnP seeding;
  stereo relative pose; board detection.
- **RIG** (FR-RIG-001..005) — multi-camera rig calibration (intrinsics + extrinsics + object poses);
  single-file `calib_param.yml` entry; per-camera intrinsics handling (load / validate / fix /
  convert+warn / scratch); detect-once keypoints reuse; MC-Calib drop-in config & intrinsics interop.
- **MVG** (FR-MVG-001..003) — two-view pose + triangulation; RANSAC on Sampson; angular BA refine.
- **STEREO** (FR-STEREO-001..002) — sphere-sweep depth; spherical epipolar rectification.
- **OPS** (FR-OPS-001..003) — undistort; multi-chart reproject; PnP on bearings.
- **ADAPT** (FR-ADAPT-001..002) — model conversion without images; automatic model selection.
- **IO** (FR-IO-001..004) — Kalibr, COLMAP, nerfstudio, MC-Calib.
- **VO** (FR-VO-001..002) — monocular trajectory; Sim(3) ATE/RPE evaluation.
- **INTEROP** (FR-INTEROP-001..002) — OpenCV-compatible API; TI Jacinto LDC export.

## 6. Non-functional requirements `[NFR]`

11 non-functional requirements in [`requirements.csv`](requirements.csv):

- **NUM** — analytic Jacobians ≤1e-6 vs FD (NFR-NUM-001); KB/RadTan ~1e-13 vs OpenCV (NFR-NUM-002);
  sub-pixel undistort/reproject round-trip (NFR-NUM-003); **real-data calibration parity within stated
  tolerance (NFR-NUM-004, release-gated)**; >180° FOV half-space validity (NFR-NUM-005).
- **ARCH** — strictly layered & acyclic (NFR-ARCH-001); cv2/scipy-free foundation (NFR-ARCH-002);
  every model satisfies the contract (NFR-ARCH-003).
- **PORT** — runs on Python 3.10/3.11/3.12 (NFR-PORT-001).
- **REPRO** — deterministic via fixed seeds (NFR-REPRO-001).
- **PRIV** — no internal R&D / process content in tracked files (NFR-PRIV-001).

## 7. Verification approach

Each requirement carries a **verify_method** (a test path / CI workflow) in the CSV. Test levels,
entry/exit criteria, and the synthetic→real-data gate are defined in the
[QA & V&V plan](../quality/QA_VV_PLAN.md). Two requirements classes are **release-gated** (ADR-0006):
they must have *both* a synthetic and a `realdata` test linked and green before a release —
`tools/check_traceability.py --release` enforces it.

## 8. Traceability

The full chain is `STK → FR/NFR ↔ ARC ↔ code module ↔ test (↔ ISS ↔ REL)`. Requirement→test links are
discovered from `@pytest.mark.req(...)` markers co-located with the tests, so they cannot silently
drift. `tools/check_traceability.py --check` fails CI on malformed/duplicate IDs, orphan requirements,
dangling links, or a matrix out of sync; the rendered matrix is
[`../traceability/TRACEABILITY.md`](../traceability/TRACEABILITY.md).
