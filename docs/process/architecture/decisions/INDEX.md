# Architecture Decision Records `[ADR]`

Immutable records of architecturally significant decisions (ISO/IEC/IEEE 42010 rationale). Each ADR
is **accepted and frozen at creation** — to change a decision, write a new ADR that *supersedes* the
old one (the old file stays, with its Status updated to `Superseded by ADR-NNNN`). IDs are
zero-padded and monotonic (`ADR-NNNN`); CI checks this index stays complete and in order
(`tools/check_traceability.py`).

| ID | Title | Status | Drivers |
|----|-------|--------|---------|
| [ADR-0001](ADR-0001-layered-capability-pipeline.md) | Two-tier layered architecture: capabilities compose into pipelines | Accepted | Composability, acyclic reuse |
| [ADR-0002](ADR-0002-protocol-camera-models.md) | One `CameraModel` protocol for all interchangeable models | Accepted | Drop-in model substitution |
| [ADR-0003](ADR-0003-analytic-jacobians.md) | Hand-derived analytic Jacobians, finite-difference-checked (no autodiff) | Accepted | Speed, stability, portability |
| [ADR-0004](ADR-0004-cv2-scipy-free-foundation.md) | The math foundation is cv2/scipy-free | Accepted | Portable solver path |
| [ADR-0005](ADR-0005-dsplus-eucmplus.md) | DS⁺ / EUCM⁺ closed-form-invertible camera models | Partially superseded by ADR-0010 | Sub-0.3px fit with a closed-form inverse |
| [ADR-0006](ADR-0006-synthetic-real-release-gate.md) | Synthetic-then-real-data release gate | Accepted | No public release without real-data validation |
| [ADR-0007](ADR-0007-deterministic-convert-seeding.md) | Deterministic shape-parameter sweep in model conversion | Accepted | Reproducible, exact self-conversion (no restart lottery) |
| [ADR-0008](ADR-0008-noncommercial-engine-scope.md) | Noncommercial license covers the robust calibrate/convert engine, not just the Plus models | Superseded by ADR-0010 | Protect the real IP, not just published math |
| [ADR-0009](ADR-0009-board-protocol.md) | One `Board` protocol unifies checkerboard / ChArUco / AprilGrid for single-camera calibration | Accepted | Config-driven, board-agnostic single-camera calibration |
| [ADR-0010](ADR-0010-mit-relicense-and-eucmplus-removal.md) | Remove EUCM⁺; relicense the whole project to plain MIT | Accepted | Simpler model surface (measured, not assumed); permissive adoption over engine protection |
| [ADR-0011](ADR-0011-rig-multiobject-merge.md) | Multi-object board fusion + merge for non-overlapping rigs | Accepted | Fix confirmed real-data bug: a rig whose cameras share no board co-observation silently calibrated as a smaller, wrong rig |
| [ADR-0012](ADR-0012-docs-top-level-allowlist.md) | Positive allowlist for top-level docs/ entries, enforced in CI | Accepted | A new top-level docs/ file with no structural check could ship to the public site unnoticed; a content-only pattern scan can miss unfamiliar phrasing |
| [ADR-0013](ADR-0013-rig-gross-outlier-reporting-and-gate.md) | Robust reporting for gross-outlier board detections, with an opt-in hard-drop gate | Accepted | A correctly down-weighted blunder no longer misreads as a failed calibration; MC-Calib's `ransac_threshold` hard-drop stays available as an explicit opt-in |

> The first six ADRs are **retrofits**: they record decisions already embodied in the codebase, so
> the governance system is demonstrated against real architecture from day one. Adoption date is the
> date the record was written, not the date the code first shipped. **ADR-0007 onward** are decisions
> recorded as they are made (ADR-0007: a convert-robustness fix found by real-data study).
