# ADR-0005 — DS⁺ / EUCM⁺ closed-form-invertible camera models

- **Status:** Partially superseded by [ADR-0010](ADR-0010-mit-relicense-and-eucmplus-removal.md)
  (EUCM⁺ removed; the DS⁺ decision below stands)
- **Deciders:** maintainer
- **Relates to:** ARC-MODELS, FR-MODEL-001..003, NFR-NUM-005
- **Supersedes:** —

## Context

On some wide-FOV lenses the standard Double Sphere (DS) model leaves residual structure that caps
reprojection accuracy. Richer models can fit better, but many achieve it with an inverse that needs
iteration (Newton) or transcendental terms (`atan`), which is awkward for real-time unprojection and
for exact round-trip guarantees.

## Decision

Add two extended models that improve the fit **while keeping a closed-form inverse**:

- **DS⁺** — Double Sphere composed with a division-model term and a small tilt/affine correction.
- **EUCM⁺** — the Enhanced UCM with the analogous extension.

Both implement the same `CameraModel` protocol ([ADR-0002](ADR-0002-protocol-camera-models.md)) with
hand-derived analytic Jacobians ([ADR-0003](ADR-0003-analytic-jacobians.md)) and a closed-form
`project`/`unproject` round-trip (no Newton iteration, no `atan` in the hot path).

## Consequences

**Positive**
- Sub-0.3px reprojection on lenses where DS plateaus, with a fast, exactly-invertible mapping.
- Drop-in across every service because they honour the shared contract.

**Negative / costs**
- More parameters to estimate → calibration must initialise and regularise them well (handled by the
  generic calibration path and its robust seeding).
- Two more models to keep under the full contract + gradient-check suites.

## Verification

- `tests/contract/test_camera_model_contract.py` and `tests/contract/test_gradcheck.py` include
  `dsplus` and `eucmplus`.
- `tests/models/test_ds_model.py` covers >180° FOV half-space validity (NFR-NUM-005).

Covered requirements: **FR-MODEL-001, FR-MODEL-002, FR-MODEL-003, NFR-NUM-005**.

## Alternatives considered

- *Iterative (Newton) inverse for a richer model.* Rejected — breaks closed-form round-trip and adds
  per-pixel cost; closed-form invertibility was a hard requirement.
- *Stay with DS only.* Rejected — insufficient accuracy on the target lenses.
