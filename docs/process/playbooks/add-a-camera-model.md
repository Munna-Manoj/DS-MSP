# Playbook — Add a camera model `[PBK]`

> End-to-end recipe for adding a new camera model. Honours [ADR-0002](../architecture/decisions/ADR-0002-protocol-camera-models.md)
> (one contract) and [ADR-0003](../architecture/decisions/ADR-0003-analytic-jacobians.md) (analytic
> Jacobians, FD-checked). Area: **MODEL** (ARC-MODELS).

## When

You need a projection model that isn't one of the existing seven (DS, UCM, EUCM, KB, RadTan, OCam,
DS⁺).

## Steps

1. **Requirement.** Add an FR row to [`requirements.csv`](../srs/requirements.csv) (area MODEL,
   `arc_ref=ARC-MODELS`) — or confirm FR-MODEL-001..003 already cover it. If the model introduces a
   genuinely new design choice (e.g. a new invertibility strategy), write an ADR.
2. **Math, pure NumPy.** Implement the numerics in `ds_msp/models/<name>_math.py` — NumPy only, no
   `ds_msp` imports, no `cv2`/`scipy` (CON-01/02, NFR-ARCH-002). Provide `project`, `unproject`, and
   the analytic point + parameter Jacobians.
3. **Model class.** Add `ds_msp/models/<name>.py` implementing the `CameraModel` protocol
   (`ds_msp/core/contracts.py`): `project`, `unproject`, `project_jacobian`, parameter
   (de)serialization, and a `sample()` classmethod returning a representative instance for the
   contract tests.
4. **Register.** Add the model to the registry so it can be constructed by name and is picked up by
   `adapt` autoselect.
5. **Tests.**
   - It is auto-covered by `tests/contract/test_camera_model_contract.py` once registered (shapes,
     dtypes, unit bearings, serialize round-trip, protocol).
   - Add it to the `_FACTORIES` list in `tests/contract/test_gradcheck.py` so the analytic Jacobian is
     FD-checked at ≤1e-6 (`-m jac`).
   - Add a model-specific test for any special behaviour (e.g. >180° FOV validity, like
     `tests/models/test_ds_model.py`).
   - Mark suites with `@pytest.mark.req(...)`.
6. **Docs.** Mention it in `docs/MULTI_MODEL.md`; update [`interfaces.md`](../srs/interfaces.md) if the
   public surface changes.
7. **Verify & gate.** `ruff`/`mypy`/`lint-imports`/`pytest` green; `check_traceability.py --check` green
   (the new FR is linked); satisfy the [Definition of Done](../quality/DEFINITION_OF_DONE.md).
8. **Release.** `feat:` commit; release-please ships it in the next minor.

## Checklist

- [ ] `<name>_math.py` pure NumPy; `<name>.py` implements the contract + `sample()`
- [ ] registered; conversion/autoselect aware
- [ ] contract suite + `-m jac` gradcheck green; `@pytest.mark.req` added
- [ ] docs + DoD + traceability green
