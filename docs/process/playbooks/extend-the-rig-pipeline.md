# Playbook — Extend the multi-camera rig pipeline `[PBK]`

> Recipe for adding or changing behaviour in the **rig** pipeline (`ds_msp/rig`) — a new
> `calib_param.yml` field, an intrinsics-handling rule, a keypoints/IO interop feature, or a new
> stage in the calibrate→reconstruct→bundle-adjust flow — while keeping the governance chain
> (`FR-RIG-* ↔ ARC-RIG ↔ code ↔ test ↔ release`) intact.
>
> For **using** the rig to calibrate a real capture (data layout, config fields, running it) see the
> user guide [`docs/RIG_CALIBRATION_GUIDE.md`](../../RIG_CALIBRATION_GUIDE.md). For a brand-new
> top-level pipeline or capability, use [add-a-pipeline-capability](add-a-pipeline-capability.md)
> instead. This playbook is for evolving the *existing* rig pipeline.

## Where things live

`rig` is a **pipeline** tier: it composes capabilities downward (`calib`, `detect`, `geometry`,
`models`, `data`) and **never imports another pipeline** ([ADR-0001](../architecture/decisions/ADR-0001-layered-capability-pipeline.md)).
Modules are named by their role:

| Module | Role (MC-Calib analogue) |
|--------|--------------------------|
| `rig/calib_param.py` | single-file `calib_param.yml` entry — parse + drive the whole run (`calibrate.cpp`/`McCalib` entry) |
| `rig/pipeline.py` | scenario-level orchestration: front-end choice, run, save (`calibrate_scenario`) |
| `rig/calibrate.py` | core rig calibration: init + staged bundle adjustment (`calibrate_rig`) |
| `rig/bundle.py` | the bundle-adjustment assembler / Schur solver (Ceres-equivalent) |
| `rig/reconstruct.py` | fused multi-board object reconstruction (`calibrate3DObjects`) |
| `rig/pose_init.py`, `rig/handeye.py`, `rig/averaging.py`, `rig/graph.py`, `rig/extrinsics.py` | extrinsics init helpers (`InitializationHelper`, `Graph`) |
| `io/mccalib.py` | MC-Calib read/write interop (governed by `FR-IO-004`) |

The governed rig capabilities: `FR-RIG-002` config-driven entry, `FR-RIG-003` intrinsics handling
(load / validate / fix / convert+warn / scratch), `FR-RIG-004` detect-once keypoints reuse,
`FR-RIG-005` MC-Calib drop-in config & intrinsics interop, all under `FR-RIG-001` (the umbrella
rig requirement, release-gated on real-data validation).

## Steps

1. **Requirement.** Map the change to an existing `FR-RIG-*` row in
   [`requirements.csv`](../srs/requirements.csv), or add a new `FR-RIG-00N`. If it adds a new
   `calib_param.yml` key or output file, it is `FR-RIG-002`/`FR-RIG-005`; an intrinsics rule is
   `FR-RIG-003`; a detection/reuse change is `FR-RIG-004`. Keep `arc_ref=ARC-RIG`.
2. **Respect the tier.** Implement under `ds_msp/rig/`; import only **downward** (capabilities and
   the math foundation), never another pipeline. If two rig modules start needing each other's
   internals, push the shared part **down** into `geometry`/`core`. The import-linter contract
   *"pipelines compose capabilities but stay independent of each other"* enforces this.
3. **Keep MC-Calib parity.** Any new config key must either be a real MC-Calib key (honoured) or a
   clearly-marked **DS-MSP extension** (like `camera_models` / `object_path`). Document it in
   [`ds_msp/rig/configs/calib_param.template.yml`](../../../ds_msp/rig/configs/calib_param.template.yml)
   — the template is CI-guaranteed to exist and parse (`test_base_template_exists_and_parses`), and
   ships as real package data (`ds-msp-calibrate-rig --init-config` works from `pip install ds-msp`
   alone, no repo clone — `check_packaging.py` guards this). Output stays in MC-Calib's schema so
   files round-trip both directions (`FR-IO-004`).
4. **Templates are part of the contract.** If you touch config or intrinsics handling, update the
   matching template (`ds_msp/rig/configs/calib_param.template.yml`,
   `ds_msp/rig/configs/calib_param.keypoints.template.yml`,
   `ds_msp/rig/configs/camera_intrinsics.template.yml`) and its `--init-*` CLI flag in
   [`ds_msp/rig/cli.py`](../../../ds_msp/rig/cli.py) (`scripts/calibrate_rig.py` is now just a
   thin wrapper around it, kept for git-clone convenience). A shipped, parseable template per
   capability is a tested guarantee, not a courtesy.
5. **Tests.** Add `integration`-level tests in `tests/rig/test_calib_param.py` (or the closest rig
   test module) and mark each with `@pytest.mark.req("FR-RIG-...")`. The marker *is* the
   bidirectional REQ↔test link the traceability tool discovers. Real-data accuracy claims go in a
   `realdata`-marked test and make the requirement `release_gated` ([ADR-0006](../architecture/decisions/ADR-0006-synthetic-real-release-gate.md)).
6. **Docs.** Update the user guide [`docs/RIG_CALIBRATION_GUIDE.md`](../../RIG_CALIBRATION_GUIDE.md)
   (fields/behaviour table) and, if a public entry point changed, [`interfaces.md`](../srs/interfaces.md).
7. **Verify & gate.** `pytest -m "not slow"` (and the slow rig suite for math changes), `ruff`,
   `lint-imports` (6 contracts kept), `python tools/check_traceability.py --check`,
   `python tools/check_tree_hygiene.py`, local `publish_guard.py`. Regenerate the matrix with
   `check_traceability.py --write` when you add/relink a requirement.
8. **Release.** `feat:`/`fix:` commit. The rig is held for 0.8.0; promoting it flips `FR-RIG-001`
   to `implemented` once its `realdata` validation gate is green.

## Checklist

- [ ] mapped to an `FR-RIG-*` row (added one if new); `arc_ref=ARC-RIG`
- [ ] rig tier respected — only downward imports; `lint-imports` clean
- [ ] MC-Calib parity kept; new keys honoured-or-marked-extension; output schema unchanged
- [ ] template(s) + `--init-*` flag updated and CI-guaranteed to parse
- [ ] `integration` tests with `@pytest.mark.req(...)` (+ `realdata` if release-gated)
- [ ] RIG_CALIBRATION_GUIDE.md (+ interfaces.md if public API) updated
- [ ] DoD + `check_traceability --check` + tree-hygiene + publish_guard green
