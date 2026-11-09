# ADR-0034 — DS-MSP Studio is a local web app shipped in the wheel

- **Status:** Accepted (recorded 2026-11-08)
- **Relates to:** ARC-UI, FR-UI-001, FR-UI-002, FR-UI-003

## Context

Rig results are spread over seven or more files (`calibration_report.json`,
`calibration_acceptance.json`, the manifest, the MC-Calib YAML files and the text report).
The per-camera checks added by ADR-0033 are recorded but are hard to read without tooling. The
interface must run on a laptop, on a remote computer reached over SSH, and on a headless rig PC,
and must keep working offline.

Options considered:

- A desktop toolkit (Qt / PySide6): no remote use, and per-OS packaging.
- Electron or Tauri: a second runtime to package.
- Streamlit, Gradio or NiceGUI: their rerun-on-change model fits long jobs, 3D views and folder
  pickers poorly.
- A local web app: a Python server inside the wheel and a prebuilt browser interface.

## Decision

1. **Form factor.** `ds-msp studio` starts a FastAPI server (the `ui` extra) that serves a
   React + TypeScript interface built with Vite. The release build writes the interface into
   `ds_msp/studio/static`, so users need no Node.js; a source checkout without a build serves a
   page that says how to build it, and the JSON API still works.
2. **One frontend.** The former `web/` learning page becomes the Learn page of Studio. The
   public demo on GitHub Pages is the same app built without a server; its Results page explains
   how to run Studio.
3. **Read-only first.** The first version reads result folders and never writes. Running a
   calibration from the interface comes later and must produce the same files as the CLI.
4. **One config schema.** `ds_msp.rig.config_schema` describes every `calib_param.yml` key, with
   the default the loader applies. Tests compare it with `load_config` and the shipped template,
   so forms cannot drift from the CLI.
5. **Verdicts only from recorded checks.** A camera's status comes from the checks the run
   recorded: the lens check (validated on the starved-camera matrix), corner coverage, the opt-in
   split check, and observability findings that name the camera. Uncertainties are labelled as
   lower bounds, and every run is labelled not validated with a second capture.
6. **Safety.** The server binds to localhost by default, needs a token on any other address,
   resolves every requested folder inside its roots, and serves only the known result files.
   Fonts and scripts are bundled; nothing loads from a CDN.
7. **Layering.** Nothing in the library imports `ds_msp.studio` (an import-linter contract and
   `tests/contract/test_independence.py`). Only the `ds-msp` command dispatcher does.

## Consequences

- The wheel grows by the built interface, about 2.2 MB (three.js is most of it).
- The release workflow needs Node.js to build the interface before building the wheel.
- Results written before ADR-0033 have no lens or split-check fields; Studio shows them as
  not recorded rather than as failed.
- Comparing two runs, launching runs and the second-capture validation step are not part of this
  decision's first version.
