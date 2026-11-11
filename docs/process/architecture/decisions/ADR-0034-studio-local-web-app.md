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

## Update 2026-11-11: reading anywhere on this computer

Point 6 limited every read to the folders given with `--root`. A user with photos elsewhere had to
restart Studio to reach them, and could not choose a folder from the page.

Studio bound to this computer, not read-only and not started with `--only-roots` can now browse
and read any folder the user's account can read. Results are still written only under served
folders, and the page can add one for the session. With this wider reach, any account on the same
computer could otherwise reach those files through the port, so a token is generated in this mode
too. Bound to another address, or read-only, Studio stays limited to its roots.

## Update 2026-11-11: camera lineup, a start gate, typed lens values

The calibration reads camera folders by name, so folders named another way were either refused or,
worse, left out with no warning: `Cam_000` next to `Cam_001` ran as one camera. Start also ignored
scan problems, and `fix_intrinsic` without a lens file passed every check and failed in the run.

- Folders are not renamed. Studio makes a folder of links with the expected names, in a hidden
  folder under a served folder, named after the choice so that the same choice gives the same path
  and earlier detections are recognised. No calibration code changes.
- What stops a run is checked on the page and again on the server.
- In Studio, the strict profile requires lens values for every camera and keeps them as given. The
  library's `profile: production` does not require this; a config run from a terminal is unchanged.
- A lens type or lens values that do not suit the photos warn and do not stop the run: the user may
  have computed them elsewhere and knows why.

## Update 2026-11-11: how far a result is from the precision needed

A precision limit was only compared after the run. The result page now states the gap and how many
frames of the same kind would close it, assuming the uncertainty shrinks with the square root of the
number of frames.

Measured on the 10 real datasets, calibrating from a subset of the frames and predicting the pair
uncertainty of the whole capture:

- Random half: predicted over actual, median 1.00, 97% within 25%. Random quarter: median 1.00, 91%
  within 25%.
- Captures of 21 to 36 frames scatter widely: 0.36 to 3.6 from a quarter of the frames. The page calls
  the estimate rough rather than applying a size cut-off.
- First half predicting the whole: up to 190 times too pessimistic, because later frames added views
  the first half lacked. The estimate holds for more of the same, and the page says that new angles can
  need far fewer.

Suggesting where to hold the board is not part of this; it is planned separately.

## Update 2026-11-11: a tracked, one-step-at-a-time setup

The setup page had too much on it at the start and did not show progress. It now asks only how to start, then shows one step at a time beside a tracker.

Steps stay mounted while hidden, so what was typed in one is kept. A step left at its default, such as lens types nobody measured, is shown as the usual choice, not as done.

Every box now has the same border on all four sides. Tailwind also reads `border-s-<colour>` as the inline-start border, so Studio borders use `border-s-rule` and `border-s-strong`; a test guards the class names.

## Update 2026-11-11: checked in a real browser before release

The installed package was driven through the page in Chrome on the seven MC-Calib real sequences
and six rendered 25-camera rigs. All 13 finished, and a second run and the terminal run of
Studio's `calib_param.yml` were byte-identical to the first. That run found two faults, now fixed:

- A single board could not be given its printed number. Three MC-Calib sequences use board 12
  or 13, so no board was found.
- Without WebGL, the 3D view failed and blanked the whole result page. A render guard now
  contains a failing view, and one at the top of the app replaces a blank page with the error.

One limitation remains. **Measure my lenses** picks *Fisheye* (Kannala-Brandt) for every lens at 100° or
wider. On the rendered rigs that cost up to 10 mm against 0.43 mm with the true types, while the
page reported every camera ready. The guides warn about it; changing the rule is left to a later
change, because the threshold must hold for any field of view.

