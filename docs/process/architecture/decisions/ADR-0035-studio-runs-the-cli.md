# ADR-0035 — Studio sets up and runs calibrations by running the rig CLI

- **Status:** Accepted (recorded 2026-11-10)
- **Relates to:** ARC-UI, ADR-0034, FR-UI-002, FR-UI-007, FR-UI-008, FR-RIG-033

## Context

ADR-0034 shipped Studio read-only and left running a calibration for later, with one condition:
a run started from the interface must produce the same files as the CLI. Until then a first run
needed a hand-written `calib_param.yml` of up to 51 keys, and the two commonest mistakes, a
folder layout the detector cannot read and a board description that does not match the target,
showed up only after a run, the second as `IndexError: list index out of range`.

## Decision

1. **A job is the CLI in its own process.** Studio writes `calib_param.yml` into a new result
   folder and starts `python -m ds_msp.rig.cli --config <that file>` there. It does not call the
   pipeline in the server process. The result files are therefore the CLI's own, a crash or a
   cancel cannot take the server down, and the saved config repeats the run from a terminal.
2. **The form holds only what the user set.** A key that is not set is left out of the file, so
   the loader's default applies and `profile: production` can still raise it. Writing all keys
   would mark each as set and switch the profile off for it.
3. **Three checks before a run:** the schema, the served roots (every path value must lie under
   one) and the CLI's own `load_config` on the file that would be written.
4. **Two checks of the data before a run,** using the calibration's own code: a scan of the
   photo folder (file-name rules, shared frames) and a search for the configured board in up to
   4 photos per camera.
5. **Progress comes from a log the CLI writes** (`--progress-file`, FR-RIG-033), not from
   parsing terminal output. The log wraps the live view's callbacks and only listens.
6. **One job at a time,** in the order started. A calibration uses every core.
7. **Guarding a server that can now start processes.** Changes are POSTs that must carry the
   `x-dsmsp-studio` header, which a page on another site cannot send without a preflight the
   server never grants, and must come from the server's own origin. Bound to a local address,
   Studio answers only requests addressed to a local host name. `--read-only` removes every
   route that writes. The process started is always the rig CLI on a file Studio wrote from
   checked values; no request supplies a command.
8. **A board found in no image is an error that says so,** in the CLI as well as in Studio.
9. **A repeat is rebuilt from the manifest, not replayed from it.** The recorded config path and
   overrides are parsed and checked (config under a served root, overrides limited to config
   keys) and only the output folder is replaced. Its inputs are re-hashed on request, and the
   finished repeat is compared with the original by the digests and file checksums both
   manifests hold.

10. **A photo folder is a capture with a history.** Results are matched to it through the
    photo folder their manifest records. Corner detection depends only on the photos, the board
    layout and the detector, so its result is offered for reuse when those are unchanged; the
    file is copied into the new result, and no earlier result is written to.

## Consequences

- A rendered three-camera capture (90 photos, 800 x 600) goes from an empty form to a result
  page in 3 s of calibration time; the lenses equal those of the CLI run on the same config
  (`tests/studio/test_jobs.py`, relative tolerance 1e-6).
- Studio adds four files to a result folder: `calib_param.yml`, `studio_job.json`,
  `studio_run.log` and `studio_progress.jsonl`.
- Photos and results must lie under a `--root` folder. Photos elsewhere need Studio restarted
  with that folder as a root.
- A job still running when Studio stops is stopped with it and shown as interrupted afterwards.
- A repeat of the rendered three-camera capture reproduced all five calibration files
  byte for byte.
- Not in this decision: pause and resume, a memory estimate before a run, excluding photos
  before a run, export to other formats, and capture guidance.

## Alternatives considered

- **Calling `calibrate_from_config` in a worker thread or process pool.** Shares the server's
  interpreter state and BLAS thread settings, and makes "the same files as the CLI" a property
  to keep testing instead of one that holds by construction.
- **Parsing the terminal output for progress.** The output is written for people and changes
  with wording.
