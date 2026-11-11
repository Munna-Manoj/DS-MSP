# ADR-0036 — LDC meshes and Isaac LUTs are written by one command, in one folder layout

- **Status:** Accepted (recorded 2026-11-11)
- **Relates to:** ARC-INTEROP, ARC-UI, ADR-0035, FR-IO-005, FR-INTEROP-002 to FR-INTEROP-005, FR-UI-025

## Context

A TI LDC mesh existed only as an array returned by `TI_LDC_MeshGenerator`: no file, no command,
and no measure of how far the stored mesh is from the lens it was made from. Isaac LUTs had a
command that exported one camera per call. Studio could do neither. Users need both files for
every camera of a rig, either at the end of a calibration or for camera values they already
have, from DS-MSP or from another tool.

## Decision

1. **One reader for every camera file** (`ds_msp.io.camera_file`): MC-Calib and DS-MSP rig YAML,
   Kalibr and DS-MSP camchains with any number of cameras, COLMAP, nerfstudio, DS-MSP model files
   and LUT manifests. It lists every camera, and one it cannot read is listed with its reason.
2. **A mesh is a file with a record.** The binary holds the Q3 mesh. The JSON record holds the
   settings, the camera, a checksum, the byte layout in words, `checked_on_board: false`, and the
   measured error. Reading checks the checksum and the size.
3. **The error is measured, not estimated.** For each output pixel, the record compares where
   the stored, quantised mesh sends it with where the camera model does. A mesh with clipped Q3
   values is marked unusable wherever it is shown.
4. **One folder layout, written by one function.** A rig run with `export_ldc` or
   `export_isaac_lut`, and `ds-msp export`, both call `ds_msp.exports.write_exports`. Both write
   `ldc/`, `isaac_lut/` and `exports.json`. A camera that fails is reported there and the others
   still export; a calibration is never lost to a failed export.
5. **Studio runs `ds-msp export`** in its own process, as ADR-0035 does for calibrations. Typed
   values are first written into the export folder, so the folder repeats from a terminal.
6. **Stereo is a rig of two cameras.** Its result is the rig file set, so it needs no export
   format of its own.

## Consequences

- Measured on 184 calibrated cameras (7 MC-Calib sequences, 6 rendered 25-camera rigs), at the
  defaults (node every 16 px, balance 0.5): median error 0.05 px, worst 3.1 px, no clipped mesh.
  With the whole view kept (balance 0), 59 of the 79 standard-lens (RadTan) cameras clipped.
- No camera model's convention is shifted by the mesh. Board photos corrected through each mesh
  and through the exact model, and detected again, agree to a median of 0.03 to 0.14 px. The
  exact correction's corners lie 0.05 to 0.10 px from the model, where a half-pixel offset would
  show as about 0.5 px.
- Through the Studio page in Chrome, 16 exports (every result, two files, typed values, a
  clipped mesh) wrote 1318 files, each byte-identical to the job's own `ds-msp export` command
  run from a terminal.
- A rig run with exports writes the same calibration files as the run without them, and its
  exports equal `ds-msp export` on its result, byte for byte.
- The mesh byte layout is not yet checked on a TI board.
- `ds-msp lut` without `--camera` now exports every camera of a file, not only `cam0`.
