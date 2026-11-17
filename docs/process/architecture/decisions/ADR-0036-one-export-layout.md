# ADR-0036 — LDC meshes and Isaac LUTs are written by one command, in one folder layout

- **Status:** Accepted (recorded 2026-11-11; mesh layout amended 2026-11-12 after a check on a TI board; table size and block size amended 2026-11-17 after a check of every camera type on the board)
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
2. **A mesh is a file with a record.** The binary holds the Q3 mesh in the layout TI's
   `tiovx_ldc_module` reads (`tiovxldc lut-file=`): `(v, h)` per 32-bit node, rows padded to 16
   nodes, covering the output rounded up to whole cells. The JSON record holds the settings, the camera, a checksum, the byte layout in words,
   the `tiovxldc` settings to load it with, and the measured error. Reading checks the checksum
   and the size.
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
- On a TI TDA4VM board (`tiovxldc`), an exported KB fisheye mesh corrected a real photo to
  within 0.21 px of DS-MSP's model of the board (PSNR 55 dB; the same image shifted by 0.5 px
  gives 36 dB). The layout released in 0.28.0, `(h, v)` with unpadded rows, was off by up to
  25 px on the board; its records (format 1) are now refused with a request to export again.
- `ds-msp lut` without `--camera` now exports every camera of a file, not only `cam0`.
- Every camera type was then checked on the same board: a KB and a Double Sphere lens, each
  converted into KB, DS, EUCM, UCM and RadTan, at 21 output sizes and node spacings. Three board
  limits that the first check missed are now handled when the mesh is written:
  - **Table size.** Below the last row of table nodes the board uses zero displacement, so the
    table covers the output rounded up to whole cells.
  - **Block memory.** Each LDC output block must fit the input it reads into the LDC's memory:
    up to 19,505 input pixels came out right, from 19,667 wrong. The record names the largest
    block within 16,384.
  - **Row pitch.** The LDC writes whole blocks, so a block width that does not divide the
    output row pitch spilled into the next row. Only widths that divide it are chosen.
- With these, every board output matched DS-MSP's model to PSNR 43.7 to 56.4 dB, with no pixel
  off by more than 30 grey levels.
- Meshes where the camera model folds are reported: a RadTan model used beyond the field of
  view it was fitted for.
