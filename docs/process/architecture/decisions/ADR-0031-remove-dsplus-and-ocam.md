# ADR-0031 — Remove the DS⁺ and OCam camera models

- **Status:** Accepted (recorded 2026-11-06)
- **Deciders:** maintainer
- **Relates to:** ARC-MODELS, ARC-RIG, ARC-IO, ARC-ADAPT, FR-MODEL-001..005, NFR-NUM-014
- **Supersedes:** [ADR-0027](ADR-0027-dsplus-experimental.md) and
  [ADR-0028](ADR-0028-ocam-rig-roll.md) (both kept the models); the DS⁺ half of
  [ADR-0005](ADR-0005-dsplus-eucmplus.md)

## Context

DS-MSP is a rig calibration tool. Its product is the rotation and translation between cameras.

ADR-0027 and ADR-0028 showed that two shipped models cannot deliver those rotations:

- **DS⁺:** its tilt term reproduces a camera rotation about x or y. The identity is exact in the
  pinhole limit and close elsewhere. On real data, the DS⁺ cameras with a tilt error above 2°
  have `tau_y ≈ −tan(error)` (correlation −0.986).
- **OCam:** its affine term reproduces a camera roll exactly, for every angle. That direction
  has zero gradient, so roll stays at its starting value, and the rotation uncertainty is
  undefined.

Both ADRs kept the models, with warnings. The 140-run MC-Calib sweep in ADR-0027 shows what
users got:

- worst-pair rotation error, median: 4.59° with DS⁺ and 0.26° with OCam;
- worst-pair rotation error, max: 29.8° with DS⁺, and 11.0° / 706 mm with OCam;
- runs using only KB, RadTan, DS, UCM or EUCM: 0.16° median, 2.0° max.

Every run with an OCam camera and no DS⁺ reported no rotation σ.

A warning in the docs does not stop a config from selecting these models. The tool's purpose is
better served by not shipping them.

## Decision

1. Delete `ds_msp/models/dsplus.py`, `dsplus_math.py`, `ocam.py` and `ocam_math.py`. Also delete
   their registry entries, Kalibr and MC-Calib I/O, rig wiring, the web-studio ports, and their
   model test files.
2. Shared tests that covered these models are re-pointed at KB and EUCM, so the stereo, two-view,
   wide-FOV PnP, conversion and LDC coverage stays.
3. `convert_best`'s default ladder now ends in Kannala-Brandt (four shape DOF, as OCam had)
   instead of OCam.
4. An old config, Kalibr stanza or MC-Calib file that names `dsplus`, `ds+`, `ocam` or
   `scaramuzza` fails with an error that names this ADR.
   - The MC-Calib reader treats this as an error instead of "model not stated". Those files carry
     `distortion_type: 1`, which would otherwise be read back silently as KB.
5. There is no deprecation window, as for EUCM⁺ (ADR-0010). This is a breaking change.

## Consequences

**Positive**

- No shipped model leaves rig extrinsics unobservable, so a model choice can no longer produce
  a degree-scale extrinsic error.
- Five models remain: RadTan for pinhole lenses; KB, DS, UCM and EUCM for wide lenses.
- The model sweep and every future validation run only on models that can pass it.

**Negative / costs**

- Breaking API change: `DSPlusModel`, `OCamModel`, their names and aliases, and the Kalibr
  `ds_plus`/`ocam` stanzas are gone. Calibrations saved with them must be redone with another model.
- Single-camera accuracy on some lenses is lost. On the real fisheye cam0, DS⁺ fitted
  0.31 px against KB's 0.37 px.
- OCam was the only model without `fx, fy`, and it is the usual model of OCamCalib users.
  Files from OCamCalib must be converted outside DS-MSP, or recalibrated.
- Of the 140 model-sweep trials, only the 57 without DS⁺ or OCam remain comparable. A new sweep is
  needed.
- The rig "pond" replay in the docs was recorded with DS⁺, and it is kept as a historical
  recording. It cannot be regenerated as is.
