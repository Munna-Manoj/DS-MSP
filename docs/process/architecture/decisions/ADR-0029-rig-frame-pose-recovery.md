# ADR-0029 — Preserve supplied lens models and recover unsupported frame poses

- **Status:** Accepted (recorded 2026-11-05)
- **Relates to:** ARC-RIG, FR-RIG-001, FR-RIG-003, FR-RIG-029, NFR-NUM-013

## Context

Partial supplied camera maps were treated as complete by the front end. The
scenario CLI reconstructed every fixed lens as RadTan, and malformed explicitly
named models could silently fall back to another model family. Sparse camera
chains also composed noncommuting camera transforms in the wrong order. Separately, redescending
robust loss can suppress all correct corners of a poorly initialized frame;
continuing the same local solve need not recover its pose. Dropping those
corners would conceal the initialization failure.

## Decision

1. Supplied complete models seed frame PnP directly. Only missing cameras receive
   front-end calibration. Fixed mode requires every observed camera and holds
   every intrinsic parameter; free mode may change every parameter during BA.
   Explicit malformed model declarations fail rather than changing model family.
2. Compose camera-from-reference paths by left multiplication:
   `T_next_reference = T_next_current @ T_current_reference`.
3. After joint BA, before optional hard rejection, and after target refinement,
   resect missing or under-supported object-frame poses using existing model-aware
   plain and robust PnP capabilities. The rig pipeline orchestrates these lower
   capabilities; no model-specific optimizer or new dependency is introduced.
4. Score candidates over all observed corners of the object-frame. Missing
   models/poses and invalid projections are unsupported. Use the existing
   `3.03 * noise_bound` pixel band (3.03 pixels without an explicit bound) and
   require 75% pooled support. This is an explicit heuristic policy, not a theorem
   establishing that a pose or detection is correct.
5. Accept only a strict support-count increase and truncated squared-cost
   decrease, without invalidating any previously valid projection. Pose-only
   polishing holds lenses, rig extrinsics and target geometry fixed. Recovery
   never removes observations. An accepted replacement triggers joint refinement
   with the original fixed/free intrinsic and target policy.
6. Publish an attempt ledger and a separate final support audit in the acceptance
   file. Recompute the latter after the last solve; a successful earlier recovery
   cannot certify the final result. Warn for unresolved final frames.

## Alternatives and limits

Increasing iteration budgets does not restore influence to a rejected pose basin.
Fixing KB's highest-order coefficient changes the model family and does not cure
missing acquisition information. Independent lens initialization alone also does
not retain its information once intrinsics are freed; freeze a trusted calibration
or retain its observations/uncertainty in a suitable joint formulation.

Recovery adds work only for unsupported frames. Pooled corner support can be
dominated by one camera; coherent wrong correspondences can still pass. Neither
the ledger nor small reprojection error is an independent physical-accuracy gate.
Existing rejection, observability, uncertainty and accuracy policies remain
separate. Auxiliary-calibration configuration and an intrinsic-prior solver are
outside this decision.

## Verification

`tests/rig/test_extrinsics.py` uses noncommuting sparse chains.
`test_intrinsics_source_model.py`, `test_cli_intrinsics.py` and
`test_supplied_intrinsics.py` check native parameter interpretation and fixed/free
initialization. `test_pose_recovery.py` checks bad and missing poses, outliers,
cross-camera disagreement, invalid projections, nonmutation, both recovery
boundaries and final-audit regression detection. The dataset-gated two-camera RealSense test
checks final-ledger consistency and deliberately damaged pose recovery on real
camera images, with detections and lens/rig parameters held unchanged. Slow synthetic verification
remains mandatory; these tests do not claim universal nonlinear convergence.

The [lens-information guide](../../../learn/rig_intrinsics_and_pose_recovery.md)
states the conditional mathematics and portable acquisition experiment.
