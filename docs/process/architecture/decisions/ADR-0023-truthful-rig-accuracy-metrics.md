# ADR-0023 — Full-pose and operating-volume rig accuracy

- **Status:** Accepted (recorded 2026-09-23)
- **Deciders:** maintainer
- **Relates to:** ARC-RIG, FR-RIG-024, IFC-12
- **Supersedes:** baseline-length-only wording in prior rig evaluation material

## Context

The rig report compared only the length of each reference-camera baseline. Equal baseline lengths
can conceal an arbitrarily wrong translation direction and any rotation error. Training
reprojection residuals establish fit to the observations, not accuracy against independent truth.
Neither is enough to support a high-precision extrinsic-accuracy claim.

A physical accuracy statement also needs a declared operating volume. The effect of angular error
grows with range; therefore a single rotation/translation parameter error does not directly state
the displacement that a user will see in their workspace.

## Decision

1. Normalize estimates and references to world-to-camera form, then compare all unordered relative
   camera pairs as `T_target_source`. This removes the arbitrary world-frame gauge.
2. Report SO(3) geodesic rotation error, target camera-center vector error in the source frame,
   relative translation error, translation-direction error, and baseline-length error. Baseline
   length is auxiliary and is never named full extrinsic accuracy.
3. A strict, schema-versioned JSON accuracy profile declares the physical length unit, an ROI in
   each pair's source-camera frame (explicit points or an AABB grid), required pairs, and optional
   limits. An empty required-pair list means every common unordered pair.
4. Compute exact Euclidean displacement after transforming every ROI point with the estimated and
   reference relative poses. Report per-pair and aggregate p50/p95/max plus the worst pair. The
   small-angle expression `||delta t|| + D||delta theta||` remains an explanatory bound, not the
   implementation metric.
5. Missing required pairs produce `INCOMPLETE`; absent independent truth or absent ROI produce
   `NOT_EVALUATED`. They never silently pass.
6. Reprojection PASS/WARN/FAIL is relabeled **training-fit verdict**. Accuracy-limit checks are
   separate evidence and do not change process exit semantics in this stage.
7. Existing top-level baseline fields remain for one compatibility cycle with deprecation metadata.

## Consequences

**Positive**

- Equal-length/wrong-direction and rotation-at-range failures become visible.
- Accuracy claims are tied to physical units, a deployment ROI, an independent reference, and all
  required camera pairs.
- Reports remain machine-readable and explicit when evaluation could not be performed.

**Costs / limits**

- A profile is meaningful only if its coordinate units match the calibrated scene and reference.
- MC-Calib output is a comparison reference, not independent truth; it is reported separately.
- A simulation can test parameter consequences but cannot prove that estimated parameters match a
  physical rig unless independent physical truth is supplied.

## Verification

- `tests/rig/test_accuracy.py` covers identity, equal-baseline/wrong-direction, pure rotation at
  range, inverse conventions, world-frame invariance, missing pairs, strict schema, and limits.
- Rig/config/report tests cover pipeline integration, packaged profile generation, and wording.
- `tests/realdata/test_rig_blender.py` is the release-gated Blender truth validation.
- `python tools/check_traceability.py --check` verifies the governance links.

## Alternatives considered

- *Keep worst baseline percentage.* Rejected because it observes only one scalar per pair.
- *Use reprojection residual as accuracy.* Rejected because training fit can remain low under biased
  intrinsics/extrinsics and shared model error.
- *Use only the linearized range formula.* Rejected because exact transformed-point displacement is
  inexpensive and remains valid outside the small-angle regime.
