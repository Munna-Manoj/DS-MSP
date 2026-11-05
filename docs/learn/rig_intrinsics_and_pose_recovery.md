# Rig calibration: lens information and frame-pose recovery

A rig can estimate accurate relative camera poses while its fitted lens map is
poor near the edge of the intended field of view. A small training residual
does not establish accuracy at unobserved angles. Conversely, a single bad
frame pose can produce enormous residuals even when its corner detections and
the rest of the rig are accurate. Diagnose these cases separately.

## Preserve independently measured lens information

For each camera, collect a separate calibration sequence that covers the intended
image region, with varied board tilt, distance and azimuth. Every pixel need not
contain a detected corner. What matters is that the measurements constrain the
model's parameter combinations over the intended operating domain. Keep separate
peripheral views for validation. Use the same resolution, crop, focus and optical
configuration as the rig sequence.

Supply the complete calibrated camera models through `cam_params_path`. With
`fix_intrinsic: true`, every observed camera needs a matching model, and all its
parameters stay fixed through camera, frame and target refinement. With
`fix_intrinsic: false`, the supplied models initialize the fit; missing cameras
are calibrated from scratch. Subsequent bundle adjustment may change **all** lens
parameters. This includes high-order distortion, not just focal length and centre.

Initialization alone does not retain the independent sequence's information.
If the rig observes only central image regions, freeing a well-calibrated lens can
again produce an inaccurate peripheral map. Freeze a trusted calibration when its
uncertainty is acceptable for the application. For joint estimation with uncertain
intrinsics, the principled alternatives are to retain the independent calibration
observations in the optimization, or use their full correlated uncertainty as a
prior. The existing bundle solver can retain observations as described below;
this change does not add a Gaussian intrinsic-prior solver. Fixed-intrinsic
rig uncertainty is conditional on those lens parameters and does not include the
uncertainty of the independent lens calibration.

Do not fix the highest-order KB coefficient to zero solely because one synthetic
lens happens to have that coefficient zero. That restricts the camera family and
can bias other valid lenses. The implementation retains the full KB model.

## Retain the independent observations during joint refinement

To retain independent observations with the multi-object bundle solver, represent
the auxiliary metric board as a fixed-shape object. Assign a separate object-frame
pose to every independent view, even when different cameras observed the same board.
Unique frame IDs prevent unrelated captures from becoming false synchronized
constraints. Initialize each pose by PnP through the estimated lens, using
`T_group_board = inverse(T_camera_group) @ T_camera_board`.

Minimize the sum of rig and auxiliary reprojection objectives, with full lens
parameters and rig geometry free. Auxiliary board poses are nuisance parameters;
do not rigidly attach that board to the rig target unless the physical acquisition
actually guarantees the attachment. This requires explicit observation assembly
through the bundle API; there is no auxiliary-calibration configuration option.

In a local whitened linear model, let `A` be an auxiliary view's intrinsic
Jacobian and `B` its pose Jacobian. Eliminating that view's pose contributes
`A.T @ (I - B @ B^+) @ A`, a positive semidefinite intrinsic information matrix.
Independent views add such terms. Under fixed weights, correct local
linearization, appropriate gauge constraints and invertible total information,
this cannot increase the corresponding local covariance. It does not prove
lower realized error for every noise draw or global nonlinear convergence.
With adaptive robust weights or an automatically estimated loss scale, adding
observations also changes the objective's weights. The fixed-weight information
argument then applies only conditionally, not directly to the full nonlinear fit.
The executable projector and covariance checks are in
`tests/calib/test_calibration_information.py`.

[Calibration Wizard](https://arxiv.org/abs/1811.03264) motivates choosing views
that reduce intrinsic uncertainty while accounting for geometric and corner
uncertainty. [Hagemann et al.](https://arxiv.org/abs/2107.13484) distinguish
calibration bias from variance: neither local covariance nor a low training
residual substitutes for independent model-accuracy checks.

## Why coverage matters mathematically

For known bearings, fixed centre and isotropic focal length, KB radial projection
is linear in `b = [f, f*k1, f*k2, f*k3, f*k4]`, with design row
`a(theta) = [theta, theta^3, theta^5, theta^7, theta^9]`.
If the design matrix `A` has full column rank and measurement error satisfies
`||e||2 <= epsilon`, ordinary least squares obeys

`|a(theta) (b_hat - b)| <= epsilon ||a(theta) A^+||2`.

This follows directly from `b_hat - b = A^+ e` and Cauchy-Schwarz. Large prediction
leverage outside the observed angular range means even bounded errors permit
large extrapolation errors. QR/SVD improves numerical stability but does not
supply missing measurements. This bound is conditional on known bearings; it is
not an exact confidence certificate when board poses, centres and aspect ratio
are estimated too. For a local uncertainty calculation, marginalize those nuisance
parameters and retain the complete intrinsic covariance, including correlations.

Zhang's calibration method estimates intrinsics from a planar target at different
unknown orientations; the Kannala–Brandt paper develops a model and calibration
method suitable for fisheye lenses. Neither establishes that arbitrary central
coverage guarantees a full-image calibration:
[Zhang (2000)](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/),
[Kannala and Brandt (2006)](https://oulu3dvision.github.io/calibgeneric/Kannala_Brandt_calibration.pdf).

## Recover a bad pose before rejecting its observations

A redescending robust loss suppresses large residuals. If an entire frame starts
in a wrong pose basin, its correct corners can receive too little influence to
move that pose. Increasing the iteration cap alone need not escape that pose basin.

After global bundle adjustment, and again after target refinement, the rig now
checks each object-frame's supported-corner fraction. Frames below 75% support
receive fresh per-camera PnP candidates. Support uses the existing pixel noise
band (`3.03 * noise_bound`, or 3.03 pixels for the legacy API). Candidates are
judged against **all cameras observing that object-frame**, with invalid
projections counted as unsupported. A local pose polish holds the camera models,
extrinsics and target fixed.

A replacement must increase supported-corner count, reach the support threshold,
reduce the full-frame truncated squared error, and retain every previously valid
projection. No observation is removed by recovery. An accepted replacement is
followed by joint refinement, with the original fixed/free intrinsic policy.
The first check occurs before optional hard rejection so a bad pose does not
automatically cause correct detections to be discarded.

`calibration_acceptance.json` records attempted recoveries under
`solver.pose_recovery`, including before/after counts, acceptance and unresolved
frames. Unresolved final-stage frames warn; this is a recovery diagnostic, not a
new physical-accuracy certificate. Correct detections cannot always be distinguished
from coherent misdetections using reprojection alone. Independent validation
remains necessary, and a frame with few supported corners may legitimately remain
unresolved.

The separate `solver.frame_support` audit measures every observed object-frame
**after the last joint solve**, so an accepted recovery does not hide a later
regression. Missing poses or camera models count as unsupported. Support pools
corners across cameras; it does not require a separate consensus from every camera.

## Coordinate-frame correctness

For column-vector transforms, a camera graph edge from `current` to `next` stores
`T_next_current`. A path accumulated from the reference therefore updates as
`T_next_reference = T_next_current @ T_current_reference`. Reversing this order
can be invisible on direct neighbours or commuting translations, but is wrong
for a general rotated camera chain. The sparse-chain regression uses independent
absolute poses with noncommuting rotations and translations.

See the [rig reproduction workflow](../how-to/reproduce_rig_calibration.md) for
configuration, provenance and acceptance gates. The self-contained
`benchmarks/kb_coverage_study.py` and its requirement-linked statistical test
`tests/calib/test_kb_acquisition.py` exercise central versus peripheral acquisition
with both signs of nonzero fourth-order KB distortion. They measure a finite
synthetic design, not a universal numerical accuracy guarantee.
