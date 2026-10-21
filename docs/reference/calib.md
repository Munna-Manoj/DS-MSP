# `ds_msp.calib`

Generic calibration (bundle adjustment) for any `CameraModel`. See
[Calibrate any model](../how-to/calibrate_any_model.md) for task recipes and the
[capstone](../learn/capstone_calibrating_a_real_camera.md) for a full worked example.

::: ds_msp.calib

Calibration observations accept optional `pixel_covariances` with shape `(N, 2, 2)` in
squared pixels. Each matrix must be finite, symmetric and positive definite. The single-camera
`calibrate(..., pixel_covariances=[...])` API accepts one array (or `None`) per view;
`calibrate_camera` forwards covariances supplied by its board adapter. `None` retains unit
weights. Detectors do not currently estimate calibrated measurement covariances automatically.
Residuals and analytic Jacobians are whitened together before robust weighting. When covariance
is supplied, robust scales and noise bounds apply in whitened units; reported reprojection
errors remain in pixels. Pixel covariance is rejected for angular rig residuals until its
propagation into bearing space is implemented.

The checkerboard adapter uses SB metadata for oversized and marked boards. An oversized
unmarked grid has a view-local origin, suitable for independent single-camera poses. A marked
grid uses the type-4 metadata corner as its origin. Shared multi-camera correspondence identity
must not be inferred from an unmarked partial grid. The low-level `detect_corners` function can
return more than `rows * cols` pixels; `detect_correspondences` returns their matching geometry.

Observation containers reject nonfinite measurements and invalid covariance matrices. ChArUco
also rejects collinear, repeated-ID and out-of-image detections. A training-fit PASS describes
reprojection residuals; physical accuracy needs independent metric validation.
