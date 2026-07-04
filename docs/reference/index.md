# API Reference

Generated from the library's own docstrings — this is the authoritative signature/parameter
reference. For task-oriented recipes, see [How-to](../how-to/README.md); for guided tutorials,
see [Learn](../learn/README.md); for the *why* behind a design, see
[Explanation](../explain/README.md).

## Package map

| Module | What it covers |
|---|---|
| [`ds_msp.core`](core.md) | The `CameraModel` protocol and shared data conventions every other module depends on. |
| [`ds_msp.models`](models.md) | The eight camera models — Double Sphere, EUCM, Kannala-Brandt, OCam, RadTan, UCM, and their `+` extensions. |
| [`ds_msp.calib`](calib.md) | Single-camera intrinsics calibration — board targets, detection, bundle adjustment. |
| [`ds_msp.detect`](detect.md) | Board/tag detection front ends (checkerboard, ChArUco, AprilGrid). |
| [`ds_msp.adapt`](adapt.md) | Converting a calibrated model from one camera family to another. |
| [`ds_msp.rig`](rig.md) | Multi-camera rig calibration (MC-Calib-compatible). |
| [`ds_msp.io`](io.md) | Reading/writing Kalibr, MC-Calib, COLMAP, and nerfstudio camera formats. |
| [`ds_msp.mvg`](mvg.md) | Multi-view geometry on bearing rays — essential matrix, RANSAC relative pose, triangulation. |
| [`ds_msp.stereo`](stereo.md) | Stereo depth directly on raw fisheye rays, no rectification. |
| [`ds_msp.vo`](vo.md) | Monocular visual odometry from tracked features. |
| [`ds_msp.geometry`](geometry.md) | Shared geometric primitives (rotations, poses, manifold ops). |
| [`ds_msp.ops`](ops.md) | Model-agnostic undistortion and PnP services. |
| [`ds_msp.data`](data.md) | Dataset loading utilities. |
| [`ds_msp.cv`](cv.md) | An `cv2`/`cv2.fisheye`-signature-compatible shim over the library's own models. |
| [`ds_msp.ldc`](ldc.md) | Lens-distortion-correction mesh export (e.g. for embedded/ISP undistortion pipelines). |
