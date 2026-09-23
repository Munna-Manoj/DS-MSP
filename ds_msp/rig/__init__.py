"""Multi-camera rig calibration — the N-camera analogue of MC-Calib.

DS-MSP's single-camera, multi-model intrinsics (``calib.bundle``) and two-view MVG
(``mvg``) are extended here to a full rig: many cameras + many planar boards observed
over many frames, fused into one consistent set of extrinsics + intrinsics.

The math is **model-agnostic** — every routine composes poses and calls a
``CameraModel``'s ``project`` / ``unproject`` / ``project_jacobian``. So the entire
pipeline works for any of DS-MSP's camera models, exactly as it is in MC-Calib.
"""

from __future__ import annotations

from ..io.mccalib import load_camera
from .accuracy import (AccuracyProfile, accuracy_metrics,
                       compare_world_to_camera_poses, load_accuracy_profile)
from .types import BoardObs, Object3D, ObjectObs, RigState
from .calibrate import calibrate_rig

__all__ = ["AccuracyProfile", "BoardObs", "Object3D", "ObjectObs", "RigState",
           "accuracy_metrics", "calibrate_rig", "compare_world_to_camera_poses",
           "load_accuracy_profile", "load_camera"]
