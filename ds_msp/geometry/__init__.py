"""Neutral geometry primitives — the shared single-/multi-view geometry layer.

One resection/PnP, robust pose averaging, and covisibility-graph utilities, consumed by
both single-camera (``calib``) and multi-camera (``rig``) calibration so neither has to
import the other. Pure NumPy + stdlib; depends only on ``core`` (and ``data``) — never on
``models``, detection, IO, or a service layer, and never on OpenCV.
"""

from .averaging import (
    average_rotation,
    average_transform,
    average_translation,
    mean_transform,
    robust_average_transform,
)
from .bearing import chordal_bearing_residual_jacobian
from .calibrate_core import bundle_adjust
from .graph import connected_components, covis_weights, shortest_path
from .resection import (
    decompose_P,
    dlt_projection,
    intrinsics_seed,
    ransac_pnp_normalized,
    ransac_resection,
)

__all__ = [
    # bundle adjustment
    "bundle_adjust",
    # bearing residuals
    "chordal_bearing_residual_jacobian",
    # resection / PnP
    "dlt_projection",
    "decompose_P",
    "ransac_resection",
    "intrinsics_seed",
    "ransac_pnp_normalized",
    # pose averaging
    "average_rotation",
    "average_translation",
    "average_transform",
    "robust_average_transform",
    "mean_transform",
    # covisibility graph
    "covis_weights",
    "connected_components",
    "shortest_path",
]
