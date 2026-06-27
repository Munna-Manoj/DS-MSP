"""Multi-view geometry on **bearing vectors**.

A fisheye measures *rays*, not pixels on a plane, so two-view geometry for a wide-FOV
camera is done on unit bearing vectors — `cam.unproject(pixels)` — and never touches a
pinhole. The calibrated epipolar constraint ``f2ᵀ E f1 = 0`` holds for **any** central
model, so the eight-point algorithm, pose recovery, and triangulation below work the same
for Double Sphere, UCM, EUCM, Kannala-Brandt, … — that is the whole point.

This layer is pure NumPy and model-agnostic: feed it rays, get relative pose and 3D points.
"""

from .bundle import (
    angular_reprojection_error,
    estimate_relative_pose,
    refine_two_view,
)
from .ransac import ransac_relative_pose, sampson_residual
from .two_view import (
    decompose_essential,
    epipolar_residual,
    essential_from_rays,
    recover_pose,
    relative_pose,
    triangulate_rays,
)

__all__ = [
    "essential_from_rays",
    "decompose_essential",
    "triangulate_rays",
    "recover_pose",
    "relative_pose",
    "epipolar_residual",
    "ransac_relative_pose",
    "sampson_residual",
    "refine_two_view",
    "estimate_relative_pose",
    "angular_reprojection_error",
]
