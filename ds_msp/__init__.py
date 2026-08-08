from .core.contracts import CameraModel
from .model import (
    DoubleSphereCamera,
    ds_project,
    ds_unproject,
    undistort_fisheye,
    solve_pnp_fisheye,
)
from .models import (
    DoubleSphereModel,
    EUCMModel,
    KannalaBrandtModel,
    OCamModel,
    RadTanModel,
    UCMModel,
)
from .adapt import convert
from .ops import Undistorter, solve_pnp, solve_pnp_ransac, solve_pnp_robust
from .mvg import (
    essential_from_rays,
    recover_pose,
    relative_pose,
    triangulate_rays,
)
from .cv import (
    projectPoints,
    undistortPoints,
    distortPoints,
    initUndistortRectifyMap,
    undistortImage,
    estimateNewCameraMatrixForUndistortRectify,
    solvePnP
)
from .ldc import (
    TI_LDC_MeshGenerator,
    TI_LDC_PointUndistorter
)

# Package version — single source of truth is the installed metadata (pyproject
# [project].version, bumped by release-please). Falls back when run from a source
# checkout that was never installed.
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("ds_msp")
except PackageNotFoundError:  # pragma: no cover - source checkout, not installed
    __version__ = "0.0.0+unknown"

# Public API (re-exported above). Listing it here documents the surface and tells
# linters these imports are intentional re-exports, not dead code.
__all__ = [
    "__version__",
    "CameraModel",
    "DoubleSphereCamera", "ds_project", "ds_unproject",
    "undistort_fisheye", "solve_pnp_fisheye",
    "DoubleSphereModel", "EUCMModel", "KannalaBrandtModel",
    "OCamModel", "RadTanModel", "UCMModel",
    "convert", "Undistorter", "solve_pnp", "solve_pnp_robust", "solve_pnp_ransac",
    "essential_from_rays", "recover_pose", "relative_pose", "triangulate_rays",
    "projectPoints", "undistortPoints", "distortPoints",
    "initUndistortRectifyMap", "undistortImage",
    "estimateNewCameraMatrixForUndistortRectify", "solvePnP",
    "TI_LDC_MeshGenerator", "TI_LDC_PointUndistorter",
]
