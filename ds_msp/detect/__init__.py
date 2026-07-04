"""Detection adapters (ChArUco / AprilGrid).

The OpenCV-facing layer: turn images into 3D<->2D correspondences as :mod:`ds_msp.data`
records. This is the only place under the calibration stack (besides ``io``) where cv2 is
allowed — the geometry/solver path stays NumPy-native. Depends on ``data`` + ``core``.
"""

from .charuco import (
    BoardSpec,
    board_object_points,
    detect_folder,
    detect_image,
    detect_rig,
    make_detectors,
    single_board_object,
)
from .checkerboard import CheckerboardSpec
from .checkerboard import board_object_points as checkerboard_object_points
from .checkerboard import detect_corners as detect_checkerboard_corners
from .detect import detect_aprilgrid

__all__ = [
    "BoardSpec",
    "make_detectors",
    "board_object_points",
    "single_board_object",
    "detect_image",
    "detect_folder",
    "detect_rig",
    "detect_aprilgrid",
    "CheckerboardSpec",
    "checkerboard_object_points",
    "detect_checkerboard_corners",
]
