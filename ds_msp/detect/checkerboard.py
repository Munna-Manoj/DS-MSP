"""Plain checkerboard detection via OpenCV's sector-based detector.

Uses ``cv2.findChessboardCornersSB`` (OpenCV >= 4.5.1), based on Duda & Frese, "Accurate
Detection and Localization of Checkerboard Corners for Calibration," BMVC 2018 — a localized
Radon-transform / point-symmetry detector, a real, documented improvement in noise/blur
robustness and sub-pixel accuracy over the classic ``cv2.findChessboardCorners``.

**Not recommended for extreme fisheye.** That accuracy gain is not documented or benchmarked
as fisheye/radial-distortion-specific. Two reference tools this project already studies for its
rig pipeline confirm the same gap from different directions: MC-Calib (the upstream project
``ds_msp.rig`` mirrors) has no plain-checkerboard board type at all — only ChArUco/AprilTag;
Kalibr does support checkerboard but its own docs steer wide-FOV users toward AprilGrid instead,
specifically because of this failure mode. For lenses beyond roughly 120 degrees FOV, prefer
``ds_msp.detect.charuco`` or ``ds_msp.detect.detect`` (AprilGrid). An "undistort first, detect,
then map corners back" strategy is a real, documented technique for pushing checkerboard
detection further into the fisheye periphery, but it needs a prior distortion/FOV estimate
before any calibration exists (a real chicken-and-egg constraint) and is deliberately not
built here — :class:`~ds_msp.calib.board.CheckerboardBoard` only takes image paths today, so
adding such a strategy later is a non-breaking extension, not an interface change.

Verified empirically against the real OpenCV detector (not assumed): corners come back in
row-major order with the first pattern axis (``cols``) varying fastest — identical to
:func:`ds_msp.detect.charuco.board_object_points`'s ``k % cols, k // cols`` convention, so no
reordering is needed between the two board types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CheckerboardSpec:
    """Plain checkerboard geometry: interior-corner counts + metric spacing."""
    cols: int                     # interior corners along the first (fastest-varying) axis
    rows: int                     # interior corners along the second axis
    square_size: float            # metric spacing used for the 3-D points (e.g. 0.025)
    larger: bool = False          # CALIB_CB_LARGER -- the board may exceed the image frame
    marker: bool = False          # CALIB_CB_MARKER -- board has an asymmetric marker cell

    @property
    def n_corners(self) -> int:
        return self.cols * self.rows


def _flags(spec: CheckerboardSpec) -> int:
    flags = 0
    if spec.larger:
        flags |= cv2.CALIB_CB_LARGER
    if spec.marker:
        flags |= cv2.CALIB_CB_MARKER
    return flags


def board_object_points(spec: CheckerboardSpec) -> np.ndarray:
    """``(rows*cols, 3)`` interior-corner positions at ``square_size`` spacing, row-major with
    ``cols`` fastest -- identical convention to
    :func:`ds_msp.detect.charuco.board_object_points`, and the direct successor to
    ``ds_msp.utils.build_checkerboard_points``."""
    k = np.arange(spec.n_corners)
    return np.c_[(k % spec.cols) * spec.square_size,
                 (k // spec.cols) * spec.square_size,
                 np.zeros(spec.n_corners)].astype(float)


def detect_corners(gray: np.ndarray, spec: CheckerboardSpec) -> Optional[np.ndarray]:
    """Detect one checkerboard in one grayscale image. Returns ``(rows*cols, 2)`` pixel
    corners in the same row-major, ``cols``-fastest order as :func:`board_object_points`, or
    ``None`` on failure. ``findChessboardCornersSB`` is all-or-nothing per image (no partial
    detection) -- the same contract as ChArUco's ``detectBoard`` per board."""
    ok, corners = cv2.findChessboardCornersSB(gray, (spec.cols, spec.rows), flags=_flags(spec))
    if not ok:
        return None
    return corners.reshape(-1, 2).astype(float)
