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
        """Total interior corner count, ``cols * rows``."""
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
    """Return detected pixels, or None. Oversized boards may return more than rows*cols.

    Use :func:`detect_correspondences` to obtain matching metric geometry, particularly
    for oversized or marked boards. Unmarked partial grids have only a view-local origin.
    """
    found = detect_correspondences(gray, spec)
    return None if found is None else found[1]


def detect_correspondences(gray: np.ndarray, spec: CheckerboardSpec):
    """Return matching (object points, pixels), using SB metadata for extended grids.

    Without a marker the origin is local to each observation, suitable for independent
    single-camera poses, not persistent multi-camera target identity. With a marker the
    metadata's type-4 corner defines the physical origin.
    """
    if spec.larger or spec.marker:
        ok, corners, meta = cv2.findChessboardCornersSBWithMeta(
            gray, (spec.cols, spec.rows), flags=_flags(spec))
    else:
        ok, corners = cv2.findChessboardCornersSB(
            gray, (spec.cols, spec.rows), flags=_flags(spec))
        meta = None
    if not ok:
        return None
    pixels = corners.reshape(-1, 2).astype(float)
    if spec.larger or spec.marker:
        if meta is None or meta.ndim != 2 or meta.size != len(pixels):
            raise ValueError("SB metadata does not match the detected checkerboard corners")
        yy, xx = np.indices(meta.shape)
        xy = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
        if spec.marker:
            origin = np.flatnonzero(meta.ravel() == 4)
            if len(origin) != 1:
                raise ValueError("marked checkerboard must have exactly one SB origin")
            xy -= xy[origin[0]].copy()
        xyz = np.column_stack((xy * spec.square_size, np.zeros(len(xy))))
    else:
        xyz = board_object_points(spec)
    if len(xyz) != len(pixels):
        raise ValueError("checkerboard geometry does not match the detected corner count")
    return xyz, pixels
