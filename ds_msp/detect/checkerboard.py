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


#: Largest accepted local-grid inconsistency, as a fraction of the local corner spacing.
GRID_CONSISTENCY_TOL = 0.05
#: Absolute floor (px) so tiny, noisy grids are not rejected for sub-pixel jitter.
GRID_CONSISTENCY_FLOOR_PX = 0.5


def _homographies(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Batched DLT: ``(B,n,2)`` grid points -> ``(B,n,2)`` pixels, returns ``(B,3,3)``."""
    B, n, _ = src.shape
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    z, o = np.zeros_like(x), np.ones_like(x)
    A = np.empty((B, 2 * n, 9))
    A[:, 0::2] = np.stack([x, y, o, z, z, z, -u * x, -u * y, -u], -1)
    A[:, 1::2] = np.stack([z, z, z, x, y, o, -v * x, -v * y, -v], -1)
    return np.linalg.svd(A)[2][:, -1].reshape(B, 3, 3)


def grid_consistency(grid_xy: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Per-corner local-grid inconsistency ``(error_px, spacing_px)``.

    Each corner whose 8 grid neighbours were all detected is predicted from them by a local
    homography. Lens distortion is negligible across one 3x3 cell patch, so a correct grid
    predicts every centre to a small fraction of the spacing, whereas a grid hallucinated on
    background structure or marker interiors does not. Corners without a full neighbourhood
    get ``nan``. ``grid_xy`` holds integer grid coordinates."""
    g = np.rint(np.asarray(grid_xy, float)).astype(int)
    uv = np.asarray(pixels, float)
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(g)}
    offsets = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
    centres, neigh = [], []
    for i, (a, b) in enumerate(g):
        rows = [index.get((int(a) + dx, int(b) + dy)) for dx, dy in offsets]
        if None not in rows:
            centres.append(i)
            neigh.append(rows)
    err, spacing = np.full(len(g), np.nan), np.full(len(g), np.nan)
    if not centres:
        return err, spacing
    c, nb = np.asarray(centres), np.asarray(neigh)
    src = np.asarray(offsets, float)[None].repeat(len(c), 0)
    spacing[c] = np.linalg.norm(uv[nb[:, [3, 4, 1, 6]]] - uv[c][:, None], axis=2).mean(1)
    # normalise the pixels (Hartley) so the DLT is well conditioned: centre on the
    # neighbourhood mean, scale by the local spacing
    origin = uv[nb].mean(1)
    scale = np.maximum(spacing[c], 1e-9)[:, None, None]
    H = _homographies(src, (uv[nb] - origin[:, None]) / scale)
    pred = H[:, :, 2]
    with np.errstate(divide="ignore", invalid="ignore"):         # degenerate -> nan -> reject
        pred = pred[:, :2] / pred[:, 2:3] * scale[:, 0] + origin
    err[c] = np.linalg.norm(pred - uv[c], axis=1)
    return err, spacing


#: Smallest accepted dark/light separation of cell intensities, as a fraction of the
#: image's dark-to-light range under the board.
CELL_CONTRAST_MIN = 0.35
#: Bilinear sample positions inside a cell (corner-quad coordinates).
_CELL_SAMPLES = np.array([[0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])


def cell_parity_ok(gray: np.ndarray, grid_xy: np.ndarray, pixels: np.ndarray) -> bool:
    """Image evidence that the grid's cells are exactly the board's squares.

    In a correct checker grid every cell (four adjacent detected corners) is one square:
    uniformly dark or light inside, alternating with ``(i + j)`` parity. Five samples per
    cell (centre and the four bilinear quarter points, small patches) must all agree and the
    two parities must separate cleanly. This rejects what geometry cannot: lattices that skip
    corners (rows on squares 0, 1, 3, 9 share the uniform cross-ratio), sheared lattices
    whose cells span two squares (odd steps still alternate parity at the centre), and grids
    extrapolated onto blank margin (no contrast)."""
    g = np.rint(np.asarray(grid_xy, float)).astype(int)
    uv = np.asarray(pixels, float)
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(g)}
    img = np.asarray(gray, np.float32)
    cells, values = [], []
    for (a, b), i in index.items():
        quad = [i, index.get((a + 1, b)), index.get((a, b + 1)), index.get((a + 1, b + 1))]
        if None in quad:
            continue
        p00, p10, p01, p11 = uv[quad]
        side = min(np.linalg.norm(p10 - p00), np.linalg.norm(p01 - p00),
                   np.linalg.norm(p11 - p10), np.linalg.norm(p11 - p01))
        r = max(1, int(0.06 * side))
        vals = []
        for su, sv in _CELL_SAMPLES:
            q = (1 - su) * (1 - sv) * p00 + su * (1 - sv) * p10 + (1 - su) * sv * p01 + su * sv * p11
            patch = cv2.getRectSubPix(img, (2 * r + 1, 2 * r + 1), (float(q[0]), float(q[1])))
            vals.append(float(np.median(patch)))
        cells.append((a + b) % 2)
        values.append(vals)
    if not cells or len(set(cells)) < 2:
        return False
    parity, v = np.asarray(cells), np.asarray(values)            # (C,), (C,5)
    lo, hi = np.percentile(v, [2, 98])
    span = float(hi - lo)
    if span <= 1e-6:
        return False
    v0, v1 = v[parity == 0].ravel(), v[parity == 1].ravel()
    gap = max(v0.min() - v1.max(), v1.min() - v0.max())        # every sample separates
    return bool(gap / span >= CELL_CONTRAST_MIN)


def grid_is_consistent(grid_xy: np.ndarray, pixels: np.ndarray) -> bool:
    """True when every checkable corner passes :func:`grid_consistency` within
    ``max(GRID_CONSISTENCY_FLOOR_PX, GRID_CONSISTENCY_TOL * spacing)``; a grid with no
    corner that has a full 3x3 neighbourhood cannot be verified and is rejected."""
    err, spacing = grid_consistency(grid_xy, pixels)
    if np.any(np.isnan(err) & np.isfinite(spacing)):              # degenerate local fit
        return False
    ok = np.isfinite(err)
    if not ok.any():
        return False
    limit = np.maximum(GRID_CONSISTENCY_FLOOR_PX, GRID_CONSISTENCY_TOL * spacing[ok])
    return bool(np.all(err[ok] <= limit))


def meta_object_points(meta: np.ndarray, spec: CheckerboardSpec) -> np.ndarray:
    """Metric object points for an SB metadata grid (row-major, as SB orders its corners).
    With ``spec.marker`` the metadata's single type-4 corner is the physical origin."""
    yy, xx = np.indices(meta.shape)
    xy = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
    if spec.marker:
        origin = np.flatnonzero(meta.ravel() == 4)
        if len(origin) != 1:
            raise ValueError("marked checkerboard must have exactly one SB origin")
        xy -= xy[origin[0]].copy()
    return np.column_stack((xy * spec.square_size, np.zeros(len(xy))))


def detect_corners(gray: np.ndarray, spec: CheckerboardSpec) -> Optional[np.ndarray]:
    """Return detected pixels, or None. Oversized boards may return more than rows*cols.

    Use :func:`detect_correspondences` to obtain matching metric geometry, particularly
    for oversized or marked boards. Unmarked partial grids have only a view-local origin.
    """
    found = detect_correspondences(gray, spec)
    return None if found is None else found[1]


def detect_correspondences(gray: np.ndarray, spec: CheckerboardSpec):
    """Return matching (object points, pixels), using SB metadata for extended grids.

    Returns ``None`` when nothing is found, the grid fails the local geometric consistency
    gate (:func:`grid_is_consistent`), or its cells do not alternate dark/light like the
    board's squares (:func:`cell_parity_ok`).

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
        xyz = meta_object_points(meta, spec)
    else:
        xyz = board_object_points(spec)
    if len(xyz) != len(pixels):
        raise ValueError("checkerboard geometry does not match the detected corner count")
    # SB can report a well-formed grid on background structure or ChArUco marker interiors,
    # most often with CALIB_CB_LARGER; a geometrically inconsistent grid is a false detection.
    grid = xyz[:, :2] / spec.square_size
    if not grid_is_consistent(grid, pixels):
        return None
    if not cell_parity_ok(gray, grid, pixels):
        return None
    return xyz, pixels
