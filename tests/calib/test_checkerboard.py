"""Plain checkerboard detection via ``cv2.findChessboardCornersSB`` — self-contained (renders
its own board), plus the PnP-based pose-initialization sanity check relocated from
``tests/test_robustness_and_ldc.py`` (see FR-CALIB-005's removal of the JSON/COCO front-end
that test's original module was part of).
"""
import cv2
import numpy as np
import pytest

from ds_msp.detect.checkerboard import CheckerboardSpec, board_object_points, detect_corners
from ds_msp.model import DoubleSphereCamera

SPEC = CheckerboardSpec(cols=6, rows=5, square_size=0.2)


def _render_checkerboard(spec: CheckerboardSpec, square_px: int = 60, margin_px: int = 60):
    """A real, synthetic checkerboard image: (cols+1) x (rows+1) squares so there are exactly
    spec.cols x spec.rows interior corners, matching OpenCV's own convention."""
    n_sq_x, n_sq_y = spec.cols + 1, spec.rows + 1
    w, h = n_sq_x * square_px + 2 * margin_px, n_sq_y * square_px + 2 * margin_px
    img = np.full((h, w), 255, dtype=np.uint8)
    for i in range(n_sq_y):
        for j in range(n_sq_x):
            if (i + j) % 2 == 0:
                y0, x0 = margin_px + i * square_px, margin_px + j * square_px
                img[y0:y0 + square_px, x0:x0 + square_px] = 0
    return img


def test_object_points_row_major_cols_fastest():
    xyz = board_object_points(SPEC)
    assert xyz.shape == (30, 3)
    assert np.allclose(xyz[0], [0.0, 0.0, 0.0])
    assert np.allclose(xyz[1], [0.2, 0.0, 0.0])          # +1 index -> +cols direction (x)
    assert np.allclose(xyz[SPEC.cols], [0.0, 0.2, 0.0])  # +cols index -> next row (y)
    assert np.allclose(xyz[:, 2], 0.0)


def test_detect_rendered_board_recovers_all_corners():
    """Render the exact board to an image and detect it back: every interior corner is found,
    in the same row-major, cols-fastest order board_object_points assumes (verified against the
    real detector, not assumed -- see module docstring)."""
    img = _render_checkerboard(SPEC)
    corners = detect_corners(img, SPEC)
    assert corners is not None
    assert corners.shape == (SPEC.n_corners, 2)
    # index 0 -> index 1 should move roughly one square in x, ~0 in y (cols-fastest, row-major)
    d01 = corners[1] - corners[0]
    d0c = corners[SPEC.cols] - corners[0]
    assert abs(d01[0]) > 30 and abs(d01[1]) < 5
    assert abs(d0c[1]) > 30 and abs(d0c[0]) < 5


def test_detect_corners_returns_none_on_a_blank_image():
    blank = np.full((400, 400), 128, dtype=np.uint8)
    assert detect_corners(blank, SPEC) is None


def test_larger_and_marker_flags_pass_through_to_opencv(monkeypatch):
    captured = {}

    def _fake(gray, pattern_size, flags=0):
        captured["pattern_size"] = pattern_size
        captured["flags"] = flags
        return False, None

    monkeypatch.setattr(cv2, "findChessboardCornersSB", _fake)
    spec = CheckerboardSpec(cols=6, rows=5, square_size=0.2, larger=True, marker=True)
    assert detect_corners(np.zeros((10, 10), np.uint8), spec) is None
    assert captured["pattern_size"] == (6, 5)
    assert captured["flags"] & cv2.CALIB_CB_LARGER
    assert captured["flags"] & cv2.CALIB_CB_MARKER


def test_robust_calibration_initialization():
    """Verify that calibration's PnP-based pose initialization is robust, on synthetic
    checkerboard correspondences (relocated from the deleted JSON/COCO front-end's test)."""
    Xw = board_object_points(SPEC)

    fx, fy, cx, cy = 711.57, 711.24, 949.18, 518.81
    xi, alpha = 0.183, 0.809
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha, 1920, 1080)

    rvec_gt = np.array([0.1, -0.2, 0.05])
    tvec_gt = np.array([-0.3, 0.15, 2.0])
    R, _ = cv2.Rodrigues(rvec_gt)
    Xc = (R @ Xw.T).T + tvec_gt
    uv, valid = cam.project(Xc)

    fx0, fy0, cx0, cy0, xi0, alpha0 = 800.0, 800.0, 960.0, 540.0, 0.5, 0.5
    cam0 = DoubleSphereCamera(fx0, fy0, cx0, cy0, xi0, alpha0, 1920, 1080)

    rays, valid_unproj = cam0.unproject(uv)
    pts_norm = rays[:, :2] / rays[:, 2:3]
    ret, rvec0, tvec0 = cv2.solvePnP(Xw[valid_unproj], pts_norm[valid_unproj], np.eye(3), None)

    assert ret, "PnP initialization failed!"
    t_diff = np.linalg.norm(tvec0.squeeze() - tvec_gt)
    assert t_diff < 0.35, f"PnP init translation too far from GT: got {tvec0.squeeze()}, GT={tvec_gt}"


# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-CALIB-005")
