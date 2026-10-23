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
        return False, None, None

    monkeypatch.setattr(cv2, "findChessboardCornersSBWithMeta", _fake)
    spec = CheckerboardSpec(cols=6, rows=5, square_size=0.2, larger=True, marker=True)
    assert detect_corners(np.zeros((10, 10), np.uint8), spec) is None
    assert captured["pattern_size"] == (6, 5)
    assert captured["flags"] & cv2.CALIB_CB_LARGER
    assert captured["flags"] & cv2.CALIB_CB_MARKER


def test_oversized_board_geometry_matches_real_sb_metadata(tmp_path):
    from ds_msp.calib.board import CheckerboardBoard
    from ds_msp.detect.checkerboard import detect_correspondences
    image = _render_checkerboard(CheckerboardSpec(8, 7, 0.025))
    spec = CheckerboardSpec(6, 5, 0.025, larger=True)
    xyz, pixels = detect_correspondences(image, spec)
    assert len(xyz) == len(pixels) == 56
    # The geometry must explain the actual image, not just have the right length.
    H, _ = cv2.findHomography(xyz[:, :2], pixels)
    predicted = cv2.perspectiveTransform(xyz[:, :2][None], H)[0]
    assert np.max(np.linalg.norm(predicted - pixels, axis=1)) < 0.2
    path = str(tmp_path / "larger.png")
    cv2.imwrite(path, image)
    obs = CheckerboardBoard(spec).detect([path])
    assert len(obs) == 1 and obs[0].pixels.shape == (56, 2)


def test_marker_metadata_defines_metric_origin():
    from ds_msp.detect.checkerboard import meta_object_points
    meta = np.ones((5, 6), np.uint8)
    meta[2, 3] = 4
    xyz = meta_object_points(meta, CheckerboardSpec(6, 5, 0.025, marker=True))
    np.testing.assert_array_equal(xyz[15], [0, 0, 0])
    np.testing.assert_allclose(xyz[16], [0.025, 0, 0])


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


def _grid(cols, rows):
    k = np.arange(cols * rows)
    return np.c_[k % cols, k // cols].astype(float)


def test_grid_consistency_accepts_perspective_and_rejects_a_displaced_corner():
    from ds_msp.detect.checkerboard import grid_is_consistent
    g = _grid(8, 6)
    H = np.array([[40.0, 5.0, 100.0], [2.0, 38.0, 80.0], [1e-2, 5e-3, 1.0]])
    p = np.c_[g, np.ones(len(g))] @ H.T
    p = p[:, :2] / p[:, 2:]
    assert grid_is_consistent(g, p)
    q = p.copy()
    q[20] += [6.0, 0.0]
    assert not grid_is_consistent(g, q)


def test_cell_parity_rejects_lattices_geometry_cannot_see():
    """A skipped-corner lattice and a sheared lattice are perfect perspective grids, so only the
    image (one uniform, alternating square per cell) can reject them."""
    from ds_msp.detect.checkerboard import cell_parity_ok, grid_is_consistent
    spec = CheckerboardSpec(8, 7, 0.025)
    img = _render_checkerboard(spec)
    ok, c = cv2.findChessboardCornersSB(img, (8, 7))
    assert ok
    uv = c.reshape(-1, 2).astype(float)
    full = _grid(8, 7)
    assert cell_parity_ok(img, full, uv)
    step2 = [i for i, (x, y) in enumerate(full) if x % 2 == 0 and y % 2 == 0]
    g2 = full[step2] / 2
    assert grid_is_consistent(g2, uv[step2]) and not cell_parity_ok(img, g2, uv[step2])
    shear = [int(y * 8 + x + y) for y in range(3) for x in range(5)]  # steps (1,0) and (1,1)
    gs = _grid(5, 3)
    assert grid_is_consistent(gs, uv[shear]) and not cell_parity_ok(img, gs, uv[shear])
    blank = np.full_like(img, 255)
    assert not cell_parity_ok(blank, full, uv)


def test_oversized_detection_on_a_charuco_image_is_rejected():
    """SB with CALIB_CB_LARGER reports grids on ChArUco images (white squares hold markers);
    the cell check rejects them instead of pairing them with checkerboard geometry."""
    from ds_msp.detect.checkerboard import detect_correspondences
    board = cv2.aruco.CharucoBoard((9, 7), 0.04, 0.03,
                                   cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000))
    img = board.generateImage((900, 700), marginSize=40)
    assert detect_correspondences(img, CheckerboardSpec(3, 3, 0.04, larger=True)) is None
