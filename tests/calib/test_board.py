"""The ``Board`` protocol and its three native implementations — each turns images straight
into :class:`~ds_msp.data.observations.Observation`\\ s, the shared seam into
:func:`ds_msp.calib.bundle.calibrate`."""
import cv2
import numpy as np
import pytest

import ds_msp.calib.board as board_mod
from ds_msp.calib.board import (AprilGridBoard, Board, CharucoBoard, CheckerboardBoard,
                                to_correspondences)
from ds_msp.calib.targets import AprilGridTarget
from ds_msp.data.observations import Observation
from ds_msp.detect.charuco import BoardSpec
from ds_msp.detect.checkerboard import CheckerboardSpec

CB_SPEC = CheckerboardSpec(cols=6, rows=5, square_size=0.2)


def _render_checkerboard(spec: CheckerboardSpec, square_px: int = 60, margin_px: int = 60):
    n_sq_x, n_sq_y = spec.cols + 1, spec.rows + 1
    w, h = n_sq_x * square_px + 2 * margin_px, n_sq_y * square_px + 2 * margin_px
    img = np.full((h, w), 255, dtype=np.uint8)
    for i in range(n_sq_y):
        for j in range(n_sq_x):
            if (i + j) % 2 == 0:
                y0, x0 = margin_px + i * square_px, margin_px + j * square_px
                img[y0:y0 + square_px, x0:x0 + square_px] = 0
    return img


def _render_charuco(spec: BoardSpec, id_offset: int = 0):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    ids = np.arange(spec.n_markers, dtype=np.int32) + id_offset
    board = cv2.aruco.CharucoBoard((spec.n_x, spec.n_y), spec.length_square,
                                   spec.length_marker, dictionary, ids)
    return board.generateImage((1000, 1000), marginSize=40)


# --- protocol satisfaction ---------------------------------------------------------------

def test_all_three_board_types_satisfy_the_protocol():
    assert isinstance(CheckerboardBoard(CB_SPEC), Board)
    assert isinstance(CharucoBoard([BoardSpec(5, 5, 0.04, 0.03, 0.192)]), Board)
    assert isinstance(AprilGridBoard(AprilGridTarget()), Board)


def test_to_correspondences_unzips_observations():
    obs = [Observation(points_3d=np.zeros((3, 3)), pixels=np.zeros((3, 2)),
                       visibility=np.array([True, False, True]), frame_id=0),
           Observation(points_3d=np.ones((2, 3)), pixels=np.ones((2, 2)),
                       visibility=np.array([True, True]), frame_id=1)]
    X, uv, vis = to_correspondences(obs)
    assert len(X) == len(uv) == len(vis) == 2
    assert np.array_equal(X[0], obs[0].points_3d) and np.array_equal(uv[1], obs[1].pixels)
    assert np.array_equal(vis[0], [True, False, True])


# --- checkerboard -------------------------------------------------------------------------

def test_checkerboard_board_detects_a_rendered_image(tmp_path):
    img = _render_checkerboard(CB_SPEC)
    path = str(tmp_path / "cb.png")
    cv2.imwrite(path, img)

    obs = CheckerboardBoard(CB_SPEC).detect([path])
    assert len(obs) == 1
    assert obs[0].points_3d.shape == (CB_SPEC.n_corners, 3)
    assert obs[0].pixels.shape == (CB_SPEC.n_corners, 2)
    assert obs[0].visibility.all()
    assert obs[0].frame_id == 0


def test_checkerboard_board_skips_images_with_no_board(tmp_path):
    blank = np.full((300, 300), 128, dtype=np.uint8)
    path = str(tmp_path / "blank.png")
    cv2.imwrite(path, blank)
    assert CheckerboardBoard(CB_SPEC).detect([path]) == []


# --- charuco, multi-board -------------------------------------------------------------------

def test_charuco_board_multi_board_each_sighting_is_independent():
    """Two DIFFERENT physical ChArUco boards, one per image: CharucoBoard must attribute
    each image's detection to the correct board's own geometry -- no fusion needed."""
    spec_a = BoardSpec(n_x=5, n_y=5, length_square=0.04, length_marker=0.03, square_size=0.192)
    spec_b = BoardSpec(n_x=4, n_y=4, length_square=0.05, length_marker=0.035, square_size=0.1)
    board = CharucoBoard([spec_a, spec_b], legacy=False)

    img_a = _render_charuco(spec_a, id_offset=0)
    img_b = _render_charuco(spec_b, id_offset=spec_a.n_markers)

    obs_a = board.detect([_write(img_a)])
    obs_b = board.detect([_write(img_b)])

    assert len(obs_a) == 1 and len(obs_b) == 1
    assert obs_a[0].points_3d.shape[0] <= spec_a.n_corners
    assert obs_b[0].points_3d.shape[0] <= spec_b.n_corners
    # board b's square_size is smaller -- its recovered 3D extent must be smaller too
    assert obs_a[0].points_3d[:, :2].max() > obs_b[0].points_3d[:, :2].max()


def _write(img):
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".png")
    import os
    os.close(fd)
    cv2.imwrite(path, img)
    return path


def test_charuco_board_both_boards_in_one_image_yield_two_observations():
    """Two boards visible in the SAME image: two independent Observations, not one fused
    one -- the concrete mechanism behind "ChArUco can have multiple boards at once"."""
    spec = BoardSpec(n_x=5, n_y=5, length_square=0.04, length_marker=0.03, square_size=0.192)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    ids_a = np.arange(spec.n_markers, dtype=np.int32)
    ids_b = np.arange(spec.n_markers, dtype=np.int32) + spec.n_markers
    board_a = cv2.aruco.CharucoBoard((spec.n_x, spec.n_y), spec.length_square,
                                     spec.length_marker, dictionary, ids_a)
    board_b = cv2.aruco.CharucoBoard((spec.n_x, spec.n_y), spec.length_square,
                                     spec.length_marker, dictionary, ids_b)
    img_a = board_a.generateImage((480, 480), marginSize=20)
    img_b = board_b.generateImage((480, 480), marginSize=20)
    canvas = np.full((480, 1000), 255, dtype=np.uint8)
    canvas[:, :480] = img_a
    canvas[:, 520:1000] = img_b

    cb = CharucoBoard([spec, spec], legacy=False)
    obs = cb.detect([_write(canvas)])
    assert len(obs) == 2
    assert {o.frame_id for o in obs} == {0}


# --- aprilgrid (fake detector, matching tests/calib/test_detect.py's style) --------------

def test_aprilgrid_board_builds_observations_from_detections(tmp_path, monkeypatch):
    target = AprilGridTarget(tag_rows=2, tag_cols=2, tag_size=0.1, tag_spacing=0.3)
    fake_corners = {0: np.array([[0., 0], [1, 0], [1, 1], [0, 1]]),
                   1: np.array([[2., 0], [3, 0], [3, 1], [2, 1]])}

    calls = []

    def _fake_detect_aprilgrid(paths, **kwargs):
        calls.append(list(paths))
        return [fake_corners]              # one image, both tags found

    monkeypatch.setattr(board_mod, "detect_aprilgrid", _fake_detect_aprilgrid)
    path = str(tmp_path / "grid.png")
    obs = AprilGridBoard(target).detect([path])

    assert len(obs) == 1
    assert obs[0].points_3d.shape == (8, 3)          # 2 tags x 4 corners
    assert obs[0].pixels.shape == (8, 2)
    assert obs[0].frame_id == 0
    assert calls == [[path]]


def test_aprilgrid_board_skips_frames_with_no_detections(tmp_path, monkeypatch):
    target = AprilGridTarget(tag_rows=2, tag_cols=2)

    def _fake_detect_aprilgrid(paths, **kwargs):
        return []                          # dropped below min_tags

    monkeypatch.setattr(board_mod, "detect_aprilgrid", _fake_detect_aprilgrid)
    obs = AprilGridBoard(target).detect([str(tmp_path / "empty.png")])
    assert obs == []


# --- progress_cb fires once per image, regardless of detection success -------------------

def test_checkerboard_board_progress_cb_fires_once_per_image_including_failures(tmp_path):
    good = tmp_path / "good.png"
    bad = tmp_path / "bad.png"
    cv2.imwrite(str(good), _render_checkerboard(CB_SPEC))
    cv2.imwrite(str(bad), np.full((200, 200), 255, dtype=np.uint8))   # no board -> detection fails
    calls = []
    obs = CheckerboardBoard(CB_SPEC).detect(
        [str(good), str(bad)], progress_cb=lambda i, n, path: calls.append((i, n, path)))
    assert len(obs) == 1                                          # only "good" detected
    assert calls == [(1, 2, str(good)), (2, 2, str(bad))]          # fires for both regardless


# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-CALIB-006")
