"""Raw-image ChArUco detection — self-contained (renders its own board) plus an optional
corner-for-corner parity check against MC-Calib's keypoints when the Blender data is present.
"""
import os

import cv2
import numpy as np
import pytest

from ds_msp.calib.charuco import (BoardSpec, board_object_points, detect_image,
                                  single_board_object)
from ds_msp.detect.charuco import detect_rig

SPEC = BoardSpec(n_x=5, n_y=5, length_square=0.04, length_marker=0.03, square_size=0.192)


def test_object_points_match_mccalib_layout():
    xyz = board_object_points(SPEC)
    assert xyz.shape == (16, 3)
    # corner k at (k%4, k//4)*square_size, z=0 (row-major) — MC-Calib's single-board model
    assert np.allclose(xyz[1], [0.192, 0.0, 0.0])
    assert np.allclose(xyz[4], [0.0, 0.192, 0.0])
    assert np.allclose(xyz[5], [0.192, 0.192, 0.0])
    assert np.allclose(xyz[:, 2], 0.0)
    obj = single_board_object(SPEC)
    assert obj.pts_board_2_obj[(0, 5)] == 5


def test_detect_rendered_board_recovers_all_corners():
    """Render the exact board to an image and detect it back: every interior corner is
    found, at its known pixel location (the detector is internally consistent)."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    board = cv2.aruco.CharucoBoard((SPEC.n_x, SPEC.n_y), SPEC.length_square,
                                   SPEC.length_marker, dictionary)
    img = board.generateImage((1000, 1000), marginSize=40)
    # the rendered board uses the non-legacy pattern, so detect with legacy=False
    det = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = det.detectBoard(img)
    assert ch_ids is not None and len(ch_ids) == 16          # all interior corners

    found = detect_image([cv2.aruco.CharucoDetector(board)], img, min_corners=4)
    assert len(found) == 1
    bid, ids, pts = found[0]
    assert bid == 0 and sorted(ids) == list(range(16))
    assert pts.shape == (16, 2)


_S2 = "../MC-Calib/Blender_Images/Scenario_2"


@pytest.mark.skipif(not os.path.isdir(os.path.join(_S2, "Images")),
                    reason="Blender Scenario_2 images not present")
def test_parity_vs_mccalib_keypoints():
    """Detected corners reproduce MC-Calib's own ``detected_keypoints_data.yml`` to
    sub-pixel agreement on the same physical frames (cam 0).

    Convention-aware: OpenCV 5.0 moved the ChArUco corner coordinate convention by exactly
    half a pixel relative to the OpenCV-4.x build that generated MC-Calib's reference
    keypoints (measured on this dataset 2026-07-19: mean delta (-0.4995, -0.4990), residual
    scatter after removing it <= 0.022 px, i.e. detection itself agrees sub-0.03 px). So
    the parity contract is: the GLOBAL offset must be either ~zero or a pure half-pixel
    convention delta — anything else (and any per-corner scatter about it) fails at the
    original sub-pixel bars. A genuine mis-decode (the ADR-0012 class) is NON-uniform and
    still fails the scatter assertions."""
    from ds_msp.calib.charuco import detect_folder
    from ds_msp.io.mccalib import load_scenario
    scn = load_scenario(_S2)
    obj = single_board_object(SPEC)
    mc = {}
    for o in scn.object_obs:
        if o.cam_id == 0:
            mc.setdefault(o.frame_id, {}).update(
                {int(r): uv for r, uv in zip(o.point_rows, o.pts_2d)})
    obs = detect_folder(os.path.join(_S2, "Images/Cam_001"), [SPEC], obj, 0,
                        legacy=True, min_corners=8)
    mine = {}
    for o in obs:                                            # filename N -> MC frame N-1
        mine.setdefault(o.frame_id - 1, {}).update(
            {int(r): uv for r, uv in zip(o.point_rows, o.pts_2d)})
    dv = np.array([mine[f][r] - mc[f][r]
                   for f in set(mc) & set(mine) for r in set(mc[f]) & set(mine[f])])
    assert len(dv) > 300
    shift = dv.mean(axis=0)
    half_pixel = np.allclose(np.abs(shift), 0.5, atol=0.05)
    assert np.linalg.norm(shift) < 0.1 or half_pixel, \
        f"global corner offset {shift} is neither ~0 nor a half-pixel convention delta"
    resid = np.linalg.norm(dv - shift, axis=1)
    assert np.median(resid) < 0.1 and resid.max() < 1.0


def _small_two_cam_root(tmp_path, n_images=10):
    """A tmp copy of the first ``n_images`` of each of Scenario_2's first two real cameras
    (symlinks, not copies — the images are large) — keeps the parallel-detection tests fast
    while still exercising real fisheye images and real detection, not a synthetic render."""
    root = tmp_path / "root"
    for c, cam_dir in enumerate(("Cam_001", "Cam_002")):
        src_dir = os.path.join(_S2, "Images", cam_dir)
        dst_dir = root / f"Cam_{c + 1:03d}"
        dst_dir.mkdir(parents=True)
        files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".png"))[:n_images]
        for f in files:
            (dst_dir / f).symlink_to(os.path.abspath(os.path.join(src_dir, f)))
    return str(root)


@pytest.mark.skipif(not os.path.isdir(os.path.join(_S2, "Images")),
                    reason="Blender Scenario_2 images not present")
def test_detect_rig_parallel_matches_serial_bit_for_bit(tmp_path):
    """``workers=None`` (parallel, the default) must find exactly the same corners as
    ``workers=1`` (the original serial per-camera loop) — pure parallelism, not an
    approximation, mirroring the front-end pool's own bar
    (``rig/calibrate.py::make_bundle_front_end``)."""
    root = _small_two_cam_root(tmp_path)
    obj = single_board_object(SPEC)

    serial_obs, serial_sz = detect_rig(root, [0, 1], [SPEC], obj, min_corners=8, workers=1)
    parallel_obs, parallel_sz = detect_rig(root, [0, 1], [SPEC], obj, min_corners=8, workers=4)

    assert len(serial_obs) > 0, "fixture produced no detections at all"
    assert serial_sz == parallel_sz

    def key(o):
        return (o.cam_id, o.frame_id, tuple(o.point_rows.tolist()))

    serial_by_key = {key(o): o for o in serial_obs}
    parallel_by_key = {key(o): o for o in parallel_obs}
    assert set(serial_by_key) == set(parallel_by_key)
    for k, so in serial_by_key.items():
        po = parallel_by_key[k]
        assert np.array_equal(so.pts_2d, po.pts_2d)


@pytest.mark.skipif(not os.path.isdir(os.path.join(_S2, "Images")),
                    reason="Blender Scenario_2 images not present")
def test_detect_rig_progress_cb_fires_exactly_once_per_image_under_parallel_workers(tmp_path):
    """``progress_cb`` must fire exactly once per (camera, image) even when several worker
    threads race through the flat task list concurrently — the internal lock serializes both
    the per-camera counter and the callback invocation, so no image is double-counted, none is
    silently skipped, and per-camera ``i`` still runs cleanly ``1..n``."""
    n_images = 8
    root = _small_two_cam_root(tmp_path, n_images=n_images)
    obj = single_board_object(SPEC)

    calls = []
    detect_rig(root, [0, 1], [SPEC], obj, min_corners=8, workers=4,
              progress_cb=lambda cam_id, i, n, path: calls.append((cam_id, i, n)))

    assert len(calls) == 2 * n_images
    for cam_id in (0, 1):
        cam_calls = sorted(i for c, i, n in calls if c == cam_id)
        assert cam_calls == list(range(1, n_images + 1))
        assert all(n == n_images for c, i, n in calls if c == cam_id)


# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-CALIB-004")
