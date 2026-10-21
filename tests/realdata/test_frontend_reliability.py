"""Dataset-gated frontend regression checks on the local two-camera Seltos capture.

Set DS_MSP_SELTOS_DATA to a directory containing Cam_001 and Cam_002. These are
behavior checks on real pixels; the capture has no independent metric pose truth.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from ds_msp.calib.board import CheckerboardBoard
from ds_msp.detect.charuco import BoardSpec
from ds_msp.detect.checkerboard import CheckerboardSpec
from ds_msp.rig.reconstruct import detect_board_obs_images

pytestmark = [pytest.mark.realdata, pytest.mark.req("FR-CALIB-005", "FR-RIG-025")]


@pytest.fixture
def data_root():
    root = Path(os.environ.get("DS_MSP_SELTOS_DATA", Path(__file__).resolve().parents[2] / "data"))
    if not (root / "calib_param_gaze.yml").exists():
        pytest.skip("Seltos capture unavailable; set DS_MSP_SELTOS_DATA")
    return root


def test_real_checker_lattice_oversized_correspondences(data_root):
    # The photographs contain ChArUco targets; SB detects the underlying lattice.
    files = sorted((data_root / "Cam_001").glob("*.png"))[:8]
    observations = CheckerboardBoard(CheckerboardSpec(4, 3, 55, larger=True)).detect(
        [str(p) for p in files])
    assert observations
    assert any(len(o.pixels) > 12 for o in observations)
    for o in observations:
        assert len(o.pixels) == len(o.points_3d)
        assert np.isfinite(o.pixels).all()


def test_real_camera_prefix_does_not_change_frame_association(data_root, tmp_path):
    specs = [BoardSpec(7, 5, .055, .04, 55)] * 2
    first, sizes = detect_board_obs_images(str(data_root), [0, 1], specs, workers=2)
    for c in [0, 1]:
        directory = tmp_path / f"Cam_{c+1:03d}"
        directory.mkdir()
        for p in (data_root / directory.name).glob("*.png"):
            (directory / f"cam{c+1}_frame{p.stem}.png").symlink_to(p.resolve())
    second, sizes2 = detect_board_obs_images(str(tmp_path), [0, 1], specs, workers=2)
    def indexed(obs):
        return {(o.cam_id, o.frame_id, o.board_id): o for o in obs}
    a, b = indexed(first), indexed(second)
    assert a and a.keys() == b.keys() and sizes == sizes2
    for key in a:
        np.testing.assert_array_equal(a[key].corner_ids, b[key].corner_ids)
        np.testing.assert_array_equal(a[key].pts_2d, b[key].pts_2d)
