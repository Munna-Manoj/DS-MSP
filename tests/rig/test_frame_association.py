"""Frame association from image filenames (FR-RIG-025): explicit ``frame_pattern``, and
rejection of names whose trailing number is not a frame index."""
import os

import cv2
import numpy as np
import pytest

from ds_msp.detect.charuco import (_frame_id_from_name, _validate_frame_association,
                                   _validate_frame_names)
from ds_msp.rig.calib_param import load_config

pytestmark = pytest.mark.req("FR-RIG-025")


@pytest.mark.parametrize("name,frame", [("000007.png", 7), ("cam1_frame000007.png", 7),
                                         ("000007_left.png", None), ("img_7.jpg", 7)])
def test_default_reads_trailing_frame_number(name, frame):
    if frame is None:
        with pytest.raises(ValueError, match="frame number"):
            _frame_id_from_name(name)
    else:
        assert _frame_id_from_name(name) == frame


@pytest.mark.parametrize("name", ["1697551234.123456.png", "1697551234_123456789.png"])
def test_timestamp_names_are_rejected_not_truncated(name):
    with pytest.raises(ValueError, match="timestamp"):
        _frame_id_from_name(name)


@pytest.mark.parametrize("name", ["img_7_cam1.png", "frame7-cam2.jpg", "000012_camera03.png"])
def test_trailing_camera_id_is_rejected(name):
    with pytest.raises(ValueError, match="camera id"):
        _frame_id_from_name(name)


def test_frame_pattern_selects_the_frame_group():
    assert _frame_id_from_name("img_7_cam1.png", r"img_(?P<frame>\d+)_cam") == 7
    assert _frame_id_from_name("frame7-cam2.jpg", r"frame(\d+)") == 7
    with pytest.raises(ValueError, match="does not match"):
        _frame_id_from_name("other.png", r"frame(\d+)")
    with pytest.raises(ValueError, match="integer"):
        _frame_id_from_name("t1697551234.5.png", r"t(?P<frame>[\d.]+)")


def test_duplicates_and_disjoint_cameras_fail():
    with pytest.raises(ValueError, match="duplicate frame 7"):
        _validate_frame_names(["a/000007.png", "a/x_7.png"])
    assert _validate_frame_names(["a/000007.png", "a/000008.png"]) == {7, 8}
    _validate_frame_association({0: {1, 2}, 1: {2, 3}})
    _validate_frame_association({0: {1, 2}})
    with pytest.raises(ValueError, match="no frame index is shared"):
        _validate_frame_association({0: {1, 2}, 1: {3, 4}})


def test_detect_rig_honours_frame_pattern(tmp_path):
    from ds_msp.detect.charuco import BoardSpec, detect_rig, single_board_object
    spec = BoardSpec(n_x=5, n_y=4, length_square=0.04, length_marker=0.03, square_size=0.04)
    for cam in (1, 2):
        d = tmp_path / f"Cam_{cam:03d}"
        d.mkdir()
        for fr in (3, 4):
            cv2.imwrite(str(d / f"shot_{fr}_cam{cam}.png"), np.full((60, 80), 255, np.uint8))
    obj = single_board_object(spec)
    with pytest.raises(ValueError, match="camera id"):
        detect_rig(str(tmp_path), [0, 1], [spec], obj, workers=1)
    obs, _ = detect_rig(str(tmp_path), [0, 1], [spec], obj, workers=1,
                        frame_pattern=r"shot_(?P<frame>\d+)_cam")
    assert obs == []                                # blank images: names parsed, no boards


def test_config_frame_pattern(tmp_path):
    cfg = tmp_path / "calib_param.yml"
    cfg.write_text("%YAML:1.0\n---\nnumber_camera: 1\nnumber_x_square: 5\nnumber_y_square: 5\n"
                   "root_path: \"\"\nsave_path: \"\"\nframe_pattern: \"img_(?P<frame>\\\\d+)_\"\n")
    assert load_config(str(cfg)).frame_pattern == r"img_(?P<frame>\d+)_"
    assert load_config(str(cfg), {"frame_pattern": ""}).frame_pattern is None
    assert os.path.exists(cfg)
