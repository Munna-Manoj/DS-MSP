"""Leaving chosen views or boards out of a calibration (FR-RIG-034)."""
import json
import os
import warnings

import numpy as np
import pytest

from ds_msp.io.mccalib import save_mccalib_detected_keypoints
from ds_msp.rig import exclude as ex
from ds_msp.rig.calib_param import calibrate_from_config, load_config
from ds_msp.rig.types import BoardObs

from ._synth import make_rig

pytestmark = pytest.mark.req("FR-RIG-034")


def test_entries_are_parsed_and_bad_ones_named():
    assert ex.parse_exclusions(["0:3", " 2:10:1 ", "0:3"]) == {(0, 3, None), (2, 10, 1)}
    assert ex.describe({(2, 10, 1), (0, 3, None), (0, 3, 0)}) == ["0:3", "0:3:0", "2:10:1"]
    assert ex.parse_exclusions(None) == set()
    for bad in ("3", "a:b", "0:1:2:3", "-1:2", "*:4"):
        with pytest.raises(ValueError, match="cam:frame"):
            ex.parse_exclusions([bad])


def test_board_observations_are_dropped_by_view_or_by_board():
    obs = [BoardObs(cam_id=c, frame_id=f, board_id=b, corner_ids=np.arange(4), pts_2d=np.zeros((4, 2)))
           for c in (0, 1) for f in (0, 1) for b in (0, 1)]
    kept, used = ex.drop_board_obs(obs, {(0, 1, None), (1, 0, 1), (5, 5, None)})
    assert used == {(0, 1, None), (1, 0, 1)}
    left = {(o.cam_id, o.frame_id, o.board_id) for o in kept}
    assert left == {(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)}
    assert ex.drop_board_obs(obs, set()) == (obs, set())


def test_target_observations_lose_only_the_excluded_boards_corners():
    obj, obs, _size, _ext, _models = make_rig(n_cam=2, n_frame=3, noise_px=0.0, seed=1, multi_board=True)
    objects = {0: obj}
    target = next(o for o in obs if o.cam_id == 1 and o.frame_id == 2)
    boards_before = set(np.asarray(obj.pts_obj_2_board)[np.asarray(target.point_rows), 0].tolist())
    assert len(boards_before) >= 2
    gone = sorted(boards_before)[0]
    n_before = len(obs)
    kept, used = ex.drop_object_obs(obs, objects, {(0, 0, None), (1, 2, gone)})
    assert used == {(0, 0, None), (1, 2, gone)} and len(kept) == n_before - 1
    assert not any(o.cam_id == 0 and o.frame_id == 0 for o in kept)
    after = next(o for o in kept if o.cam_id == 1 and o.frame_id == 2)
    assert gone not in np.asarray(obj.pts_obj_2_board)[np.asarray(after.point_rows), 0]
    assert len(after.pts_2d) == len(after.point_rows) > 0


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    root = tmp_path_factory.mktemp("exclude")
    obj, obs, img_size, _ext, _models = make_rig(n_cam=3, n_frame=16, noise_px=0.05, seed=3, multi_board=False)
    kp = root / "kp" / "detected_keypoints_data.yml"
    kp.parent.mkdir()
    save_mccalib_detected_keypoints(obs, obj, img_size, str(kp))
    cfg = root / "calib_param.yml"
    cfg.write_text(f'%YAML:1.0\nnumber_camera: 3\nnumber_board: 1\nnumber_x_square: 5\nnumber_y_square: 5\n'
                   f'square_size: 0.1\nkeypoints_path: "{kp}"\nverbose: "false"\nwebviewer: "false"\n')
    return root, cfg


def _run(cfg, out, **overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return calibrate_from_config(str(cfg), {"save_path": str(out) + os.sep, **overrides})


def test_excluded_views_take_no_part_and_are_recorded(capture, capsys):
    root, cfg = capture
    assert load_config(str(cfg)).exclude_views == []
    assert load_config(str(cfg), {"exclude_views": "1:4, 0:2"}).exclude_views == ["0:2", "1:4"]
    with pytest.raises(ValueError, match="cam:frame"):
        load_config(str(cfg), {"exclude_views": "first photo"})

    res = _run(cfg, root / "out", exclude_views="1:4,0:2,2:99")
    seen = {(o.cam_id, o.frame_id) for o in res["scenario"].object_obs}
    assert (1, 4) not in seen and (0, 2) not in seen and (1, 5) in seen and (2, 4) in seen
    acc = json.loads((root / "out" / "calibration_acceptance.json").read_text())
    assert acc["excluded_views"] == ["0:2", "1:4"]
    assert "2:99 matches no view" in capsys.readouterr().err
    z = np.load(root / "out" / "diagnostics" / "residuals.npz")
    assert not np.any((z["cam"] == 1) & (z["frame"] == 4))

    with pytest.raises(ValueError, match="leaves no view"):
        _run(cfg, root / "none", exclude_views=",".join(f"{c}:{f}" for c in range(3) for f in range(16)))


def test_without_exclusions_the_calibration_is_unchanged(capture):
    """The setting does nothing when it is empty: same cameras with and without the key."""
    root, cfg = capture
    a = _run(cfg, root / "plain")
    b = _run(cfg, root / "empty", exclude_views="")
    for c in a["rig"].cameras:
        np.testing.assert_array_equal(np.asarray(a["rig"].cameras[c].params), np.asarray(b["rig"].cameras[c].params))
        np.testing.assert_array_equal(a["rig"].T_c_g[c], b["rig"].T_c_g[c])
    assert json.loads((root / "plain" / "calibration_acceptance.json").read_text())["excluded_views"] == []
