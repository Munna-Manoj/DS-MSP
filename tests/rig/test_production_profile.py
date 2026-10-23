"""Production profile, strict configuration, MC-Calib ``boards_index`` and the extrinsic
uncertainty gate (FR-RIG-027, FR-RIG-028)."""
import json
import warnings

import numpy as np
import pytest

from ds_msp.detect.charuco import BoardSpec, detect_image, make_detectors
from ds_msp.io.mccalib import Scenario
from ds_msp.rig import pipeline
from ds_msp.rig.calib_param import load_config
from ds_msp.rig.pipeline import make_fixed_intrinsic_front_end

from . import _render
from ._synth import make_rig

pytestmark = pytest.mark.req("FR-RIG-027", "FR-RIG-028")

BASE = ("%YAML:1.0\n---\nnumber_camera: 2\nnumber_x_square: 5\nnumber_y_square: 4\n"
        "root_path: \"\"\nsave_path: \"\"\n")


def _cfg(tmp_path, extra=""):
    p = tmp_path / "calib_param.yml"
    p.write_text(BASE + extra)
    return str(p)


def test_unknown_and_ignored_keys_are_reported(tmp_path):
    with pytest.warns(UserWarning) as rec:
        cfg = load_config(_cfg(tmp_path, "refine_corner: 1\naudit_gat: \"refuse\"\n"))
    msgs = [str(w.message) for w in rec]
    assert any("'refine_corner'" in m and "ignored" in m for m in msgs)
    assert any("unknown config key 'audit_gat'" in m for m in msgs)
    assert len(cfg.config_warnings) == 2
    assert cfg.audit_gate == "warn"                       # the typo did not take effect


def test_production_profile_tightens_gates_and_refuses_typos(tmp_path):
    limits = "max_rotation_sigma_deg: 0.05\nmax_centre_sigma: 2.0\n"
    cfg = load_config(_cfg(tmp_path, "profile: \"production\"\n" + limits))
    assert (cfg.audit_gate, cfg.uncertainty_gate, cfg.reject_outliers) == ("refuse", "refuse", True)
    explicit = load_config(_cfg(tmp_path, "profile: \"production\"\naudit_gate: \"warn\"\n" + limits))
    assert explicit.audit_gate == "warn"                  # an explicit value still wins
    with pytest.raises(ValueError, match="unknown config key 'audit_gat'"):
        load_config(_cfg(tmp_path, "profile: \"production\"\naudit_gat: \"warn\"\n" + limits))
    with pytest.raises(ValueError, match="max_rotation_sigma_deg"):
        load_config(_cfg(tmp_path, "profile: \"production\"\n"))


#: Every key upstream MC-Calib v1.5.0 reads from calib_param.yml (McCalib.cpp, Board.cpp).
MCCALIB_KEYS = {
    "boards_index": "[]", "camera_params_file_name": "\"\"", "cam_params_path": "\"None\"",
    "cam_prefix": "\"Cam_\"", "distortion_model": "0", "distortion_per_camera": "[]",
    "fix_intrinsic": "0", "he_approach": "0", "keypoints_path": "\"None\"", "length_marker": "0.75",
    "length_square": "0.04", "min_perc_pts": "0.5", "number_board": "1", "number_iterations": "1000",
    "number_x_square_per_board": "[]", "number_y_square_per_board": "[]",
    "quaternion_averaging": "1", "ransac_threshold": "10", "refine_corner": "1",
    "resolution_x": "500", "resolution_y": "400", "resolution_x_per_board": "[]",
    "resolution_y_per_board": "[]", "save_detection": "0", "save_reprojection": "0",
    "square_size": "0.192", "square_size_per_board": "[]",
}


def test_production_profile_accepts_every_mccalib_key(tmp_path):
    keys = "".join(f"{k}: {v}\n" for k, v in MCCALIB_KEYS.items())
    limits = "max_rotation_sigma_deg: 0.05\nmax_centre_sigma: 2.0\n"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = load_config(_cfg(tmp_path, "profile: \"production\"\n" + limits + keys))
    assert not any(m.startswith("unknown") for m in cfg.config_warnings)


def test_overrides_reach_every_field(tmp_path):
    """``--set`` replaces file values for fields read straight from the file too (camera
    prefix, distortion model, board geometry, sequences), and an unknown override key is
    reported like an unknown file key."""
    path = _cfg(tmp_path, "cam_prefix: \"Cam_\"\ndistortion_model: 0\nnumber_board: 2\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = load_config(path, {"cam_prefix": "cam_", "distortion_model": "1",
                                 "number_x_square": "7", "boards_index": "[3, 4]",
                                 "ransac_threshold": "4.5", "audit_gat": "refuse"})
    assert cfg.cam_prefix == "cam_"
    assert cfg.camera_models == ["kb", "kb"]
    assert [b.n_x for b in cfg.boards] == [7, 7]
    assert [b.id_offset for b in cfg.boards] == [3 * 14, 4 * 14]
    assert cfg.ransac_threshold == 4.5
    assert any("unknown config key 'audit_gat'" in m for m in cfg.config_warnings)


def test_boards_index_sets_mccalib_marker_offsets(tmp_path):
    cfg = load_config(_cfg(tmp_path, "number_board: 2\nboards_index: [6, 7]\n"))
    assert [b.id_offset for b in cfg.boards] == [6 * 10, 7 * 10]
    with pytest.raises(ValueError, match="boards_index"):
        load_config(_cfg(tmp_path, "number_board: 2\nboards_index: [6]\n"))
    # a board printed as physical board 6 is only found with the matching offset
    phys = BoardSpec(5, 4, 0.04, 0.03, 40.0, id_offset=60)
    img = _render.render(*_frontal(phys))
    assert detect_image(make_detectors([phys]), img)
    assert not detect_image(make_detectors([BoardSpec(5, 4, 0.04, 0.03, 40.0)]), img)


def _frontal(spec):
    from ds_msp.models.radtan import RadTanModel
    T = np.eye(4)
    T[:3, 3] = [-80.0, -60.0, 400.0]
    return (RadTanModel(500, 500, 319.5, 239.5), (640, 480), {0: T}, [spec],
            _render.board_textures([spec]))


def _scenario():
    obj, obs, img_size, _ext, models = make_rig(n_cam=2, n_frame=14, noise_px=0.3, seed=2)
    scn = Scenario(name="unc", object=obj, object_obs=obs, cam_ids=[0, 1], img_size=img_size,
                   gt={}, mccalib={}, mccalib_rms={})
    return scn, models


def test_uncertainty_is_recorded_and_refusal_writes_nothing(tmp_path):
    scn, models = _scenario()
    ok = pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=models,
                                     save_dir=str(tmp_path / "ok"))
    unc = ok["uncertainty"]
    pair = unc["pairs"]["0-1"]
    assert 0 < pair["rotation_sigma_deg"] < 0.1 and 0 < pair["centre_sigma"] < 0.01
    from scipy import stats
    assert unc["dof"] == unc["n_frame_clusters"] - 1
    assert unc["t_factor"] == pytest.approx(stats.t.ppf(0.975, unc["dof"]) / 1.959964, rel=1e-5)
    assert pair["rotation_sigma_deg_t"] == pytest.approx(pair["rotation_sigma_deg"] * unc["t_factor"])
    accepted = json.loads((tmp_path / "ok" / "calibration_acceptance.json").read_text())
    assert accepted["extrinsic_uncertainty"]["pairs"]["0-1"]["rotation_sigma_deg"] == \
        pytest.approx(pair["rotation_sigma_deg"])
    assert accepted["rejected_observations"] == [] and accepted["profile"] == "default"

    scn, models = _scenario()
    with pytest.raises(RuntimeError, match="uncertainty gate refused"):
        pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=models,
                                    save_dir=str(tmp_path / "refused"), uncertainty_gate="refuse",
                                    max_rotation_sigma_deg=pair["rotation_sigma_deg"] / 10,
                                    max_centre_sigma=1.0)
    assert not (tmp_path / "refused").exists()


def test_outlier_ledger_reaches_acceptance(tmp_path):
    scn, models = _scenario()
    bad = scn.object_obs[5]
    bad.pts_2d = bad.pts_2d + 60.0                         # a mis-decoded board
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True,
                                          init_cameras=models, save_dir=str(tmp_path),
                                          reproj_gate_px=10.0)
    ledger = res["acceptance"]["rejected_observations"]
    assert any(r["cam_id"] == bad.cam_id and r["frame_id"] == bad.frame_id for r in ledger)
    assert json.loads((tmp_path / "calibration_acceptance.json").read_text())[
        "rejected_observations"] == ledger
    assert make_fixed_intrinsic_front_end is not None
