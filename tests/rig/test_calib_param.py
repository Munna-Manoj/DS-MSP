"""MC-Calib calib_param.yml parsing + the single-file ``calibrate_from_config`` entry."""
import os
import textwrap
from pathlib import Path

import numpy as np
import pytest

from ds_msp.rig.calib_param import calibrate_from_config, load_config

_TEMPLATE = Path(__file__).parents[2] / "configs" / "calib_param.template.yml"
_INTRINSICS_TEMPLATE = Path(__file__).parents[2] / "configs" / "camera_intrinsics.template.yml"
_KEYPOINTS_TEMPLATE = Path(__file__).parents[2] / "configs" / "calib_param.keypoints.template.yml"


@pytest.mark.req("FR-RIG-004")
def test_keypoints_reuse_template_exists_and_parses():
    """The detect-once / calibrate-many template must ship and parse: keypoints reuse is the
    documented fast path (no image detection), so its template is a supported, tested artifact."""
    assert _KEYPOINTS_TEMPLATE.exists(), f"missing keypoints template at {_KEYPOINTS_TEMPLATE}"
    cfg = load_config(str(_KEYPOINTS_TEMPLATE))
    assert cfg.number_camera >= 1 and cfg.number_board >= 1
    assert cfg.keypoints_path is not None        # reuse mode points at a saved keypoints file
    assert _KEYPOINTS_TEMPLATE.read_text().lstrip().startswith("%YAML:1.0")


@pytest.mark.req("FR-RIG-002", "FR-RIG-005")
def test_base_template_exists_and_parses():
    """DS-MSP[rig] must always ship a base, parseable MC-Calib-compatible config template.

    Guards against the template going missing or drifting out of sync with the parser."""
    assert _TEMPLATE.exists(), f"missing base config template at {_TEMPLATE}"
    cfg = load_config(str(_TEMPLATE))
    # MC-Calib core keys are honored
    assert cfg.number_camera >= 1
    assert cfg.number_board >= 1 and len(cfg.boards) == cfg.number_board
    assert cfg.boards[0].n_x >= 2 and cfg.boards[0].n_y >= 2
    # default distortion_model 0 -> radtan; per-camera model list is materialized
    assert cfg.camera_models and all(isinstance(m, str) for m in cfg.camera_models)
    # the CLI --init-config copies exactly this file
    assert _TEMPLATE.read_text().lstrip().startswith("%YAML:1.0")


def _write_cfg(tmp_path, body):
    p = tmp_path / "calib_param.yml"
    p.write_text("%YAML:1.0\n" + textwrap.dedent(body))
    return str(p)


@pytest.mark.req("FR-RIG-002", "FR-RIG-005")
def test_parse_models_precedence(tmp_path):
    # camera_models (extension) overrides distortion_model
    cfg = load_config(_write_cfg(tmp_path, """
        number_camera: 3
        number_board: 1
        number_x_square: 5
        number_y_square: 5
        square_size: 0.192
        distortion_model: 0
        camera_models: [ ds, ucm, eucm ]
        keypoints_path: "None"
        root_path: "None"
    """))
    assert cfg.camera_models == ["ds", "ucm", "eucm"]
    assert cfg.number_board == 1 and cfg.boards[0].n_x == 5


@pytest.mark.req("FR-RIG-005")
def test_distortion_model_mapping(tmp_path):
    # 0 -> radtan (Brown), 1 -> kb (Kannala); per-camera overrides the global
    cfg = load_config(_write_cfg(tmp_path, """
        number_camera: 3
        number_board: 1
        distortion_model: 1
        distortion_per_camera: [ 0, 1, 0 ]
        keypoints_path: "None"
    """))
    assert cfg.camera_models == ["radtan", "kb", "radtan"]


_S2 = "../MC-Calib/Blender_Images/Scenario_2"


@pytest.mark.req("FR-RIG-002")
@pytest.mark.skipif(not os.path.isdir(os.path.join(_S2, "Images")),
                    reason="Blender Scenario_2 images not present")
def test_calibrate_from_config_raw_images(tmp_path):
    """One config file → detect from raw images → MC-Calib output, extrinsics within 1%."""
    cfgp = "../MC-Calib/configs/Blender_Images/calib_param_synth_Scenario2.yml"
    ov = {"root_path": os.path.abspath(os.path.join(_S2, "Images")),
          "keypoints_path": "None", "save_path": str(tmp_path),
          "camera_models": ["radtan", "ucm", "eucm", "ds", "radtan"]}
    res = calibrate_from_config(cfgp, ov)
    assert res["models"] == {0: "radtan", 1: "ucm", 2: "eucm", 3: "ds", 4: "radtan"}
    assert res["metrics"]["worst_baseline_pct_vs_gt"] < 1.0
    for fn in ("calibrated_cameras_data.yml", "calibrated_objects_data.yml",
               "calibrated_objects_pose_data.yml"):
        assert (tmp_path / fn).exists()


# --- intrinsic initialization from cam_params_path (init_K seeding + model conversion) ---

@pytest.mark.req("FR-RIG-003")
def test_init_model_native_is_exact_and_conversion_reproduces_lens():
    """``_init_model`` returns the file's native params for the matching model, and converts the
    *same lens* into another model so it reprojects the source over the FOV (sub-pixel for a
    moderate fisheye — the "two models, one camera" identity used by the fixed-intrinsic path)."""
    from ds_msp.io.mccalib import CameraGT
    from ds_msp.models.kb import KannalaBrandtModel
    from ds_msp.rig.calib_param import _init_model, _source_model

    K = np.array([[420.0, 0, 320.0], [0, 421.0, 240.0], [0, 0, 1.0]])
    kb = CameraGT(K=K, dist=np.array([-0.02, 0.004, -0.001, 0.0003]), pose=np.eye(4))
    wh = (640, 480)

    native = _init_model(kb, "kb", wh)                       # native: exact stored params
    assert native.name == "kb"
    assert np.allclose(native.params, KannalaBrandtModel(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                                                         *kb.dist).params)

    src = _source_model(kb)
    uu, vv = np.meshgrid(np.linspace(0.1 * wh[0], 0.9 * wh[0], 30),
                         np.linspace(0.1 * wh[1], 0.9 * wh[1], 30))
    pix = np.column_stack([uu.ravel(), vv.ravel()])
    rays, val = src.unproject(pix)
    val = np.asarray(val).ravel().astype(bool)
    rays, pix = np.asarray(rays, float)[val], pix[val]
    for name in ("ucm", "eucm", "ds"):
        m = _init_model(kb, name, wh)
        assert m.name == name
        uv, ok = m.project(rays)
        ok = np.asarray(ok).ravel().astype(bool)
        rms = float(np.sqrt(np.mean(np.sum((uv[ok] - pix[ok]) ** 2, axis=1))))
        assert rms < 1.0, f"{name} conversion RMS {rms:.3f}px"


# --- camera_parameters yaml: explicit per-camera model + the intrinsics policy ---------------

@pytest.mark.req("FR-RIG-003", "FR-RIG-005")
def test_intrinsics_template_states_and_round_trips_every_model():
    """The shipped camera-intrinsics template names every DS-MSP model and round-trips through the
    reader + reconstructor to the exact same model (the inverse of how the rig serializes them)."""
    from ds_msp.io.mccalib import _load_cameras
    from ds_msp.rig.calib_param import _provided_model_name, _source_model

    assert _INTRINSICS_TEMPLATE.exists(), f"missing intrinsics template at {_INTRINSICS_TEMPLATE}"
    cams, _ = _load_cameras(str(_INTRINSICS_TEMPLATE))
    expect = {0: "radtan", 1: "kb", 2: "ucm", 3: "eucm",
              4: "ds", 5: "dsplus", 6: "eucmplus", 7: "ocam"}
    assert set(cams) == set(expect)
    for c, name in expect.items():
        assert _provided_model_name(cams[c]) == name          # model read from the file's statement
        assert _source_model(cams[c]).name == name            # and reconstructed natively


def _cfg(cam_params_path, fix, models):
    from ds_msp.rig.calib_param import RigConfig
    return RigConfig(boards=[], number_camera=len(models), camera_models=models,
                     root_path=None, cam_prefix="Cam_", keypoints_path=None,
                     fix_intrinsic=fix, cam_params_path=cam_params_path, save_path=None,
                     camera_params_file_name="", save_detection=False, save_reprojection=False,
                     ransac_threshold=10.0, number_iterations=1000, he_approach=0)


@pytest.mark.req("FR-RIG-003")
def test_fix_intrinsic_holds_matching_model_natively():
    from ds_msp.rig.calib_param import _resolve_init_intrinsics
    cfg = _cfg(str(_INTRINSICS_TEMPLATE), True, ["radtan"])
    init_cameras, init_K = _resolve_init_intrinsics(cfg, [0], {0: "radtan"}, {0: (1280, 960)})
    assert init_cameras[0].name == "radtan" and 0 in init_K


@pytest.mark.req("FR-RIG-003")
def test_fix_intrinsic_model_mismatch_is_an_error():
    from ds_msp.rig.calib_param import _resolve_init_intrinsics
    cfg = _cfg(str(_INTRINSICS_TEMPLATE), True, ["ds"])     # cam 0 is stored as radtan
    with pytest.raises(ValueError, match="must match"):
        _resolve_init_intrinsics(cfg, [0], {0: "ds"}, {0: (1280, 960)})


@pytest.mark.req("FR-RIG-003")
def test_fix_intrinsic_missing_camera_is_an_error():
    from ds_msp.rig.calib_param import _resolve_init_intrinsics
    cfg = _cfg(str(_INTRINSICS_TEMPLATE), True, ["radtan"])
    with pytest.raises(ValueError, match="no intrinsics"):
        _resolve_init_intrinsics(cfg, [99], {99: "radtan"}, {99: (1280, 960)})


@pytest.mark.req("FR-RIG-003")
def test_fix_intrinsic_without_file_is_an_error():
    from ds_msp.rig.calib_param import _resolve_init_intrinsics
    cfg = _cfg(None, True, ["radtan"])
    with pytest.raises(FileNotFoundError):
        _resolve_init_intrinsics(cfg, [0], {0: "radtan"}, {0: (1280, 960)})


@pytest.mark.req("FR-RIG-003")
def test_optimize_model_mismatch_warns_and_converts(capsys):
    from ds_msp.rig.calib_param import _resolve_init_intrinsics
    cfg = _cfg(str(_INTRINSICS_TEMPLATE), False, ["ds"])    # cam 0 stored radtan, chosen ds
    init_cameras, init_K = _resolve_init_intrinsics(cfg, [0], {0: "ds"}, {0: (1280, 960)})
    assert init_cameras[0].name == "ds" and 0 in init_K     # converted into the chosen model
    err = capsys.readouterr().err
    assert "WARNING" in err and "convert()" in err


@pytest.mark.req("FR-RIG-003")
def test_no_intrinsics_file_initializes_from_scratch():
    from ds_msp.rig.calib_param import _resolve_init_intrinsics
    cfg = _cfg(None, False, ["radtan"])
    assert _resolve_init_intrinsics(cfg, [0], {0: "radtan"}, {0: (1280, 960)}) == (None, None)


def test_init_K_seeds_front_end_and_bypasses_heterogeneous_consensus():
    """A provided ``init_K`` seeds each camera's focal and bypasses the cross-camera focal
    consensus reset, so a rig whose cameras have very different focals (a mixed-resolution /
    mixed-lens rig) is not collapsed onto the median focal."""
    from ds_msp.models.radtan import RadTanModel
    from ds_msp.rig.calibrate import make_bundle_front_end, paraxial_focal
    from ._synth import make_rig

    w, h = 1280, 960

    def mf(cam_id, rng):
        f = 820.0 if cam_id == 0 else 380.0                  # >25% apart -> consensus would reset
        return RadTanModel(f, f, w / 2, h / 2, -0.05, 0.01, 0.0, 0.0, 0.0)

    obj, obs, img_size, _gt_ext, gt_models = make_rig(n_cam=2, n_frame=40, w=w, h=h,
                                                      model_factory=mf, seed=3)
    from collections import defaultdict
    obs_by_cam = defaultdict(list)
    for o in obs:
        obs_by_cam[o.cam_id].append(o)
    init_K = {c: gt_models[c].K for c in (0, 1)}
    cams = make_bundle_front_end({0: "radtan", 1: "radtan"}, init_K=init_K)(obj, obs_by_cam, img_size)
    # the low-focal camera keeps its own focal, not the ~600 median it would be reset to.
    assert abs(paraxial_focal(cams[1])[0] - 380.0) / 380.0 < 0.1
    assert abs(paraxial_focal(cams[0])[0] - 820.0) / 820.0 < 0.1
