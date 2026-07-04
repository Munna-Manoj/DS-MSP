"""MC-Calib-format writer round-trip + model-name registry + per-camera model selection.

Confirms DS-MSP writes ``calibrated_cameras_data.yml`` etc. in MC-Calib's exact OpenCV-YAML
schema (so the result is interchangeable) and that each camera can use its own model.
"""
import numpy as np
import pytest

from ds_msp.models.registry import (canonical_name, model_class, mccalib_name,
                                     PINHOLE_MODELS, FISHEYE_MODELS)
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.dsplus import DSPlusModel
from ds_msp.models.eucm import EUCMModel
from ds_msp.models.eucmplus import EUCMPlusModel
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.ocam import OCamModel
from ds_msp.models.radtan import RadTanModel
from ds_msp.models.ucm import UCMModel
from ds_msp.io.mccalib import (save_mccalib_cameras, save_mccalib_objects,
                               save_mccalib_object_poses, _load_cameras, load_camera)
from ds_msp.rig.calibrate import calibrate_rig, make_bundle_front_end
from ds_msp.rig.pipeline import random_model_assignment
from ds_msp.rig.types import RigState
import cv2

from tests.rig._synth import make_rig

W, H = 1280, 960


def test_registry_resolves_mccalib_and_dsmsp_names():
    assert model_class("double_sphere") is DoubleSphereModel   # MC-Calib spelling
    assert model_class("ds") is DoubleSphereModel              # DS-MSP spelling
    assert model_class("0") is RadTanModel                     # legacy distortion_model int
    assert model_class("1") is KannalaBrandtModel
    assert canonical_name("Double_Sphere") == "ds"
    assert mccalib_name("ds") == "double_sphere"
    assert mccalib_name("radtan") == "radtan"
    with pytest.raises(KeyError):
        model_class("not_a_model")


def test_random_assignment_respects_pinhole_fisheye_pools():
    pin = random_model_assignment([0, 1, 2, 3, 4], kind="pinhole", seed=1)
    assert set(pin.values()) <= set(PINHOLE_MODELS)
    assert "kb" not in pin.values()                            # KB is fisheye-only
    fish = random_model_assignment([0, 1, 2], kind="fisheye", seed=1)
    assert set(fish.values()) <= set(FISHEYE_MODELS)
    assert "radtan" not in fish.values()                       # RadTan can't do fisheye


def _radtan_rig():
    def fac(cam_id, rng):
        fx = 800.0 * rng.uniform(0.98, 1.02)
        return RadTanModel(fx, fx, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
    obj, obs, img, gt, gtm = make_rig(n_cam=3, n_frame=45, noise_px=0.3, seed=0,
                                      w=W, h=H, model_factory=fac)
    return obj, obs, img


def test_per_camera_models_and_mccalib_writer_roundtrip(tmp_path):
    obj, obs, img = _radtan_rig()
    spec = {0: "radtan", 1: "double_sphere", 2: "ucm"}         # different model per camera
    rig = calibrate_rig(obj, obs, img, fix_intrinsics=False,
                        front_end=make_bundle_front_end(spec))
    assert [rig.cameras[c].name for c in (0, 1, 2)] == ["radtan", "ds", "ucm"]

    cam_path = tmp_path / "calibrated_cameras_data.yml"
    save_mccalib_cameras(rig, str(cam_path))
    save_mccalib_objects(obj, str(tmp_path / "calibrated_objects_data.yml"))
    save_mccalib_object_poses(rig, str(tmp_path / "calibrated_objects_pose_data.yml"))

    # exact-schema spot checks via OpenCV FileStorage (the reader MC-Calib itself uses)
    fs = cv2.FileStorage(str(cam_path), cv2.FILE_STORAGE_READ)
    assert int(fs.getNode("nb_camera").real()) == 3
    assert fs.getNode("camera_1").getNode("camera_model").string() == "double_sphere"
    fs.release()

    cams, _ = _load_cameras(str(cam_path))
    for c in (0, 1, 2):
        # camera_matrix matches the calibrated K
        assert np.allclose(cams[c].K, rig.cameras[c].K, atol=1e-6)
        # camera_pose_matrix is camera->world = inv(T_c_g)
        assert np.allclose(cams[c].pose, np.linalg.inv(rig.T_c_g[c]), atol=1e-6)
    # distortion_vector length is model-specific (radtan 5, ds 2, ucm 1)
    assert len(cams[0].dist) == 5 and len(cams[1].dist) == 2 and len(cams[2].dist) == 1


MODELS = [RadTanModel.sample, DoubleSphereModel.sample, UCMModel.sample, EUCMModel.sample,
          KannalaBrandtModel.sample, DSPlusModel.sample, EUCMPlusModel.sample, OCamModel.sample]


def _single_camera_rig(model):
    return RigState(cameras={0: model}, T_c_g={0: np.eye(4)}, ref_cam_id=0,
                    object_poses={}, objects={}, img_size={0: (640, 480)})


@pytest.mark.parametrize("factory", MODELS, ids=lambda f: f().name)
def test_load_camera_roundtrip(factory, tmp_path):
    """ds_msp.io.mccalib.load_camera is the read-back analogue of load_kalibr -- given
    calibrated_cameras_data.yml, it must hand back a ready CameraModel, not a raw K +
    distortion array, for every model MC-Calib format can name."""
    m = factory()
    path = tmp_path / "cams.yml"
    save_mccalib_cameras(_single_camera_rig(m), str(path))
    m2 = load_camera(str(path), 0)
    assert type(m2) is type(m)
    assert np.allclose(m2.params, m.params, atol=1e-6)


def test_load_camera_unknown_camera_id_raises(tmp_path):
    path = tmp_path / "cams.yml"
    save_mccalib_cameras(_single_camera_rig(RadTanModel.sample()), str(path))
    with pytest.raises(KeyError, match="camera_5"):
        load_camera(str(path), 5)


def test_rig_reexports_load_camera():
    """ds_msp.rig.load_camera is the discoverable entry point a ds-msp-calibrate-rig user
    reaches for -- must be exactly this module's load_camera, not a second implementation."""
    import ds_msp.rig as rig
    assert rig.load_camera is load_camera


def test_load_camera_no_stated_model_raises(tmp_path):
    """A plain MC-Calib file with no camera_model string and no recognizable
    distortion_type -- load_camera must say so clearly rather than guess."""
    path = tmp_path / "cams.yml"
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("nb_camera", 1)
    fs.startWriteStruct("camera_0", cv2.FileNode_MAP)
    fs.write("camera_matrix", np.eye(3))
    fs.write("distortion_vector", np.zeros((1, 5)))
    fs.endWriteStruct()
    fs.release()
    with pytest.raises(ValueError, match="cannot tell which model"):
        load_camera(str(path), 0)


# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-IO-004")
