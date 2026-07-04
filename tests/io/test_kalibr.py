"""Kalibr YAML I/O: round-trip per model + known-format field-order checks."""

import numpy as np
import pytest

from ds_msp.io.kalibr import to_kalibr_cam, from_kalibr_cam, save_kalibr, load_kalibr
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.dsplus import DSPlusModel
from ds_msp.models.eucm import EUCMModel
from ds_msp.models.eucmplus import EUCMPlusModel
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.radtan import RadTanModel
from ds_msp.models.ucm import UCMModel

MODELS = [DoubleSphereModel.sample, EUCMModel.sample, KannalaBrandtModel.sample,
          RadTanModel.sample, UCMModel.sample, DSPlusModel.sample, EUCMPlusModel.sample]


@pytest.mark.parametrize("factory", MODELS, ids=lambda f: f().name)
def test_kalibr_roundtrip(factory, tmp_path):
    m = factory()
    if m.name == "radtan":
        m = RadTanModel(600, 602, 320, 240, -0.12, 0.05, 0.001, -0.0015, 0.0)  # k3=0 (Kalibr)
    path = tmp_path / "camchain.yaml"
    save_kalibr(m, str(path), 1920, 1080)
    m2 = load_kalibr(str(path))
    assert type(m2) is type(m)
    assert np.allclose(m2.params, m.params, atol=1e-9)


def test_ds_intrinsics_order_is_xi_alpha_first():
    ds = DoubleSphereModel(700, 701, 640, 360, xi=-0.18, alpha=0.59)
    block = to_kalibr_cam(ds, 1024, 1024)
    assert block["camera_model"] == "ds"
    assert block["intrinsics"][:2] == [-0.18, 0.59]          # xi, alpha first
    assert block["intrinsics"][2:] == [700, 701, 640, 360]


def test_kb_is_pinhole_equidistant():
    kb = KannalaBrandtModel.sample()
    block = to_kalibr_cam(kb, 752, 480)
    assert block["camera_model"] == "pinhole"
    assert block["distortion_model"] == "equidistant"
    assert len(block["distortion_coeffs"]) == 4


def test_parse_real_euroc_pinhole_equidistant(tmp_path):
    yaml_text = """
cam0:
  camera_model: pinhole
  intrinsics: [461.629, 460.152, 362.680, 246.049]
  distortion_model: equidistant
  distortion_coeffs: [0.0034, -0.0006, 0.0004, -0.0001]
  resolution: [752, 480]
  rostopic: /cam0/image_raw
"""
    p = tmp_path / "euroc.yaml"
    p.write_text(yaml_text)
    m = load_kalibr(str(p))
    assert isinstance(m, KannalaBrandtModel)
    assert np.allclose(m.K[0, 0], 461.629)
    assert np.allclose(m.distortion, [0.0034, -0.0006, 0.0004, -0.0001])


def test_radtan_k3_dropped_with_warning(tmp_path):
    m = RadTanModel(600, 600, 320, 240, 0.1, 0.05, 0.001, 0.001, k3=0.02)
    with pytest.warns(UserWarning):
        to_kalibr_cam(m, 640, 480)


def test_ucm_omni_xi_mapping_roundtrips(tmp_path):
    ucm = UCMModel(700, 700, 640, 360, alpha=0.62)
    block = to_kalibr_cam(ucm, 1280, 720)
    assert block["camera_model"] == "omni"
    back = from_kalibr_cam(block)
    assert np.isclose(back.alpha, 0.62)


def test_dsplus_is_a_ds_msp_extension_with_no_kalibr_native_equivalent():
    """DS+/EUCM+ have no camera_model in a real Kalibr installation -- this is a DS-MSP
    extension (see the module docstring), analogous to how ds_msp.rig already extends
    MC-Calib's own format. Only asserts DS-MSP's own round-trip; a real Kalibr cannot read
    ds_plus/eucm_plus."""
    m = DSPlusModel(996.1, 995.9, 1276.6, 875.3, 0.7226, -0.2795, 0.1122, -0.0014, 0.0020)
    block = to_kalibr_cam(m, 2592, 1800)
    assert block["camera_model"] == "ds_plus"
    assert block["intrinsics"][0] == m.alpha           # alpha first, then fx,fy,cx,cy
    assert block["intrinsics"][1:] == [m.fx, m.fy, m.cx, m.cy]
    assert block["distortion_coeffs"] == [m.lambda1, m.lambda2, m.tau_x, m.tau_y]
    back = from_kalibr_cam(block)
    assert isinstance(back, DSPlusModel)
    assert np.allclose(back.params, m.params)


def test_calib_load_camera_reexports_load_kalibr():
    """``ds_msp.calib.load_camera`` is the discoverable, format-agnostic-sounding entry point
    a ds-msp-calibrate user reaches for -- must be exactly this module's load_kalibr, not a
    second, drifting implementation."""
    import ds_msp.calib as calib
    assert calib.load_camera is load_kalibr


def test_eucmplus_is_a_ds_msp_extension_with_no_kalibr_native_equivalent():
    m = EUCMPlusModel(700, 700, 640, 360, 0.6, 1.1, 0.05, 0.001, -0.002)
    block = to_kalibr_cam(m, 1280, 720)
    assert block["camera_model"] == "eucm_plus"
    assert block["intrinsics"][:2] == [m.alpha, m.beta]
    assert block["distortion_coeffs"] == [m.lambda1, m.tau_x, m.tau_y]
    back = from_kalibr_cam(block)
    assert isinstance(back, EUCMPlusModel)
    assert np.allclose(back.params, m.params)

# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-IO-001")
