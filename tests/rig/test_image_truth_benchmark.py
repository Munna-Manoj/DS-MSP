"""Image-level ground-truth benchmark: rendered ChArUco images -> full config pipeline.

A converging stereo rig observes a two-board target whose boards meet at 35 degrees. Images
are ray-cast through the true camera models (``_render``), so the detector's pixel
convention, target reconstruction, multi-board refinement and the joint bundle adjustment are
all scored against known extrinsics and inter-board pose -- in degrees and millimetres, not
training reprojection. The limits are CI regression limits measured on this scene:

* ``pinhole`` (two pinhole+distortion cameras, fast lane): rigid refinement reaches
  ~0.004 deg / 0.16 mm and must beat a frozen inter-board pose. (A half-pixel corner
  convention error mostly shifts the principal point -- here only 0.004 -> 0.009 deg -- so it
  is guarded at the detector instead: ``tests/calib/test_charuco_compat.py``.)
* ``fisheye`` (pinhole + double-sphere, ``slow``): ~0.09 deg / 1.4 mm. The floor is the
  detector, not the solver: ChArUco interpolates corners through a marker homography, which
  strong distortion bends (0.32 px RMS on the fisheye vs 0.12 px pinhole). Tighten these
  limits when native-model corner refinement lands.
"""
import cv2
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.detect.charuco import BoardSpec
from ds_msp.io.mccalib import save_mccalib_cameras
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig.calib_param import calibrate_from_config
from ds_msp.rig.types import RigState

from . import _render

pytestmark = pytest.mark.req("FR-RIG-026", "FR-CALIB-004")

SIZE = (800, 600)
SQUARE_MM = 50.0
SPECS = [BoardSpec(7, 5, 0.05, 0.037, SQUARE_MM), BoardSpec(7, 5, 0.05, 0.037, SQUARE_MM)]


def _T(rotvec, t):
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.asarray(rotvec, float))
    T[:3, 3] = t
    return T


RIGS = {
    "pinhole": ({0: RadTanModel(620.0, 618.0, 401.3, 297.8, -0.12, 0.03, 0.0005, -0.0003, 0.0),
                 1: RadTanModel(600.0, 599.0, 398.6, 302.1, -0.08, 0.015, 0.0, 0.0, 0.0)},
                ["radtan", "radtan"]),
    "fisheye": ({0: RadTanModel(620.0, 618.0, 401.3, 297.8, -0.12, 0.03, 0.0005, -0.0003, 0.0),
                 1: DoubleSphereModel(390.0, 389.0, 398.6, 302.1, -0.12, 0.6)},
                ["radtan", "ds"]),
}
# (pair rotation deg, pair camera-centre mm, inter-board rotation deg, inter-board mm)
LIMITS = {"pinhole": (0.015, 0.4, 0.1, 0.1), "fisheye": (0.15, 2.5, 0.15, 0.1)}
T_C_G = {0: np.eye(4), 1: _T([0.0, -0.25, 0.01], [-300.0, 5.0, 40.0])}
# board 1 hinged on board 0's right edge, turned 35 degrees towards the cameras
T_O_B = {0: np.eye(4), 1: _T([0.0, -np.radians(35), 0.0], [7 * SQUARE_MM + 20.0, 0.0, -10.0])}


def render_dataset(root, kind):
    """Render 18 synchronized frames of the target into ``root/Cam_00k`` and write a config
    (true intrinsics as ``cam_params_path``, held fixed). Returns the config path."""
    models, names = RIGS[kind]
    tex = _render.board_textures(SPECS)
    rng = np.random.default_rng(7)
    for fr in range(18):
        T_g_o = _T(rng.normal(0, 0.2, 3), [rng.uniform(-380, -180), rng.uniform(-150, 30),
                                           rng.uniform(900, 1200)])
        for c, model in models.items():
            T_c_b = {b: T_C_G[c] @ T_g_o @ T for b, T in T_O_B.items()}
            d = root / f"Cam_{c + 1:03d}"
            d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / f"{fr:06d}.png"),
                        _render.render(model, SIZE, T_c_b, SPECS, tex, noise=1.0, rng=rng))
    rig = RigState(cameras=models, T_c_g=T_C_G, ref_cam_id=0, object_poses={}, objects={},
                   img_size={0: SIZE, 1: SIZE})
    save_mccalib_cameras(rig, str(root / "init_cameras.yml"))
    cfg = root / "calib_param.yml"
    cfg.write_text(
        "%YAML:1.0\n---\n"
        "number_x_square: 7\nnumber_y_square: 5\nlength_square: 0.05\nlength_marker: 0.037\n"
        f"number_board: 2\nsquare_size: {SQUARE_MM}\nnumber_camera: 2\n"
        f"camera_models: [\"{names[0]}\", \"{names[1]}\"]\n"
        f"cam_params_path: \"{root / 'init_cameras.yml'}\"\nfix_intrinsic: 1\n"
        f"root_path: \"{root}\"\ncam_prefix: \"Cam_\"\nkeypoints_path: \"None\"\n"
        "save_path: \"None\"\nhe_approach: 0\nsave_detection: 0\n"
        "save_reprojection: 0\nverbose: 0\nwebviewer: 0\n")
    return cfg


@pytest.fixture(scope="module", params=[
    "pinhole", pytest.param("fisheye", marks=pytest.mark.slow)])
def scene(request, tmp_path_factory):
    return request.param, render_dataset(tmp_path_factory.mktemp(request.param), request.param)


def _run(cfg, **overrides):
    return calibrate_from_config(str(cfg), overrides=overrides)


def _angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def pair_error(rig):
    """Relative cam1<-cam0 error vs truth: rotation (deg) and camera-centre (mm)."""
    est = rig.T_c_g[1] @ np.linalg.inv(rig.T_c_g[0])
    gt = T_C_G[1] @ np.linalg.inv(T_C_G[0])

    def centre(T):
        return -T[:3, :3].T @ T[:3, 3]
    return _angle(est[:3, :3].T @ gt[:3, :3]), float(np.linalg.norm(centre(est) - centre(gt)))


def board_error(rig):
    """Inter-board pose error (deg, mm) of the solved target vs the true hinge."""
    obj = rig.objects[0]
    est = np.linalg.inv(obj.T_co_b[0]) @ obj.T_co_b[1]
    gt = np.linalg.inv(T_O_B[0]) @ T_O_B[1]
    return _angle(est[:3, :3].T @ gt[:3, :3]), float(np.linalg.norm(est[:3, 3] - gt[:3, 3]))


def test_extrinsics_and_target_match_truth(scene):
    kind, cfg = scene
    res = _run(cfg)
    rot, centre = pair_error(res["rig"])
    b_rot, b_t = board_error(res["rig"])
    lim = LIMITS[kind]
    assert rot < lim[0] and centre < lim[1], (kind, rot, centre)
    assert b_rot < lim[2] and b_t < lim[3], (kind, b_rot, b_t)
    assert res["acceptance"]["structure_mode"] == "rigid"


#: Inter-board translation both modes reach at the detector noise floor, 0.1% of a 50 mm
#: square. Measured rigid / frozen: pinhole 0.031 / 0.073 mm (OpenCV 4.12), 0.044 / 0.025 mm
#: (5.0); fisheye 0.015 / 0.214 mm and 0.041 / 0.189 mm. A strict "rigid < frozen" on
#: translation is a coin flip on detector noise where the averaged PnP poses are already good.
BOARD_NOISE_MM = 0.05


def test_rigid_target_refinement_beats_frozen_interboard_pose(scene):
    """Rigid refinement improves the inter-board rotation, never loses translation beyond the
    noise floor, and does not degrade the camera pair. Board rotation, rigid / frozen:
    pinhole 0.045 / 0.066 deg (OpenCV 4.12), 0.005 / 0.041 deg (5.0); fisheye 0.088 / 0.116
    deg and 0.010 / 0.066 deg."""
    kind, cfg = scene
    rigid = _run(cfg)["rig"]
    off = _run(cfg, structure_mode="off")["rig"]
    b_rot, b_t = board_error(rigid)
    assert b_rot < LIMITS[kind][2] and b_t < LIMITS[kind][3], (kind, b_rot, b_t)
    assert b_rot < board_error(off)[0]
    assert b_t <= board_error(off)[1] + BOARD_NOISE_MM
    assert pair_error(rigid)[0] <= pair_error(off)[0] + 0.005
