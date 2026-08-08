"""Calibrate to within 1% of the *original* intrinsics AND pose with a model of choice.

The cameras are built with an *original* model (KB or RadTan); we calibrate the rig with a
possibly-*different* chosen model and check we recover, to 1%:

  * the inter-camera **pose** (vs ground truth), and
  * the **intrinsics** — compared apples-to-apples by converting the original camera into
    the chosen model (``adapt.convert``) and measuring the paraxial-focal error (the one
    intrinsic comparable across models). For pairs where the chosen model covers the
    original's field of view (KB fisheye -> sphere models) we additionally check the
    **functional** reprojection RMS over the whole image (the strict "same camera" test);
    that metric is meaningless when the chosen model cannot represent the original's FOV
    (a fisheye is not reproducible by RadTan at the periphery), so it is checked only there.

Under gross outliers this exercises the from-scratch robust front-end (RANSAC DLT intrinsic
resection + deterministic bearing GNC-TLS PnP — no ``cv2.calibrateCamera``), the inlier-gated
per-view PnP, and the robust SE(3) extrinsics-init consensus. Marked ``slow``::

    pytest tests/rig/test_param_pose.py -q
"""
import numpy as np
import pytest

from ds_msp.rig.calibrate import calibrate_rig, make_bundle_front_end, paraxial_focal
from ds_msp.adapt.convert import convert
from ds_msp.adapt.evaluate import reprojection_report
from ds_msp.models.radtan import RadTanModel
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.ucm import UCMModel
from ds_msp.models.eucm import EUCMModel
from ds_msp.models.kb import KannalaBrandtModel
from ._synth import make_rig

pytestmark = pytest.mark.req("NFR-NUM-006")

W, H = 1280, 960
N_SEEDS = 4
CHOSEN = {"radtan": RadTanModel, "ds": DoubleSphereModel, "ucm": UCMModel,
          "eucm": EUCMModel, "kb": KannalaBrandtModel}


def _factory(name):
    def f(cam_id, rng):
        fx = 800.0 * rng.uniform(0.97, 1.03)
        fy = fx * rng.uniform(0.99, 1.01)
        cx, cy = W / 2 + rng.uniform(-10, 10), H / 2 + rng.uniform(-10, 10)
        if name == "radtan":
            return RadTanModel(fx, fy, cx, cy, rng.uniform(-0.08, -0.02),
                               rng.uniform(0.0, 0.03), 0.0, 0.0, 0.0)
        if name == "kb":
            return KannalaBrandtModel(fx, fy, cx, cy, rng.uniform(0.0, 0.02),
                                      rng.uniform(0.0, 0.005), 0.0, 0.0)
        raise ValueError(name)
    return f


def _rel(Tref, Ti):
    return Ti @ np.linalg.inv(Tref)


def _trial(original, chosen, outlier, seed):
    """Return (pose%, focal%, func%) for one rig calibration."""
    cls = CHOSEN[chosen]
    obj, obs, img, gt, gtm = make_rig(n_cam=3, n_frame=60, noise_px=0.3, seed=seed,
                                      w=W, h=H, model_factory=_factory(original),
                                      outlier_frac=outlier, outlier_px=40.0)
    rig = calibrate_rig(obj, obs, img, fix_intrinsics=False,
                        front_end=make_bundle_front_end(cls))
    ref = rig.ref_cam_id
    pose = 100.0 * max(
        abs(np.linalg.norm(_rel(rig.T_c_g[ref], rig.T_c_g[c])[:3, 3])
            - np.linalg.norm(_rel(gt[ref], gt[c])[:3, 3]))
        / np.linalg.norm(_rel(gt[ref], gt[c])[:3, 3])
        for c in rig.T_c_g if c != ref)
    foc, func = [], []
    for c in rig.cameras:
        ref_model, _ = convert(gtm[c], cls, width=W, height=H, n_samples=400,
                               n_restarts=3, seed=1)
        f_ref = paraxial_focal(ref_model)[0]
        foc.append(100.0 * abs(paraxial_focal(rig.cameras[c])[0] - f_ref) / f_ref)
        rep = reprojection_report(ref_model, rig.cameras[c], W, H, n_samples=1200)
        func.append(100.0 * rep["rms_px"] / f_ref)
    return pose, max(foc), max(func)


def _worst(original, chosen, outlier):
    return np.array([_trial(original, chosen, outlier, s) for s in range(N_SEEDS)]).max(axis=0)


# 1. The headline robustness result: a KB (fisheye) rig calibrated with a *different* sphere
#    model recovers param+pose+function within 1% under 10% gross outliers — the regime the
#    old OpenCV-seeded front-end diverged in. func is meaningful here (sphere models cover the
#    fisheye FOV).
@pytest.mark.slow
@pytest.mark.parametrize("chosen", ["ds", "ucm", "eucm"])
def test_kb_original_robust_under_outliers(chosen):
    pose, foc, func = _worst("kb", chosen, 0.10)
    assert pose < 1.0, f"kb->{chosen}: worst pose {pose:.3f}%"
    assert foc < 1.0, f"kb->{chosen}: worst paraxial-focal {foc:.3f}%"
    assert func < 1.0, f"kb->{chosen}: worst functional RMS {func:.3f}% of focal"


# 2. Same-family robustness past the old ceiling, and the FOV-incompatible KB->RadTan case
#    (func excluded — RadTan cannot reproduce a fisheye periphery; param+pose still hold).
@pytest.mark.slow
@pytest.mark.parametrize("original,chosen,outlier", [
    ("radtan", "radtan", 0.15),
    ("kb", "radtan", 0.10),
])
def test_param_pose_robust(original, chosen, outlier):
    pose, foc, _func = _worst(original, chosen, outlier)
    assert pose < 1.0, f"{original}->{chosen}@{outlier:.0%}: worst pose {pose:.3f}%"
    assert foc < 1.0, f"{original}->{chosen}@{outlier:.0%}: worst paraxial-focal {foc:.3f}%"


# 3. Model-of-choice at the measurement floor (no outliers): every chosen model that can
#    represent the original recovers param+pose within 1% — proving the choice of calibration
#    model is accurate, not just robust.
@pytest.mark.slow
@pytest.mark.parametrize("original,chosen", [
    ("kb", "radtan"), ("kb", "ds"), ("kb", "ucm"), ("kb", "eucm"), ("kb", "kb"),
    ("radtan", "radtan"),
])
def test_model_of_choice_clean(original, chosen):
    pose, foc, _func = _worst(original, chosen, 0.0)
    assert pose < 1.0, f"{original}->{chosen}: worst pose {pose:.3f}%"
    assert foc < 1.0, f"{original}->{chosen}: worst paraxial-focal {foc:.3f}%"
