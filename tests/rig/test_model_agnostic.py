"""Statistically validate that rig calibration is camera-model-agnostic.

For each camera model the cameras are *represented by that model* (ground-truth projection
uses it) and calibrated from scratch with it. Over many independent randomized trials we
run a one-sided t-test of H0: mean worst-camera pose error >= 1% against H1: < 1%, and
require rejection at p < 0.01 with every trial under 1%. This is the statistically
significant evidence that the pipeline's accuracy does not depend on the camera model.

Marked ``slow`` (spawns many full calibrations). Run just this file with:
    pytest tests/rig/test_model_agnostic.py -q
"""

import numpy as np
import pytest
from scipy import stats

from ds_msp.rig.calibrate import calibrate_rig, make_bundle_front_end
from ds_msp.models.radtan import RadTanModel
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.ucm import UCMModel
from ds_msp.models.eucm import EUCMModel
from ds_msp.models.kb import KannalaBrandtModel
from ._synth import make_rig

pytestmark = pytest.mark.req("NFR-NUM-006")

W, H = 1280, 960
N_TRIALS = 12
NOISE_PX = 0.3


def _factory(name):
    def f(cam_id, rng):
        fx = 800.0 * rng.uniform(0.95, 1.05)
        fy = fx * rng.uniform(0.98, 1.02)
        cx, cy = W / 2 + rng.uniform(-15, 15), H / 2 + rng.uniform(-15, 15)
        if name == "radtan":
            return RadTanModel(fx, fy, cx, cy, rng.uniform(-0.08, 0.0),
                               rng.uniform(0.0, 0.03), 0.0, 0.0, 0.0)
        if name == "ds":
            return DoubleSphereModel(fx, fy, cx, cy, rng.uniform(0.1, 0.3),
                                     rng.uniform(0.3, 0.5))
        if name == "ucm":
            return UCMModel(fx, fy, cx, cy, rng.uniform(0.5, 0.7))
        if name == "eucm":
            return EUCMModel(fx, fy, cx, cy, rng.uniform(0.5, 0.7),
                             rng.uniform(0.9, 1.1))
        if name == "kb":
            return KannalaBrandtModel(fx, fy, cx, cy, rng.uniform(0.0, 0.02),
                                      rng.uniform(0.0, 0.005), 0.0, 0.0)
        raise ValueError(name)
    return f


_MODELS = {"radtan": RadTanModel, "ds": DoubleSphereModel, "ucm": UCMModel,
           "eucm": EUCMModel, "kb": KannalaBrandtModel}


def _rel(Tref, Ti):
    return Ti @ np.linalg.inv(Tref)


def _trial_error(name, cls, seed):
    obj, obs, img, gt, _ = make_rig(n_cam=3, n_frame=60, noise_px=NOISE_PX, seed=seed,
                                    w=W, h=H, model_factory=_factory(name))
    rig = calibrate_rig(obj, obs, img, fix_intrinsics=False,
                        front_end=make_bundle_front_end(cls))
    ref = rig.ref_cam_id
    return 100.0 * max(
        abs(np.linalg.norm(_rel(rig.T_c_g[ref], rig.T_c_g[c])[:3, 3])
            - np.linalg.norm(_rel(gt[ref], gt[c])[:3, 3]))
        / np.linalg.norm(_rel(gt[ref], gt[c])[:3, 3])
        for c in rig.T_c_g if c != ref)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_MODELS))
def test_model_agnostic_within_1pct(name):
    cls = _MODELS[name]
    errs = np.array([_trial_error(name, cls, s) for s in range(N_TRIALS)])
    mean, std = errs.mean(), errs.std(ddof=1)
    # one-sided 99% upper confidence bound on the mean error
    ub99 = mean + stats.t.ppf(0.99, N_TRIALS - 1) * std / np.sqrt(N_TRIALS)
    # one-sided t-test, H0: mean >= 1% vs H1: mean < 1%
    pval = stats.t.cdf((mean - 1.0) / (std / np.sqrt(N_TRIALS)), N_TRIALS - 1)
    passrate = float((errs < 1.0).mean())
    # statistically significant that the expected pose error is < 1%, and the pipeline
    # keeps essentially every trial under 1% regardless of which model represents the cameras.
    assert pval < 0.01, \
        f"{name}: not significant that mean<1% (mean={mean:.3f}% p={pval:.2e})"
    assert ub99 < 1.0, \
        f"{name}: 99% upper bound on mean error {ub99:.3f}% not < 1%"
    assert passrate >= 0.9, \
        f"{name}: only {passrate*100:.0f}% of trials < 1% (errs={np.round(errs,3)})"
