"""Rig-level numerical guards: pose refinement and target-structure refinement never return
non-finite estimates, hand-eye linking does not depend on OpenCV's global RNG, and an
indefinite pair covariance is reported as unidentified instead of clamped to a tight sigma."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import ds_msp.rig.pose_init as pose_init
from ds_msp.core.lie import so3_exp
from ds_msp.data.observations import RigState
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig import bundle
from ds_msp.rig.handeye import handeye_bootstrap
from ds_msp.rig.types import ObjectObs

from ._synth import make_object

pytestmark = pytest.mark.req("NFR-NUM-010")


def _T(rvec, t):
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.asarray(rvec, float))
    T[:3, 3] = t
    return T


def _view():
    cam = RadTanModel(800.0, 800.0, 640.0, 480.0, -0.05, 0.01, 0.0, 0.0, 0.0)
    g = np.stack(np.meshgrid(np.arange(5), np.arange(4)), -1).reshape(-1, 2) * 0.05
    X = np.c_[g, np.zeros(len(g))]
    T = _T([0.1, -0.2, 0.05], [-0.1, -0.05, 1.0])
    uv, ok = cam.project((T[:3, :3] @ X.T).T + T[:3, 3])
    assert ok.all()
    return cam, X, uv, T


def test_pose_irls_never_returns_a_non_finite_pose(monkeypatch):
    cam, X, uv, T = _view()
    T0 = _T([0.12, -0.18, 0.04], [-0.09, -0.06, 1.02])
    real = pose_init.bearing_pose_residual_jacobian

    def poisoned(Tc, *a, **k):                      # finite at T0, NaN everywhere else
        e, J, extra = real(Tc, *a, **k)
        if not np.array_equal(Tc, T0):
            e = np.full_like(e, np.nan)
            J = np.full_like(J, np.nan)
        return e, J, extra

    monkeypatch.setattr(pose_init, "bearing_pose_residual_jacobian", poisoned)
    out = pose_init.robust_pose_irls(cam, X, uv, T0=T0)
    np.testing.assert_array_equal(out, T0)          # the old guard returned the NaN pose


def test_pose_irls_still_refines_a_clean_view():
    cam, X, uv, T = _view()
    T0 = _T([0.12, -0.18, 0.04], [-0.09, -0.06, 1.02])
    out = pose_init.robust_pose_irls(cam, X, uv, T0=T0)
    np.testing.assert_allclose(out, T, atol=1e-6)


class _NaNForSomePoints:
    """Wraps a model so a few points project to NaN and are flagged invalid: the
    'multiply a NaN by a zero weight' trap."""

    def __init__(self, model, bad_rows):
        self.model, self.bad = model, bad_rows

    def __getattr__(self, name):
        return getattr(self.model, name)

    def project_jacobian(self, P):
        uv, Jp, Jk, valid = self.model.project_jacobian(P)
        uv, Jp, valid = uv.copy(), Jp.copy(), np.asarray(valid, bool).copy()
        uv[self.bad] = np.nan
        Jp[self.bad] = np.nan
        valid[self.bad] = False
        return uv, Jp, Jk, valid


def test_structure_refinement_never_writes_nan_into_the_target():
    cam = RadTanModel(800.0, 800.0, 640.0, 480.0, -0.05, 0.01, 0.0, 0.0, 0.0)
    obj = make_object({0: np.eye(4)})
    poses, obs = {}, []
    rng = np.random.default_rng(0)
    for fr in range(8):
        Tgo = _T(rng.normal(scale=0.2, size=3), [rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 1.5])
        poses[(0, fr)] = Tgo
        uv, ok = cam.project((Tgo[:3, :3] @ obj.pts_3d.T).T + Tgo[:3, 3])
        rows = np.where(ok)[0]
        obs.append(ObjectObs(cam_id=0, frame_id=fr, object_id=0, point_rows=rows,
                             pts_2d=uv[rows] + rng.normal(scale=0.3, size=(len(rows), 2))))
    rig = RigState(cameras={0: _NaNForSomePoints(cam, bad_rows=[5, 6])}, T_c_g={0: np.eye(4)},
                   ref_cam_id=0, object_poses=poses, objects={0: obj})
    out = bundle.refine_object_structure(rig, obs)
    pts = out.objects[0].pts_3d
    assert np.isfinite(pts).all()
    assert not np.array_equal(pts, obj.pts_3d)       # the clean points still moved


def test_handeye_bootstrap_ignores_opencv_global_rng_state():
    T_b = _T([0.05, 0.9, -0.1], [0.7, -0.2, 0.15])
    rng = np.random.default_rng(1)
    traj = []
    for _ in range(30):
        axis = rng.normal(size=3)
        traj.append(_T(axis / np.linalg.norm(axis) * rng.uniform(0.2, 1.0),
                       [rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(1.5, 2.5)]))
    poses_a, poses_b = traj, [T_b @ Tw for Tw in traj]
    out = []
    for cv_seed in (0, 12345, 999):
        cv2.setRNGSeed(cv_seed)
        out.append(handeye_bootstrap(poses_a, poses_b, seed=0))
    for T in out[1:]:
        np.testing.assert_array_equal(T, out[0])


def _unc(cov):
    rig = SimpleNamespace(T_c_g={0: np.eye(4), 1: _T([0.0, 0.3, 0.0], [0.2, 0.0, 0.0])})
    return rig, {"cov": cov, "cam_col": {1: 0}}


def test_pair_uncertainty_reports_an_indefinite_covariance_as_unidentified():
    good = np.diag([1e-6, 2e-6, 3e-6, 1e-4, 1e-4, 1e-4])
    out = bundle.pair_uncertainty(*_unc(good))[(0, 1)]
    assert out["identifiable"] and np.isfinite(out["rotation_sigma_deg"])

    bad = good.copy()
    bad[0, 0] = -1e-6                                  # a clearly negative variance
    out = bundle.pair_uncertainty(*_unc(bad))[(0, 1)]
    assert not out["identifiable"]
    assert out["rotation_sigma_deg"] == np.inf         # never clamped to a tight sigma
