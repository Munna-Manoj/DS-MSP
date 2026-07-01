"""GNC-TLS high-breakdown robust BA wired into the rig and single-camera calibration paths.

The default robust BA (Cauchy/Huber + MAD auto-scale) is capped at MAD's 50% breakdown. Passing
``noise_bound`` switches both bundle adjusters to the median-free GNC-TLS solve, which recovers
under MAJORITY-outlier corner contamination where the kernel path is dragged off. These are the
integration tests for the two wired call sites (FR-CORE-001).
"""

import copy

import numpy as np
import pytest

from ds_msp.rig import bundle, calibrate_rig
from ds_msp.models.radtan import RadTanModel
from ._synth import make_rig

pytestmark = pytest.mark.req("FR-CORE-001")


def _rel(Tref, Ti):
    return Ti @ np.linalg.inv(Tref)


def _worst_err(rig, gt):
    ref = rig.ref_cam_id
    return 100.0 * max(
        abs(np.linalg.norm(_rel(rig.T_c_g[ref], rig.T_c_g[c])[:3, 3])
            - np.linalg.norm(_rel(gt[ref], gt[c])[:3, 3]))
        / np.linalg.norm(_rel(gt[ref], gt[c])[:3, 3])
        for c in rig.T_c_g if c != ref)


def _inject(obs, frac, px, seed):
    rng = np.random.default_rng(seed)
    dirty = []
    for o in obs:
        o2 = copy.copy(o)
        p = o.pts_2d.copy()
        bad = rng.random(len(p)) < frac
        p[bad] += rng.uniform(-px, px, size=(int(bad.sum()), 2))
        o2.pts_2d = p
        dirty.append(o2)
    return dirty


def test_rig_ba_gnc_tls_recovers_past_mad_breakdown():
    """At 60% gross corner blunders the MAD-auto-scale kernel BA is dragged off; the GNC-TLS
    path (noise_bound set) recovers the rig extrinsics to <1% and beats it."""
    obj, obs, img, gt, _ = make_rig(n_cam=3, n_frame=30, noise_px=0.3, seed=0)
    rig0 = calibrate_rig(obj, obs, img, fix_intrinsics=False)        # clean initialization
    dirty = _inject(obs, frac=0.60, px=50.0, seed=1)

    # default robust kernel path (MAD auto-scale + GNC) — expected to break past 50%
    kern = bundle.refine(rig0, dirty, fix_intrinsics=False,
                         robust_kernel="cauchy", robust_scale="auto",
                         gnc_iters=5, gnc_start=4.0)
    # GNC-TLS path: explicit noise bound (inlier σ ≈ 0.3 px)
    gtls = bundle.refine(rig0, dirty, fix_intrinsics=False, noise_bound=0.3)

    e_kern, e_gtls = _worst_err(kern, gt), _worst_err(gtls, gt)
    assert e_gtls < 1.0, f"GNC-TLS rig BA should stay <1% at 60% outliers, got {e_gtls:.2f}%"
    assert e_gtls < e_kern, f"GNC-TLS ({e_gtls:.2f}%) should beat the MAD kernel ({e_kern:.2f}%)"


def test_rig_ba_gnc_tls_clean_no_harm():
    """With no outliers the GNC-TLS path must not hurt — it stays accurate on clean data."""
    obj, obs, img, gt, _ = make_rig(n_cam=3, n_frame=20, noise_px=0.3, seed=4)
    rig0 = calibrate_rig(obj, obs, img, fix_intrinsics=False)
    out = bundle.refine(rig0, obs, fix_intrinsics=False, noise_bound=0.3)
    assert _worst_err(out, gt) < 1.0


def _synth_single_cam(n_img=14, outlier_frac=0.0, outlier_px=40.0, noise_px=0.3, seed=0):
    """Self-contained single-camera calibration problem built only from bundle_adjust's
    documented contract: a known RadTan model viewing a planar board over n_img poses with
    diverse depth + rotation (so the focal length is observable)."""
    from ds_msp.core.lie import so3_exp
    rng = np.random.default_rng(seed)
    cls = RadTanModel
    f, w, h = 600.0, 1280.0, 960.0
    model_true = cls(f, f, w / 2, h / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
    p_true = np.asarray(model_true.params, float)
    gx, gy = np.meshgrid(np.linspace(-0.25, 0.25, 6), np.linspace(-0.25, 0.25, 6))
    Xw = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]

    X_list, uv_list, vis_list, R0, t0 = [], [], [], [], []
    for _ in range(n_img):
        w_ax = rng.uniform(-0.6, 0.6, 3)
        R = so3_exp(w_ax)
        t = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), rng.uniform(0.9, 2.4)])
        Xc = (R @ Xw.T).T + t
        uv, valid = model_true.project(Xc)
        uv = uv + noise_px * rng.normal(size=uv.shape)
        n_out = int(round(outlier_frac * len(uv)))
        if n_out:
            idx = rng.choice(len(uv), n_out, replace=False)
            uv[idx] += rng.uniform(-outlier_px, outlier_px, size=(n_out, 2))
        X_list.append(Xw)
        uv_list.append(uv)
        vis_list.append(valid.copy())
        # seed poses: ground truth perturbed slightly (BA is local — mimics a good seed)
        R0.append(so3_exp(w_ax + 0.01 * rng.normal(size=3)))
        t0.append(t + 0.01 * rng.normal(size=3))
    return cls, p_true, X_list, uv_list, vis_list, np.array(R0), np.array(t0)


def test_calib_bundle_adjust_gnc_tls_recovers_past_50pct():
    """Single-camera calibration BA: at 60% corner blunders the default robust path (Cauchy +
    MAD auto-scale + GNC) is dragged off — MAD breaks at 50% — while the GNC-TLS path
    (noise_bound set) recovers the intrinsics."""
    from ds_msp.geometry.calibrate_core import bundle_adjust

    cls, p_true, X, uv, vis, R0, t0 = _synth_single_cam(
        n_img=14, outlier_frac=0.60, outlier_px=40.0, noise_px=0.3, seed=3)
    seed0 = np.asarray(cls(600, 600, 640, 480).params)      # deliberately off intrinsics seed

    p_kern, *_ = bundle_adjust(cls, seed0.copy(), R0, t0, X, uv, vis,
                               kernel="cauchy", scale="auto", gnc_start=4.0, gnc_iters=5,
                               max_iter=40)
    p_gtls, *_ = bundle_adjust(cls, seed0.copy(), R0, t0, X, uv, vis,
                               noise_bound=0.3, max_iter=40)

    # focal + principal point recovery (the well-constrained intrinsics)
    err_kern = np.linalg.norm(p_kern[:4] - p_true[:4])
    err_gtls = np.linalg.norm(p_gtls[:4] - p_true[:4])
    assert err_gtls < 15.0, f"GNC-TLS calib should recover f,c (<2.5% of f=600), got {err_gtls:.1f}px"
    assert err_gtls < 0.2 * err_kern, (
        f"GNC-TLS ({err_gtls:.1f}px) should crush the MAD-capped kernel ({err_kern:.1f}px) at 60%")


def test_calib_bundle_adjust_signature_accepts_noise_bound():
    """Lightweight contract test: bundle_adjust exposes noise_bound and dispatches to GNC-TLS
    (the heavy numeric recovery is covered by tests/core/test_gnc_tls.py + the rig BA test)."""
    import inspect

    from ds_msp.geometry.calibrate_core import bundle_adjust
    sig = inspect.signature(bundle_adjust)
    assert "noise_bound" in sig.parameters
    assert sig.parameters["noise_bound"].default is None
