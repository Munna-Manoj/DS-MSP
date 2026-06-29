"""Robust weighting (no rejection) handles outliers better than L2.

Two checks:
1. **BA isolation** — from a clean initialization, the redescending-Cauchy + MAD-auto-scale
   + GNC bundle adjustment recovers extrinsics far better than a plain L2 BA when 12% of
   corners are gross blunders, while keeping every corner (IRLS down-weighting, not
   rejection). Median reprojection over inliers stays sub-pixel.
2. **Full pipeline** — end to end at a realistic outlier rate, the rig stays accurate.
"""

import copy

import numpy as np

from ds_msp.rig import bundle, calibrate_rig
from ds_msp.rig.calibrate import make_bundle_front_end
from ds_msp.models.radtan import RadTanModel
from ._synth import make_rig


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


def test_robust_ba_beats_l2_under_outliers():
    obj, obs, img, gt, _ = make_rig(n_cam=3, n_frame=50, noise_px=0.3, seed=0)
    rig0 = calibrate_rig(obj, obs, img, fix_intrinsics=False)   # clean initialization
    dirty = _inject(obs, frac=0.12, px=50.0, seed=1)

    def two_pass(kernel_joint, **kw):
        r = bundle.refine(rig0, dirty, fix_intrinsics=True, robust_kernel="none"
                      if kernel_joint == "none" else "huber",
                      robust_scale=1.0 if kernel_joint == "none" else "auto")
        return bundle.refine(r, dirty, fix_intrinsics=False, robust_kernel=kernel_joint, **kw)

    rl = two_pass("none", robust_scale=1.0)
    rr = two_pass("cauchy", robust_scale="auto", gnc_iters=5, gnc_start=4.0)
    e_l2, e_rob = _worst_err(rl, gt), _worst_err(rr, gt)
    assert e_rob < e_l2, f"robust ({e_rob:.2f}%) should beat L2 ({e_l2:.2f}%)"
    assert e_rob < 1.0, f"robust BA should stay <1% under 12% outliers, got {e_rob:.2f}%"
    m = bundle.reprojection_metrics(rr, dirty)
    assert all(v["median"] < 1.0 for v in m.values())         # inlier median sub-pixel
    assert all(v["inlier_frac"] > 0.8 for v in m.values())    # ~all genuine corners kept


def test_full_pipeline_robust_under_outliers():
    # 6% gross (50 px) blunders, every point kept: robust pinhole + RANSAC-inlier seed +
    # IRLS bundle adjustment hold the full from-scratch pipeline accurate.
    obj, obs, img, gt, _ = make_rig(n_cam=3, n_frame=50, noise_px=0.3,
                                    outlier_frac=0.06, outlier_px=50.0, seed=2)
    rig = calibrate_rig(obj, obs, img, fix_intrinsics=False,
                        front_end=make_bundle_front_end(RadTanModel))
    assert _worst_err(rig, gt) < 2.0
