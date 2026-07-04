"""Front-end intrinsic pre-calibration should not re-solve an identical deterministic problem
twice. Verifies NFR-PERF-002: for a model whose ``initialize_from_correspondences`` is a
neutral-distortion no-op (RadTan, OCam), the model-aware seed and the plain from-``K`` seed are
byte-identical, so the "Two-start" step must skip the duplicate solve; a model with a genuine
closed-form distortion seed (KB, DS/UCM/EUCM sphere models) must keep both starts.
"""
from unittest.mock import patch

import numpy as np
import pytest

import ds_msp.rig.calibrate as rc
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig.calibrate import _model_aware_seed, _seed_from_K, make_bundle_front_end

from ._synth import make_rig

pytestmark = pytest.mark.req("NFR-PERF-002")

W, H = 640, 480


def _count_calibrate_single_calls(model_name, model_factory, n_frame=10, seed=0):
    obj, obs, img, gt_ext, gtm = make_rig(n_cam=1, n_frame=n_frame, noise_px=0.2, seed=seed,
                                          w=W, h=H, model_factory=model_factory)
    obs_by_cam = {0: obs}
    orig = rc._calibrate_single
    calls = {"n": 0}

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    with patch.object(rc, "_calibrate_single", side_effect=counting):
        fe = make_bundle_front_end(model_name)
        fe(obj, obs_by_cam, img)
    return calls["n"]


def test_radtan_skips_the_duplicate_two_start_solve():
    """RadTan's neutral-distortion seed and from-K seed are identical -> 2 calls/camera
    (1 robust-pinhole refine + 1 deduped Two-start), not the old 3."""
    def fac(cam_id, rng):
        fx = 700.0 * rng.uniform(0.98, 1.02)
        return RadTanModel(fx, fx, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
    assert _count_calibrate_single_calls("radtan", fac) == 2


def test_ds_keeps_both_two_start_seeds():
    """DS's model-aware (rays-derived xi/alpha) seed genuinely differs from the neutral
    from-K seed -> the full 3 calls/camera (basin-rescue) must be preserved."""
    def fac(cam_id, rng):
        return DoubleSphereModel(300, 300, W / 2, H / 2, 0.1, 0.6)
    assert _count_calibrate_single_calls("ds", fac) == 3


def test_kb_keeps_both_two_start_seeds():
    """KB's closed-form theta-polynomial LS seed genuinely differs from neutral -> full 3
    calls/camera preserved (the case the Two-start docstring calls out explicitly)."""
    def fac(cam_id, rng):
        return KannalaBrandtModel(300, 300, W / 2, H / 2, 0.0, 0.0, 0.0, 0.0)
    assert _count_calibrate_single_calls("kb", fac) == 3


def test_the_skipped_candidate_is_a_true_duplicate_not_an_approximation():
    """The dedup is only correct if the two Two-start seeds are genuinely byte-identical on
    real correspondence data (not just in the zero-ray edge case) — i.e. skipping the second
    solve cannot change the result, because it would have converged to what the first solve
    already found. RadTan/OCam: identical (skip is lossless). KB/DS: genuinely different
    (skip would be lossy, so the front-end must keep both)."""
    def fac(cam_id, rng):
        fx = 700.0 * rng.uniform(0.98, 1.02)
        return RadTanModel(fx, fx, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
    obj, obs, img, gt_ext, gtm = make_rig(n_cam=1, n_frame=10, noise_px=0.2, seed=0,
                                          w=W, h=H, model_factory=fac)
    Kp = gtm[0].K
    ge6 = [o for o in obs if len(o.point_rows) >= 6]

    seed_ma, Xcal, uvcal = _model_aware_seed(RadTanModel, Kp, ge6, obj)
    seed_k = _seed_from_K(RadTanModel, Kp)
    assert np.array_equal(seed_ma.params, seed_k.params), "RadTan seeds must be identical"

    seed_ma_kb, _, _ = _model_aware_seed(KannalaBrandtModel, Kp, ge6, obj)
    seed_k_kb = _seed_from_K(KannalaBrandtModel, Kp)
    assert not np.array_equal(seed_ma_kb.params, seed_k_kb.params), \
        "KB seeds must genuinely differ (closed-form theta-polynomial vs neutral)"
