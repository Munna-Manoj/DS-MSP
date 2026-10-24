"""The pose-only stage (cameras and intrinsics fixed) must use the block-diagonal Schur path
and agree with the dense solver (NFR-PERF-005)."""
import copy

import numpy as np
import pytest

from ds_msp.rig import bundle
from ds_msp.rig.calibrate import calibrate_rig
from ds_msp.rig.pipeline import make_fixed_intrinsic_front_end

from ._synth import make_rig

pytestmark = pytest.mark.req("NFR-PERF-005")


def _perturbed(rig, rng):
    out = copy.copy(rig)
    out.object_poses = {}
    for k, T in rig.object_poses.items():
        P = T.copy()
        P[:3, 3] += rng.normal(0, 0.01, 3)
        out.object_poses[k] = P
    return out


def test_pose_only_refine_is_sparse_and_matches_dense(monkeypatch):
    obj, obs, img_size, _ext, models = make_rig(n_cam=2, n_frame=15, noise_px=0.3, seed=1)
    rig = calibrate_rig(obj, copy.deepcopy(obs), img_size, fix_intrinsics=True,
                        front_end=make_fixed_intrinsic_front_end(models), structure_mode="off")
    start = _perturbed(rig, np.random.default_rng(0))
    kw = dict(fix_intrinsics=True, fix_extrinsics=True, robust_kernel="huber", robust_scale="auto")
    dense = bundle.refine(start, obs, sparse=False, **kw)
    calls = []
    real = bundle.lm_solve
    monkeypatch.setattr(bundle, "lm_solve", lambda *a, **k: calls.append(1) or real(*a, **k))
    sparse = bundle.refine(start, obs, **kw)
    assert calls == []                                   # no dense fallback
    for k in dense.object_poses:
        np.testing.assert_allclose(sparse.object_poses[k], dense.object_poses[k], atol=1e-7)
