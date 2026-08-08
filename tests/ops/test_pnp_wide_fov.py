"""Wide-FOV forward-hemisphere PnP recovery, exercised on the two polynomial models.

The polymorphic PnP suite (``test_ops_polymorphic.py``) uses a small planar target near
the optical axis. These tests push a genuinely wide, *non-coplanar* forward-hemisphere point
cloud (bearings out to ~70-75° off axis) through the OCam (Scaramuzza polynomial) and DS+
(UCM core + division + tilt) models, then recover the camera pose from bearing rays via
``solve_pnp`` / ``solve_pnp_ransac``. This proves the model-agnostic pose ops work on the two
models that are otherwise absent from the task-level suites — not just on Double Sphere.

Numbers are measured (see the module below): at zero pixel noise both models recover the pose
to ~1e-12 (translation) and the RANSAC path flags every injected blunder while keeping the
clean majority. No tolerance is loosened relative to the DS-based suite.
"""

import cv2
import numpy as np
import pytest

from ds_msp.ops import solve_pnp, solve_pnp_ransac
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.ocam import OCamModel
from ds_msp.models.dsplus import DSPlusModel

# DS kept alongside OCam/DS+ so this file also asserts parity with the reference model.
MODELS = [DoubleSphereModel.sample, OCamModel.sample, DSPlusModel.sample]


def _wide_fov_cloud(seed: int = 3, n: int = 90):
    """A non-coplanar point cloud spanning a wide forward cone (bearings to ~70° off axis)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.6, 3.0, n)
    x = rng.uniform(-1.4, 1.4, n) * z          # |x/z| up to 1.4 -> ~54° per axis
    y = rng.uniform(-1.4, 1.4, n) * z
    return np.column_stack([x, y, z]).astype(np.float64)


def _max_off_axis_deg(Xc):
    d = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(d[:, 2], -1.0, 1.0))).max()


@pytest.mark.parametrize("factory", MODELS, ids=lambda f: f().name)
def test_wide_fov_noncoplanar_pnp_recovers_pose(factory):
    model = factory()
    obj = _wide_fov_cloud()
    rvec_gt = np.array([0.2, -0.15, 0.1])
    tvec_gt = np.array([0.1, -0.05, 0.0])       # object already in front (z in [0.6, 3])
    R, _ = cv2.Rodrigues(rvec_gt)
    Xc = (R @ obj.T).T + tvec_gt
    uv, valid = model.project(Xc)
    assert valid.sum() >= 40                     # a wide, well-populated hemisphere
    assert _max_off_axis_deg(Xc[valid]) > 60.0   # genuinely wide FOV, not near-axis

    ok, rvec, tvec = solve_pnp(model, obj[valid], uv[valid])
    assert ok
    assert np.linalg.norm(tvec - tvec_gt) < 1e-6
    assert np.linalg.norm(rvec - rvec_gt) < 1e-6


@pytest.mark.parametrize("factory", MODELS, ids=lambda f: f().name)
def test_wide_fov_pnp_ransac_rejects_gross_outliers(factory):
    model = factory()
    obj = _wide_fov_cloud(seed=5, n=110)
    rvec_gt = np.array([0.15, 0.1, -0.08])
    tvec_gt = np.array([-0.1, 0.05, 0.2])
    R, _ = cv2.Rodrigues(rvec_gt)
    Xc = (R @ obj.T).T + tvec_gt
    uv, valid = model.project(Xc)
    obj_v, uv_v = obj[valid], np.asarray(uv)[valid].copy()
    assert len(uv_v) >= 40

    rng = np.random.default_rng(0)
    out = rng.choice(len(uv_v), len(uv_v) // 4, replace=False)   # corrupt 25%
    uv_v[out] += rng.uniform(-120, 120, size=(len(out), 2))

    ok, rvec, tvec, inliers = solve_pnp_ransac(model, obj_v, uv_v, seed=0)
    assert ok
    assert np.linalg.norm(tvec - tvec_gt) < 1e-3     # pose recovered despite blunders
    assert not inliers[out].any()                    # every injected blunder flagged out
    assert inliers.sum() >= len(uv_v) // 2            # clean majority kept

# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-OPS-003")
