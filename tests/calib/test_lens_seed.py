"""Lens fits must reach the right basin of every model across fields of view, without a seed."""
import numpy as np
import pytest

from ds_msp.calib.lens_seed import calibrate_from_views, fit_model_candidates
from ds_msp.core.lie import so3_exp
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.eucm import EUCMModel
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.radtan import RadTanModel
from ds_msp.models.ucm import UCMModel

pytestmark = pytest.mark.req("FR-CALIB-001", "FR-CALIB-010")

W, H = 1600, 1200


def _scene(model, half_fov_deg, rng, n_views=40, noise=0.15):
    """Random board views kept inside the lens's stated half-FOV and the image."""
    x, y = np.meshgrid(np.arange(9) * 0.05, np.arange(7) * 0.05)
    X = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    centre = X.mean(0)
    points, pixels, tries = [], [], 0
    while len(points) < n_views and tries < 20000:
        tries += 1
        R = so3_exp(rng.uniform(-0.9, 0.9, 3))
        dirn = so3_exp(rng.uniform(-1.8, 1.8, 3)) @ np.array([0, 0, 1.0])
        t = dirn * rng.uniform(0.35, 1.2) - R @ centre
        Xc = X @ R.T + t
        theta = np.degrees(np.arctan2(np.linalg.norm(Xc[:, :2], axis=1), Xc[:, 2]))
        if theta.max() > half_fov_deg:
            continue
        uv, ok = model.project(Xc)
        uv = np.asarray(uv, float)
        if not np.all(ok) or uv.min() < 2 or uv[:, 0].max() > W - 3 or uv[:, 1].max() > H - 3:
            continue
        points.append(X)
        pixels.append(uv + rng.normal(0, noise, uv.shape))
    assert len(points) == n_views
    return points, pixels


def _mapping_error(est, true, half_fov_deg, n=400):
    th = np.radians(np.linspace(0.5, half_fov_deg, n))
    az = np.linspace(0, 2 * np.pi, n)
    rays = np.column_stack([np.sin(th) * np.cos(az), np.sin(th) * np.sin(az), np.cos(th)])
    a, ok_a = est.project(rays)
    b, ok_b = true.project(rays)
    ok = np.asarray(ok_b, bool).ravel()
    assert np.asarray(ok_a, bool).ravel()[ok].all(), "fit left the valid domain inside the cone"
    return float(np.max(np.linalg.norm(np.asarray(a)[ok] - np.asarray(b)[ok], axis=1)))


CASES = [
    # model, half FOV the views may reach; both xi signs and narrow to very wide lenses
    (DoubleSphereModel(350, 350, 803, 595, -0.2, 0.6), 95),
    (DoubleSphereModel(300, 300, 795, 604, 0.5, 0.55), 110),
    (DoubleSphereModel(420, 420, 800, 600, -0.3, 0.55), 80),
    (UCMModel(420, 420, 804, 596, 0.6), 88),
    (EUCMModel(420, 420, 797, 603, 0.6, 1.1), 85),
    (KannalaBrandtModel(320, 320, 806, 594, 0.02, -0.004, 0.0005, 0.0), 100),
    (KannalaBrandtModel(500, 500, 792, 607, -0.04, 0.01, -0.002, 0.0), 75),
    (RadTanModel(900, 900, 805, 595, -0.12, 0.03, 0.0004, -0.0003, 0.0), 40),
]


@pytest.mark.parametrize("true,half_fov", CASES, ids=lambda c: getattr(c, "name", str(c)))
def test_recovers_the_lens_with_no_prior_intrinsics(true, half_fov):
    rng = np.random.default_rng(int(abs(true.params).sum()))
    points, pixels = _scene(true, half_fov, rng)
    res = calibrate_from_views(type(true), points, pixels, W, H)
    assert res is not None
    # Compared as a mapping inside the observed cone, since raw parameters can trade off.
    assert _mapping_error(res["model"], true, half_fov) < 1.5
    assert res["median_px"] < 0.5


def test_opposite_signs_of_xi_are_both_reachable():
    """Neither sign may be favoured: that would be a prior tuned on one lens family."""
    for xi in (-0.25, 0.45):
        true = DoubleSphereModel(330, 330, 800, 600, xi, 0.58)
        points, pixels = _scene(true, 100, np.random.default_rng(11))
        res = calibrate_from_views(DoubleSphereModel, points, pixels, W, H)
        assert res is not None and abs(res["model"].params[4] - xi) < 0.1


def test_tolerates_a_decentred_lens():
    true = KannalaBrandtModel(340, 340, 800 + 35, 600 - 28, 0.02, -0.004, 0.0005, 0.0)
    points, pixels = _scene(true, 95, np.random.default_rng(12))
    res = calibrate_from_views(KannalaBrandtModel, points, pixels, W, H)
    assert res is not None
    np.testing.assert_allclose(res["model"].params[2:4], true.params[2:4], atol=1.0)


def test_candidates_are_distinct_basins():
    true = DoubleSphereModel(350, 350, 800, 600, -0.2, 0.6)
    th = np.radians(np.linspace(1, 95, 800))
    az = np.linspace(0, 40, 800)
    rays = np.column_stack([np.sin(th) * np.cos(az), np.sin(th) * np.sin(az), np.cos(th)])
    pixels = np.asarray(true.project(rays)[0])
    found = fit_model_candidates(DoubleSphereModel, rays, pixels, 800, 600)
    assert found and abs(found[0][0].params[4] + 0.2) < 1e-3
    shapes = np.array([m.params[4:] for m, _ in found])
    for i in range(len(shapes)):
        for j in range(i):
            assert np.abs(shapes[i] - shapes[j]).max() > 0.05


def test_is_deterministic():
    true = EUCMModel(420, 420, 797, 603, 0.6, 1.1)
    points, pixels = _scene(true, 80, np.random.default_rng(13))
    a = calibrate_from_views(EUCMModel, points, pixels, W, H)
    b = calibrate_from_views(EUCMModel, points, pixels, W, H)
    np.testing.assert_array_equal(a["model"].params, b["model"].params)


def test_returns_none_when_there_is_no_consistent_view_set():
    assert calibrate_from_views(KannalaBrandtModel, [], [], W, H) is None
