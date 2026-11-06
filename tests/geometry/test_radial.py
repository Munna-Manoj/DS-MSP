"""The radial seed must recover poses with no lens knowledge, including rays past 90 degrees."""
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.geometry.radial import radial_seed, radial_view_candidates

pytestmark = pytest.mark.req("FR-CALIB-002", "FR-CALIB-010")

CX, CY = 640.0, 480.0


def _equidistant(Xc, f=300.0):
    """A pure theta lens: valid for any incidence angle, and unknown to the code under test."""
    theta = np.arctan2(np.linalg.norm(Xc[:, :2], axis=1), Xc[:, 2])
    az = Xc[:, :2] / np.maximum(np.linalg.norm(Xc[:, :2], axis=1, keepdims=True), 1e-12)
    return az * (f * theta)[:, None] + (CX, CY)


def _board(nx=8, ny=6, sq=0.05):
    x, y = np.meshgrid(np.arange(nx) * sq, np.arange(ny) * sq)
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])


def _views(rng, n=20, wide=True):
    X = _board()
    centre = X.mean(0)
    views = []
    for _ in range(n):
        R = so3_exp(rng.uniform(-0.7, 0.7, 3))
        # Off-axis placements reach large incidence angles; depth keeps the board in view.
        off = rng.uniform(-1.0, 1.0, 2) * (1.0 if wide else 0.2)
        depth = rng.uniform(-0.1, 0.5) if wide else rng.uniform(0.4, 0.8)
        t = np.array([off[0], off[1], depth]) - R @ centre
        views.append((R, t))
    return X, views


def test_planar_views_recover_pose_past_ninety_degrees():
    rng = np.random.default_rng(3)
    X, views = _views(rng)
    pixels, reach = [], 0.0
    for R, t in views:
        Xc = X @ R.T + t
        reach = max(reach, float(np.degrees(np.arctan2(np.linalg.norm(Xc[:, :2], axis=1),
                                                       Xc[:, 2]).max())))
        pixels.append(_equidistant(Xc) + rng.normal(0, 0.1, (len(X), 2)))
    assert reach > 100.0, "the scene must exercise rays a pinhole cannot represent"
    seed = radial_seed([X] * len(views), pixels, (CX, CY))
    assert seed is not None and len(seed.views) >= 0.7 * len(views)
    # A seed, so judged by its typical error and by the absence of a wrong branch.
    rot = [np.degrees(np.arccos(np.clip((np.trace(views[v.index][0].T @ v.R) - 1) / 2, -1, 1)))
           for v in seed.views]
    tra = [np.linalg.norm(views[v.index][1] - v.t) for v in seed.views]
    assert np.median(rot) < 0.5 and max(rot) < 3.0
    assert np.median(tra) < 0.02 and max(tra) < 0.1


def test_non_planar_target_uses_the_general_constraint():
    rng = np.random.default_rng(4)
    X = np.vstack([_board(), _board() @ np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0.]]) + (0, 0.3, 0)])
    pixels, poses = [], []
    for _ in range(10):
        R, t = so3_exp(rng.uniform(-0.4, 0.4, 3)), np.array([*rng.uniform(-0.3, 0.3, 2), 1.0])
        poses.append((R, t))
        pixels.append(_equidistant(X @ R.T + t) + rng.normal(0, 0.1, (len(X), 2)))
    seed = radial_seed([X] * 10, pixels, (CX, CY))
    assert seed is not None
    for v in seed.views:
        assert np.linalg.norm(poses[v.index][0] - v.R) < 0.03


def test_views_are_independent_of_the_focal_length():
    """Equation (1) has no lens in it, so scaling the lens must not move the rotation."""
    X, views = _views(np.random.default_rng(5), n=1)
    R, t = views[0]
    Xc = X @ R.T + t
    a = radial_view_candidates(_equidistant(Xc, 300.0) - (CX, CY), X)
    b = radial_view_candidates(_equidistant(Xc, 700.0) - (CX, CY), X)
    assert a and b
    np.testing.assert_allclose(a[0][0], b[0][0], atol=1e-9)
    np.testing.assert_allclose(a[1][0], b[1][0], atol=1e-9)


def test_a_mis_decoded_view_does_not_shape_the_lens():
    rng = np.random.default_rng(6)
    X, views = _views(rng, wide=False)
    pixels = [_equidistant(X @ R.T + t) + rng.normal(0, 0.1, (len(X), 2)) for R, t in views]
    clean = radial_seed([X] * len(views), pixels, (CX, CY))
    bad = [p.copy() for p in pixels]
    bad[0] = bad[0][rng.permutation(len(X))]                  # ids scrambled, pixels plausible
    seed = radial_seed([X] * len(views), bad, (CX, CY))
    assert seed is not None and 0 not in {v.index for v in seed.views}
    assert abs(seed.coeffs[0] / clean.coeffs[0] - 1.0) < 0.02


def test_too_few_views_gives_no_seed_rather_than_a_guess():
    X, views = _views(np.random.default_rng(7), n=1)
    R, t = views[0]
    assert radial_seed([X], [_equidistant(X @ R.T + t)], (CX, CY)) is None


def test_result_is_deterministic():
    rng = np.random.default_rng(8)
    X, views = _views(rng)
    pixels = [_equidistant(X @ R.T + t) + rng.normal(0, 0.2, (len(X), 2)) for R, t in views]
    a, b = (radial_seed([X] * len(views), pixels, (CX, CY)) for _ in range(2))
    np.testing.assert_array_equal(a.coeffs, b.coeffs)
    np.testing.assert_array_equal(a.rays, b.rays)
