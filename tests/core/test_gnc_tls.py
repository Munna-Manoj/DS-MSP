"""GNC-TLS high-breakdown robust optimization (ds_msp.core.optimize.gnc_tls_solve).

The redescending-kernel + MAD-auto-scale path in ``lm_solve`` is capped at MAD's 50% breakdown:
past half gross outliers the median-based scale is dragged up and the solve fails. ``gnc_tls_solve``
graduates a truncated-least-squares surrogate against an *explicit* noise bound, so it recovers
well past 50% from the declared identity seed and returns a hard inlier set. These tests pin both
the weight mechanics and that breakdown gap on a self-contained SE(3) registration problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from ds_msp.core.lie import hat, so3_exp, so3_log
from ds_msp.core.optimize import gnc_tls_schur_solve, gnc_tls_solve, lm_solve, schur_lm
from ds_msp.core.robust import gnc_tls_mu_init, gnc_tls_weight

pytestmark = pytest.mark.req("FR-CORE-001")


def _make_problem(angle_deg, n=140, outlier_frac=0.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    R_true = so3_exp(np.deg2rad(angle_deg) * axis)
    t_true = rng.normal(size=3)
    Y = (R_true @ X.T).T + t_true + noise * rng.normal(size=(n, 3))
    n_out = int(round(outlier_frac * n))
    if n_out:
        idx = rng.choice(n, n_out, replace=False)
        Y[idx] += rng.uniform(3.0, 8.0, size=(n_out, 1)) * rng.normal(size=(n_out, 3))
        is_out = np.zeros(n, bool)
        is_out[idx] = True
    else:
        is_out = np.zeros(n, bool)
    return X, Y, R_true, t_true, is_out


def _registration_fns(X, Y):
    def residual(state):
        R, t = state
        return ((R @ X.T).T + t - Y).ravel()

    def jacobian(state):
        R, t = state
        n = X.shape[0]
        J = np.zeros((3 * n, 6))
        for i in range(n):
            J[3 * i:3 * i + 3, :3] = -R @ hat(X[i])
            J[3 * i:3 * i + 3, 3:] = np.eye(3)
        return J

    def retract(state, d):
        R, t = state
        return (R @ so3_exp(d[:3]), t + d[3:])

    return residual, jacobian, retract


def _rot_err_deg(R, R_ref):
    return float(np.degrees(np.linalg.norm(so3_log(R.T @ R_ref))))


# --------------------------------------------------------------------------- weight mechanics
def test_gnc_tls_weight_three_regions_and_bounds():
    """Sure-inlier→1, sure-outlier→0, boundary in (0,1); all weights in [0,1]."""
    barc2, mu = 1.0, 1.0
    th1 = (mu + 1) / mu * barc2
    th2 = mu / (mu + 1) * barc2
    s = np.array([0.0, th2 * 0.5, 0.5 * (th1 + th2), th1 * 1.5])
    w = gnc_tls_weight(s, barc2, mu)
    assert w[0] == 1.0 and w[1] == 1.0          # below th2
    assert 0.0 < w[2] < 1.0                      # boundary
    assert w[3] == 0.0                           # above th1
    assert np.all((w >= 0.0) & (w <= 1.0))


def test_gnc_tls_weight_binarizes_as_mu_grows():
    """As μ→∞ the surrogate becomes hard TLS — weights are exactly 0 or 1."""
    s = np.array([0.01, 0.4, 0.9, 5.0])
    w = gnc_tls_weight(s, barc2=1.0, mu=1e6)
    assert np.all(np.isin(np.round(w, 6), [0.0, 1.0]))


def test_gnc_tls_mu_init_admits_everything():
    """The data-driven μ init makes th1 = 2·max(s): no row is rejected at the first level."""
    s = np.array([0.1, 0.5, 3.0])
    mu0 = gnc_tls_mu_init(s, barc2=0.01)
    assert np.all(gnc_tls_weight(s, 0.01, mu0) > 0.0)


# --------------------------------------------------------------------------- the breakdown gap
@pytest.mark.parametrize("outlier_frac", [0.6, 0.7])
def test_recovers_past_mad_50pct_breakdown(outlier_frac):
    """GNC-TLS with an explicit noise bound recovers at 60–70% gross outliers, where the
    MAD-auto-scale kernel path cannot (MAD breaks at 50%)."""
    X, Y, R_true, t_true, _ = _make_problem(60.0, outlier_frac=outlier_frac, noise=0.01, seed=3)
    res, jac, ret = _registration_fns(X, Y)
    state0 = (np.eye(3), np.zeros(3))

    out = gnc_tls_solve(state0, res, jac, ret, noise_bound=0.1, block=3)
    R, t = out.state
    assert _rot_err_deg(R, R_true) < 1.0
    assert np.allclose(t, t_true, atol=0.1)

    # the MAD-auto-scale path is dragged off the true pose at this contamination
    mad = lm_solve(state0, res, jac, ret, block=3, robust_kernel="geman_mcclure",
                   robust_scale="auto", gnc_start=10.0, gnc_iters=12, max_iter=120)
    assert _rot_err_deg(out.state[0], R_true) < _rot_err_deg(mad.state[0], R_true)


def test_returns_hard_inlier_set():
    """The returned weights are ≈binary and recover the planted inlier/outlier labels."""
    X, Y, R_true, _, is_out = _make_problem(60.0, outlier_frac=0.55, noise=0.01, seed=7)
    res, jac, ret = _registration_fns(X, Y)
    out = gnc_tls_solve((np.eye(3), np.zeros(3)), res, jac, ret, noise_bound=0.1, block=3)

    w = out.weights
    assert w is not None and np.all(np.isin(np.round(w, 6), [0.0, 1.0]))
    inlier = w > 0.5
    # planted inliers kept, planted outliers rejected (allow a tiny boundary slack)
    assert inlier[~is_out].mean() > 0.95
    assert inlier[is_out].mean() < 0.05


def test_clean_problem_matches_plain_solve():
    """With no outliers GNC-TLS must not hurt — it recovers the pose like a plain solve."""
    X, Y, R_true, t_true, _ = _make_problem(45.0, noise=0.0, seed=1)
    res, jac, ret = _registration_fns(X, Y)
    out = gnc_tls_solve((np.eye(3), np.zeros(3)), res, jac, ret, noise_bound=0.05, block=3)
    assert _rot_err_deg(out.state[0], R_true) < 1e-3
    assert np.allclose(out.state[1], t_true, atol=1e-3)


def test_deterministic():
    """No random init: identical inputs give bit-identical results."""
    X, Y, *_ = _make_problem(60.0, outlier_frac=0.6, noise=0.01, seed=5)
    res, jac, ret = _registration_fns(X, Y)
    a = gnc_tls_solve((np.eye(3), np.zeros(3)), res, jac, ret, noise_bound=0.1, block=3)
    b = gnc_tls_solve((np.eye(3), np.zeros(3)), res, jac, ret, noise_bound=0.1, block=3)
    assert np.array_equal(a.state[0], b.state[0]) and np.array_equal(a.state[1], b.state[1])


# ---------------------------------------------- the Schur (separable / sparse BA) path
def _separable_problem(n_groups=6, m=24, sdim=2, ldim=2, outlier_frac=0.0, noise=0.01, seed=0):
    """Shared params s + independent per-group locals L_i: y_ij = A_i·s + B_i·L_i (+noise),
    a fraction of observations grossly corrupted. Mirrors BA's shared-intrinsics + per-image-pose
    separability so the test exercises gnc_tls_schur_solve the way calibration BA would call it."""
    rng = np.random.default_rng(seed)
    A = [rng.normal(size=(m, sdim)) for _ in range(n_groups)]
    B = [rng.normal(size=(m, ldim)) for _ in range(n_groups)]
    s_true = rng.normal(size=sdim)
    L_true = rng.normal(size=(n_groups, ldim))
    y = [A[i] @ s_true + B[i] @ L_true[i] + noise * rng.normal(size=m) for i in range(n_groups)]
    n_out = int(round(outlier_frac * m))
    if n_out:
        for i in range(n_groups):
            idx = rng.choice(m, n_out, replace=False)
            y[i][idx] += rng.uniform(5.0, 15.0, size=n_out) * rng.choice([-1.0, 1.0], size=n_out)
    return A, B, y, s_true, L_true


def _separable_fns(A, B, y, n_groups):
    def residual(state):
        s, L = state
        return np.concatenate([A[i] @ s + B[i] @ L[i] - y[i] for i in range(n_groups)])

    def linearize(state):
        s, L = state
        return ([A[i] @ s + B[i] @ L[i] - y[i] for i in range(n_groups)], A, B)

    def retract(state, ds, dl):
        return (state[0] + ds, state[1] + dl)

    return residual, linearize, retract


def test_schur_gnc_tls_recovers_shared_params_past_50pct():
    """gnc_tls_schur_solve recovers the shared parameters at 60% gross outliers, where a plain
    Schur solve (block=1 reprojection-like residuals) is dragged off."""
    n_groups, m, sdim, ldim = 6, 24, 2, 2
    A, B, y, s_true, _ = _separable_problem(n_groups, m, sdim, ldim,
                                            outlier_frac=0.6, noise=0.01, seed=2)
    res, lin, ret = _separable_fns(A, B, y, n_groups)
    state0 = (np.zeros(sdim), np.zeros((n_groups, ldim)))

    out = gnc_tls_schur_solve(state0, res, lin, ret, noise_bound=0.1,
                              n_groups=n_groups, shared_dim=sdim, local_dim=ldim, block=1)
    s_rb, _ = out.state
    assert np.allclose(s_rb, s_true, atol=0.05)

    plain = schur_lm(state0, res, lin, ret, n_groups=n_groups,
                     shared_dim=sdim, local_dim=ldim, block=1)
    s_ls, _ = plain.state
    assert np.linalg.norm(s_rb - s_true) < np.linalg.norm(s_ls - s_true)


def test_schur_gnc_tls_returns_hard_inlier_set():
    """The returned per-block weights are ≈binary and recover the planted inlier/outlier labels."""
    n_groups, m, sdim, ldim = 5, 30, 2, 2
    rng = np.random.default_rng(11)
    A = [rng.normal(size=(m, sdim)) for _ in range(n_groups)]
    B = [rng.normal(size=(m, ldim)) for _ in range(n_groups)]
    s_true = rng.normal(size=sdim)
    L_true = rng.normal(size=(n_groups, ldim))
    y, is_out = [], []
    n_out = int(round(0.4 * m))
    for i in range(n_groups):
        yi = A[i] @ s_true + B[i] @ L_true[i] + 0.01 * rng.normal(size=m)
        idx = rng.choice(m, n_out, replace=False)
        yi[idx] += rng.uniform(5.0, 15.0, size=n_out) * rng.choice([-1.0, 1.0], size=n_out)
        oi = np.zeros(m, bool)
        oi[idx] = True
        y.append(yi)
        is_out.append(oi)
    res, lin, ret = _separable_fns(A, B, y, n_groups)
    out = gnc_tls_schur_solve((np.zeros(sdim), np.zeros((n_groups, ldim))), res, lin, ret,
                              noise_bound=0.1, n_groups=n_groups, shared_dim=sdim,
                              local_dim=ldim, block=1)
    w = out.weights
    assert w is not None and np.all(np.isin(np.round(w, 6), [0.0, 1.0]))
    # weights are stacked group0..groupN in linearize order; compare to the planted labels
    inlier = (w > 0.5).reshape(n_groups, m)
    planted_out = np.array(is_out)
    assert inlier[~planted_out].mean() > 0.95
    assert inlier[planted_out].mean() < 0.05


def test_schur_gnc_tls_clean_matches_plain():
    """No outliers: the Schur GNC-TLS path recovers the shared params like a plain Schur solve."""
    n_groups, m, sdim, ldim = 5, 20, 2, 2
    A, B, y, s_true, _ = _separable_problem(n_groups, m, sdim, ldim, noise=0.0, seed=4)
    res, lin, ret = _separable_fns(A, B, y, n_groups)
    out = gnc_tls_schur_solve((np.zeros(sdim), np.zeros((n_groups, ldim))), res, lin, ret,
                              noise_bound=0.05, n_groups=n_groups, shared_dim=sdim,
                              local_dim=ldim, block=1)
    assert np.allclose(out.state[0], s_true, atol=1e-3)
