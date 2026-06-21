"""Tests for the in-house manifold LM solver (ds_msp.core.optimize).

The solver is geometry-blind, so we exercise it on a self-contained SE(3) point-cloud
registration problem: estimate (R, t) from Y_i ≈ R X_i + t. This isolates the solver's
two claims — manifold re-basing makes it converge at *large* rotation, and the robust
IRLS + GNC path makes it converge despite *outliers* — without dragging in camera models.
"""

from __future__ import annotations

import numpy as np

from ds_msp.core.lie import hat, so3_exp, so3_log
from ds_msp.core.optimize import lm_solve, schur_lm


def _make_problem(angle_deg, n=120, outlier_frac=0.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)) * 2.0
    axis = np.array([0.3, -0.7, 0.6])
    axis /= np.linalg.norm(axis)
    R_true = so3_exp(np.deg2rad(angle_deg) * axis)
    t_true = np.array([0.5, -1.2, 2.0])
    Y = (R_true @ X.T).T + t_true
    if noise:
        Y += rng.normal(size=Y.shape) * noise
    n_out = int(outlier_frac * n)
    if n_out:
        idx = rng.choice(n, n_out, replace=False)
        Y[idx] += rng.normal(size=(n_out, 3)) * 5.0          # gross outliers
    return X, Y, R_true, t_true


def _registration_fns(X, Y):
    """residual / jacobian / retract for state = (R, t), block = 3."""
    def residual(state):
        R, t = state
        return ((R @ X.T).T + t - Y).ravel()

    def jacobian(state):
        R, t = state
        n = X.shape[0]
        J = np.zeros((3 * n, 6))
        # d(R·exp(δω)·X + t)/dδω |_0 = -R [X]_×   ;   d/dδt = I
        for i in range(n):
            J[3 * i:3 * i + 3, :3] = -R @ hat(X[i])
            J[3 * i:3 * i + 3, 3:] = np.eye(3)
        return J

    def retract(state, delta):
        R, t = state
        return (R @ so3_exp(delta[:3]), t + delta[3:])

    return residual, jacobian, retract


def _rot_err_deg(R, R_ref):
    return float(np.degrees(np.linalg.norm(so3_log(R.T @ R_ref))))


def test_converges_at_large_rotation():
    """A 165° rotation is past where flat axis-angle wobbles near ‖r‖=π; the manifold
    re-basing solver should still nail it to machine precision from a crude init."""
    X, Y, R_true, t_true = _make_problem(165.0, seed=1)
    res, jac, ret = _registration_fns(X, Y)
    state0 = (np.eye(3), np.zeros(3))                        # identity init, far away
    out = lm_solve(state0, res, jac, ret, block=3, max_iter=100)
    R, t = out.state
    assert out.converged
    assert _rot_err_deg(R, R_true) < 1e-6
    assert np.allclose(t, t_true, atol=1e-6)


def test_robust_recovers_pose_under_outliers():
    """30% gross outliers: plain L2 is dragged off the true pose, but Cauchy IRLS with
    auto MAD scale + GNC recovers it to sub-degree accuracy."""
    X, Y, R_true, t_true = _make_problem(60.0, outlier_frac=0.30, noise=0.01, seed=2)
    res, jac, ret = _registration_fns(X, Y)
    state0 = (np.eye(3), np.zeros(3))

    l2 = lm_solve(state0, res, jac, ret, block=3, max_iter=100)
    robust = lm_solve(state0, res, jac, ret, block=3, max_iter=100,
                      robust_kernel="cauchy", robust_scale="auto",
                      gnc_start=10.0, gnc_iters=10)

    R_l2, _ = l2.state
    R_rb, t_rb = robust.state
    assert _rot_err_deg(R_rb, R_true) < 0.5
    assert np.allclose(t_rb, t_true, atol=0.1)
    # Robust must be meaningfully better than the outlier-dragged L2 fit.
    assert _rot_err_deg(R_rb, R_true) < _rot_err_deg(R_l2, R_true)


def test_clean_problem_matches_l2_exactly():
    """With no outliers, kernel='none' must reduce to ordinary Gauss-Newton and hit
    the exact solution — the robust path is strictly opt-in."""
    X, Y, R_true, t_true = _make_problem(40.0, seed=3)
    res, jac, ret = _registration_fns(X, Y)
    out = lm_solve((np.eye(3), np.zeros(3)), res, jac, ret, block=3)
    R, t = out.state
    assert _rot_err_deg(R, R_true) < 1e-7
    assert out.rms < 1e-9


def test_degenerate_hessian_does_not_crash():
    """All points collinear → rank-deficient Hessian. The damped-Cholesky jitter
    fallback must keep the solve finite (no LinAlgError, no NaN)."""
    n = 30
    X = np.zeros((n, 3))
    X[:, 0] = np.linspace(-1, 1, n)    # collinear on x-axis
    R_true = so3_exp(np.array([0.0, 0.0, 0.5]))
    Y = (R_true @ X.T).T + np.array([0.1, 0.2, 0.3])
    res, jac, ret = _registration_fns(X, Y)
    out = lm_solve((np.eye(3), np.zeros(3)), res, jac, ret, block=3, max_iter=50)
    R, t = out.state
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(t))


def test_schur_matches_closed_form_on_linear_problem():
    """The Schur complement must be exact. On a *linear* separable least-squares
    problem (shared params + independent per-group locals) schur_lm must reproduce
    the closed-form normal-equations solution to machine precision."""
    rng = np.random.default_rng(7)
    n_groups, sdim, ldim, m = 5, 3, 4, 8
    A = [rng.normal(size=(m, sdim)) for _ in range(n_groups)]
    B = [rng.normal(size=(m, ldim)) for _ in range(n_groups)]
    s_true = rng.normal(size=sdim)
    L_true = rng.normal(size=(n_groups, ldim))
    y = [A[i] @ s_true + B[i] @ L_true[i] for i in range(n_groups)]

    def residual(state):
        s, L = state
        return np.concatenate([A[i] @ s + B[i] @ L[i] - y[i] for i in range(n_groups)])

    def linearize(state):
        return ([A[i] @ state[0] + B[i] @ state[1][i] - y[i] for i in range(n_groups)],
                A, B)

    def retract(state, ds, dl):
        return (state[0] + ds, state[1] + dl)

    state0 = (np.zeros(sdim), np.zeros((n_groups, ldim)))
    out = schur_lm(state0, residual, linearize, retract,
                   n_groups=n_groups, shared_dim=sdim, local_dim=ldim, block=1)
    s, L = out.state
    # Closed form: stack the full design matrix and solve.
    cols = sdim + n_groups * ldim
    M = np.zeros((n_groups * m, cols))
    rhs = np.zeros(n_groups * m)
    for i in range(n_groups):
        M[i * m:(i + 1) * m, :sdim] = A[i]
        M[i * m:(i + 1) * m, sdim + i * ldim:sdim + (i + 1) * ldim] = B[i]
        rhs[i * m:(i + 1) * m] = y[i]
    x = np.linalg.lstsq(M, rhs, rcond=None)[0]
    assert np.allclose(s, x[:sdim], atol=1e-7)
    assert np.allclose(L.ravel(), x[sdim:], atol=1e-7)
    assert out.rms < 1e-7
