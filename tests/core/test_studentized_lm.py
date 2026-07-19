"""Studentized (bounded-influence) IRLS inside lm_solve — the leverage counterexample.

A high-leverage point pulls the fit toward itself so its OWN residual stays small:
a residual-only kernel under-weights it ("self-masking"). Applying the kernel to the
studentized squared residual s̃_i = r_iᵀ(I − H_ii + εI)⁻¹ r_i (hat-matrix leverage
deflation undone) inflates exactly those points so the kernel finally sees them.

A classic high-leverage counterexample, constructed independently here. Honest
scope note: inside a monotone-descent LM (accept/reject on the robust cost), what
studentization buys is *bias removal at the converged minimum* — the leverage point
retains enough weight in plain IRLS to visibly bias the estimate, and studentization
kills that bias. Escaping a leverage-controlled LS *basin* is a different mechanism
(raw IRLS fixed-point iteration can hop it; enforced descent cannot — under either
metric), which is why this test measures converged bias from a neutral init rather
than basin escape from the LS init.
"""

from __future__ import annotations

import numpy as np

from ds_msp.core.optimize import lm_solve
from ds_msp.core.robust import (
    STUDENT_EPS, _deflation_blocks, studentized_scale_factors, studentized_sq,
)


# --- scenario: z_i = s·x_i + t on 2-D points, theta = (s, t1, t2) -----------------
# 30 compact inliers (|x| ≤ 1, true s = 2, σ = 0.002) + one leverage point at
# x = (60, 60) generated from the WRONG scale s = 1.7. It dominates JᵀJ ~ 7200 : 20,
# so the LS fit tilts to s ≈ 1.70 and its own residual self-masks there.

def _make_problem(seed=0, n=30, lev=60.0, noise=0.002):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 2))
    s_true, t_true = 2.0, np.array([0.3, -0.2])
    z = s_true * x + t_true + rng.normal(scale=noise, size=(n, 2))
    x_o = np.array([[lev, lev]])
    z_o = 1.7 * x_o + t_true                     # consistent with s = 1.7: self-masks
    return np.vstack([x, x_o]), np.vstack([z, z_o]), s_true


def _callbacks(X, Z):
    n = len(X)
    J = np.zeros((2 * n, 3))
    J[0::2, 0] = X[:, 0]
    J[1::2, 0] = X[:, 1]
    J[0::2, 1] = 1.0
    J[1::2, 2] = 1.0

    def residual(theta):
        return ((theta[0] * X + theta[1:]) - Z).ravel()

    def jacobian(theta):
        return J

    def retract(theta, d):
        return theta + d

    return residual, jacobian, retract


def _ls_fit(X, Z):
    _, jacobian, _ = _callbacks(X, Z)
    theta, *_ = np.linalg.lstsq(jacobian(None), Z.ravel(), rcond=None)
    return theta


def test_scale_factors_match_studentized_sq():
    """F_i = (I − H_ii + εI)⁻¹ (adjugate closed form) must reproduce studentized_sq's
    linalg.solve quadratic form: s̃_i = r_iᵀ F_i r_i."""
    X, Z, _ = _make_problem()
    residual, jacobian, _ = _callbacks(X, Z)
    theta = _ls_fit(X, Z)
    J = jacobian(theta)
    r = residual(theta)
    F = studentized_scale_factors(J, block=2)
    rp = r.reshape(-1, 2)
    s_from_factors = np.einsum("nk,nkl,nl->n", rp, F, rp)
    s_ref = studentized_sq(J, r, block=2)
    assert np.allclose(s_from_factors, s_ref, rtol=1e-10, atol=1e-14)


def test_adjugate_inverse_matches_linalg_inv():
    """The closed-form 2×2 adjugate path must equal np.linalg.inv (exact algebra)."""
    X, Z, _ = _make_problem(seed=3)
    _, jacobian, _ = _callbacks(X, Z)
    J = jacobian(None)
    M = _deflation_blocks(J, None, 2, STUDENT_EPS)
    F = studentized_scale_factors(J, block=2)
    assert np.allclose(F, np.linalg.inv(M), rtol=1e-12, atol=1e-14)


def test_leverage_point_inflated_at_ls_fit():
    """At the LS fit (which the leverage point controls), studentization inflates the
    leverage block by ~1/ε (its hat block eats its residual) while genuinely compact
    inlier blocks are left nearly untouched — the unmasking mechanism itself."""
    X, Z, _ = _make_problem()
    residual, jacobian, _ = _callbacks(X, Z)
    theta = _ls_fit(X, Z)
    r = residual(theta).reshape(-1, 2)
    raw = np.einsum("nk,nk->n", r, r)
    stud = studentized_sq(jacobian(theta), residual(theta), block=2)
    # leverage block (last): inflated by nearly the 1/eps = 20x cap
    assert stud[-1] > 15.0 * raw[-1]
    # inlier blocks: leverage h_i ~ 1/n, inflation stays < ~15%
    assert np.all(stud[:-1] < 1.2 * raw[:-1])


def test_lm_solve_studentize_removes_leverage_bias():
    """From a neutral init, plain Cauchy IRLS converges with a visible bias in the
    scale parameter (the self-masking leverage point re-enters the inlier band near
    its own fit and keeps pulling); studentize=True removes it (>= 4x smaller error
    at seed 0; >= 2x across all seeds tried)."""
    X, Z, s_true = _make_problem()
    residual, jacobian, retract = _callbacks(X, Z)
    theta0 = np.array([1.0, 0.0, 0.0])
    kw = dict(block=2, robust_kernel="cauchy", robust_scale="auto",
              robust_scale_floor=0.05, max_iter=100)

    out_plain = lm_solve(theta0, residual, jacobian, retract, studentize=False, **kw)
    out_stud = lm_solve(theta0, residual, jacobian, retract, studentize=True, **kw)

    err_plain = abs(out_plain.state[0] - s_true)
    err_stud = abs(out_stud.state[0] - s_true)
    assert err_plain > 1e-3, f"plain unexpectedly unbiased: err={err_plain}"
    assert err_stud < 5e-4, f"studentize failed to de-bias: err={err_stud}"
    assert err_stud < 0.25 * err_plain
