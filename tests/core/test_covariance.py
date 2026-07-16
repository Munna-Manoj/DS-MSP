"""Sandwich vs naive covariance: agreement when the model is right, honesty when
it is contaminated.

Synthetic linear-Gaussian problem (r_i = A_i θ − y_i, 2-row blocks): with clean
Gaussian noise the two estimates must agree; under 20% gross contamination with a
Cauchy kernel, the Monte-Carlo covariance of the estimates over 200 re-solves is
matched by the sandwich (exact ψ′ bread) and NOT by the naive (Jᵀ W̃ J)⁻¹ σ̂².
"""

from __future__ import annotations

import numpy as np
import pytest

from ds_msp.core.covariance import (
    naive_covariance, robust_weight_derivative, sandwich_covariance,
)
from ds_msp.core.optimize import lm_solve
from ds_msp.core.robust import robust_weight

N_BLOCKS = 200
P = 3
SIGMA = 0.1
THETA_STAR = np.array([1.0, -2.0, 0.5])
CAUCHY_C = 3.0 * SIGMA


def _design(seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(2 * N_BLOCKS, P))


def _solve(A, y, kernel, scale):
    residual = lambda th: A @ th - y
    jacobian = lambda th: A
    retract = lambda th, d: th + d
    out = lm_solve(np.zeros(P), residual, jacobian, retract, block=2,
                   robust_kernel=kernel, robust_scale=scale, max_iter=60)
    return out.state


@pytest.mark.parametrize("kernel", ["huber", "pseudo_huber", "cauchy",
                                    "geman_mcclure", "barron"])
def test_weight_derivative_matches_finite_difference(kernel):
    """ω'(s) formulas (the Triggs term the sandwich bread needs) vs central FD."""
    s = np.linspace(0.05, 25.0, 400)
    h = 1e-6
    for c in (0.7, 2.0):
        an = robust_weight_derivative(s, kernel, c)
        fd = (robust_weight(s + h, kernel, c)
              - robust_weight(s - h, kernel, c)) / (2 * h)
        m = np.abs(s - c * c) > 1e-2 if kernel == "huber" else np.ones_like(s, bool)
        rel = np.abs(an[m] - fd[m]) / np.maximum(np.abs(fd[m]), 1e-12)
        assert rel.max() < 1e-6, f"{kernel} c={c}: {rel.max():.2e}"


def test_sandwich_matches_naive_on_clean_gaussian():
    """When the model is correct (L2, clean noise), the sandwich reduces to the
    naive estimate up to finite-sample noise."""
    A = _design()
    rng = np.random.default_rng(7)
    y = A @ THETA_STAR + rng.normal(scale=SIGMA, size=2 * N_BLOCKS)
    th = np.linalg.lstsq(A, y, rcond=None)[0]
    r = A @ th - y
    d_sw = np.diag(sandwich_covariance(A, r, kernel="none"))
    d_nv = np.diag(naive_covariance(A, r, kernel="none"))
    ratio = d_sw / d_nv
    assert np.all(ratio > 1 / 1.35) and np.all(ratio < 1.35), ratio


def test_sandwich_honest_under_contamination_naive_not():
    """20% gross outliers + Cauchy kernel: over 200 seeded re-solves, the
    Monte-Carlo covariance of θ̂ is matched by the sandwich (each diagonal within
    a loose factor 1.5, well inside factor 2) while the naive form is off by a
    visibly larger margin on every diagonal."""
    A = _design()
    n_out = int(0.2 * N_BLOCKS)

    thetas, sw_diags, nv_diags = [], [], []
    for k in range(200):
        rng = np.random.default_rng(1000 + k)
        y = A @ THETA_STAR + rng.normal(scale=SIGMA, size=2 * N_BLOCKS)
        idx = rng.choice(N_BLOCKS, n_out, replace=False)
        rows = np.stack([2 * idx, 2 * idx + 1], 1).ravel()
        y[rows] += rng.normal(scale=5.0, size=rows.size)      # gross outliers
        th = _solve(A, y, "cauchy", CAUCHY_C)
        r = A @ th - y
        thetas.append(th)
        sw_diags.append(np.diag(sandwich_covariance(A, r, kernel="cauchy",
                                                    scale=CAUCHY_C)))
        nv_diags.append(np.diag(naive_covariance(A, r, kernel="cauchy",
                                                 scale=CAUCHY_C)))

    mc = np.asarray(thetas).var(axis=0, ddof=1)
    sw = np.mean(sw_diags, axis=0)
    nv = np.mean(nv_diags, axis=0)

    sw_off = np.maximum(sw / mc, mc / sw)          # multiplicative distance to MC
    nv_off = np.maximum(nv / mc, mc / nv)
    # sandwich: honest (loose factor-of-2 bound; empirically within ~15%)
    assert np.all(sw_off < 1.5), f"sandwich off by {sw_off}"
    # naive: measurably dishonest on every component, and worse than sandwich
    assert np.all(nv_off > 1.35), f"naive unexpectedly close: {nv_off}"
    assert np.all(nv_off > sw_off * 1.2)
