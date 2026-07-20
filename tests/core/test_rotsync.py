"""Rotation-synchronization certificate math (core/rotsync.py, FR-RIG-022).

The certificate's contract (Eriksson CVPR 2018 Lemma 3.2; Rosen IJRR 2019 Thm 7):
sound (a pass at a stationary point proves global optimality of the weighted chordal cost)
with one-sided error (a fail is inconclusive). The sign tests below are the mandatory
guards from the soundness audit: the noise-free problem MUST certify and a planted
non-global critical point MUST refuse — together they catch any sign/convention slip in
the L−Λ construction, which fails silently otherwise.
"""
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.core.rotsync import (certificate, chordal_cost, chordal_grad,
                                 connection_laplacian, refine_chordal)


def _random_rotations(n, rng, scale=1.0):
    R = np.zeros((3 * n, 3))
    for i in range(n):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        R[3 * i:3 * i + 3] = so3_exp(ax * rng.uniform(-scale, scale))
    return R


def _bipartite_edges(Z, n_a, n_b, rng, noise_rad=0.0):
    """Edges (a, n_a+b, Z_a Z_b^T (+noise), w) for every (a, b) pair."""
    edges = []
    for a in range(n_a):
        Za = Z[3 * a:3 * a + 3]
        for b in range(n_b):
            j = n_a + b
            Zb = Z[3 * j:3 * j + 3]
            M = Za @ Zb.T
            if noise_rad > 0:
                ax = rng.normal(size=3)
                ax /= np.linalg.norm(ax)
                M = so3_exp(ax * rng.normal(scale=noise_rad)) @ M
            edges.append((a, j, M, 1.0))
    return edges


@pytest.mark.req("FR-RIG-022")
@pytest.mark.jac
def test_chordal_grad_matches_finite_differences():
    """Analytic Riemannian gradient vs central finite differences, rel err < 1e-5."""
    rng = np.random.default_rng(0)
    n = 6
    Z = _random_rotations(n, rng)
    L = connection_laplacian(n, _bipartite_edges(Z, 2, 4, rng, noise_rad=0.1))
    R = _random_rotations(n, rng)          # generic (non-stationary) point
    g = chordal_grad(L, R)
    eps = 1e-6
    for i in range(n):
        for k in range(3):
            d = np.zeros(3)
            d[k] = eps
            Rp, Rm = R.copy(), R.copy()
            Rp[3 * i:3 * i + 3] = R[3 * i:3 * i + 3] @ so3_exp(d)
            Rm[3 * i:3 * i + 3] = R[3 * i:3 * i + 3] @ so3_exp(-d)
            fd = (chordal_cost(L, Rp) - chordal_cost(L, Rm)) / (2 * eps)
            assert abs(fd - g[i, k]) <= 1e-5 * max(1.0, abs(fd)), \
                f"node {i} axis {k}: analytic {g[i, k]} vs FD {fd}"


@pytest.mark.req("FR-RIG-022")
def test_certificate_passes_at_noise_free_global_optimum():
    """Sign test 1 (mandatory): the exact solution of a noise-free graph must certify,
    with exactly 3 machine-zero gauge eigenvalues and a clean gap to the 4th."""
    rng = np.random.default_rng(1)
    n_a, n_b = 3, 8
    n = n_a + n_b
    Z = _random_rotations(n, rng)
    L = connection_laplacian(n, _bipartite_edges(Z, n_a, n_b, rng))
    assert np.linalg.norm(chordal_grad(L, Z)) < 1e-9      # truth is stationary
    _S, lam_min, ev, gap = certificate(L, Z)
    scale = float(np.mean(np.diag(L)))
    assert lam_min / scale >= -1e-10
    assert gap <= 1e-9 * scale        # machine-zero gauge modes may leave a ~1e-13 residue
    assert np.all(np.abs(ev[:3]) < 1e-9 * scale)          # 3-dim gauge nullspace
    assert ev[3] > 1e-3 * scale                            # clean gap to the 4th


@pytest.mark.req("FR-RIG-022")
def test_certificate_refuses_planted_non_global_critical_point():
    """Sign test 2 (mandatory): a genuine critical point that is NOT the global optimum
    must NOT certify (this is the direction a sign/convention bug flips silently)."""
    rng = np.random.default_rng(2)
    n_a, n_b = 2, 6
    n = n_a + n_b
    Z = _random_rotations(n, rng)
    L = connection_laplacian(n, _bipartite_edges(Z, n_a, n_b, rng, noise_rad=0.02))
    # plant a far-away start and refine it to whatever critical point it falls into;
    # if that point's cost is well above the truth-refined cost it is non-global.
    R_bad = _random_rotations(n, rng, scale=np.pi * 0.9)
    R_bad = refine_chordal(L, R_bad, tol=1e-12)
    R_good = refine_chordal(L, Z, tol=1e-12)
    if chordal_cost(L, R_bad) <= chordal_cost(L, R_good) + 1e-9:
        pytest.skip("random far start happened to reach the global basin")
    _S, lam_min, _ev, gap = certificate(L, R_bad)
    scale = float(np.mean(np.diag(L)))
    assert lam_min / scale < -1e-6, "non-global critical point wrongly certified"
    assert gap > 0.0


@pytest.mark.req("FR-RIG-022")
def test_certificate_is_sound_under_moderate_noise():
    """With moderate rotational noise the (refined) estimate still certifies — the
    relaxation is tight far beyond calibration-grade noise (Eriksson Cor 4.1)."""
    rng = np.random.default_rng(3)
    n_a, n_b = 3, 10
    n = n_a + n_b
    Z = _random_rotations(n, rng)
    L = connection_laplacian(n, _bipartite_edges(Z, n_a, n_b, rng,
                                                 noise_rad=np.radians(5.0)))
    R = refine_chordal(L, Z, tol=1e-12)
    _S, lam_min, _ev, _gap = certificate(L, R)
    assert lam_min / float(np.mean(np.diag(L))) >= -1e-8
