r"""Parameter covariance for calibration reports (pure NumPy).

Two estimates of ``Cov(θ̂)`` at a converged robust solve, both computed from the
same ``(J, r)`` the solver ends with:

* :func:`naive_covariance` — the model-trusting ``σ̂²·(Jᵀ W̃ J)⁻¹`` (the CRLB /
  Fisher form). Exact when the assumed noise model is correct; **over- or
  under-confident under contamination or misspecification**, because it assumes
  the score covariance equals the Fisher information.
* :func:`sandwich_covariance` — the M-estimator asymptotic (Huber) sandwich::

      Cov(θ̂) ≈ G⁻¹ · M · G⁻ᵀ

  with the *bread* ``G = Σ_i w_i ∂ψ_i/∂θ`` (the exact estimating-equation
  Jacobian, including the redescending kernel's ``ω′`` curvature term) and the
  *meat* ``M = Σ_i (w_i ω_i)² (J_iᵀ r_i)(J_iᵀ r_i)ᵀ`` (empirical score
  covariance, HC1-corrected). Honest under contamination — it measures the score
  scatter that actually occurred instead of the one the model promises — and
  reduces to the naive form when the model is correct.

Conventions match :mod:`.robust` / :mod:`.optimize`: residual rows come in
consecutive ``block``-row groups, one robustness decision per group, per-block
score ``ψ_i = ω(s_i)·J_iᵀ r_i`` with ``s_i = ‖r_i‖²`` and IRLS weight
``ω = 2ρ′(s)``.
"""

from __future__ import annotations

import numpy as np

from .robust import VALID_KERNELS, _barron_bd, robust_weight


def robust_weight_derivative(s: np.ndarray, kernel: str, scale: float,
                             alpha: float = -2.0) -> np.ndarray:
    r"""``ω′(s) = dω/ds`` of the IRLS weight in :func:`.robust.robust_weight`.

    This is the Triggs curvature term — deliberately omitted from the forward
    solver (see :mod:`.robust`'s module docstring), but required here because the
    sandwich *bread* is the exact derivative of the estimating equation:
    ``∂ψ_i/∂θ = ω_i·J_iᵀJ_i + 2ω′_i·(J_iᵀr_i)(J_iᵀr_i)ᵀ``. All formulas verified
    against central differences of :func:`.robust.robust_weight`.
    """
    s = np.asarray(s, float)
    if kernel == "none":
        return np.zeros_like(s)
    c2 = scale * scale
    if kernel == "huber":
        root = np.sqrt(np.maximum(s, 1e-24))
        return np.where(s <= c2, 0.0,
                        -0.5 * scale / (root * np.maximum(s, 1e-24)))
    if kernel == "pseudo_huber":
        return -0.5 / c2 * np.power(1.0 + s / c2, -1.5)
    if kernel == "cauchy":
        u = 1.0 + s / c2
        return -1.0 / (c2 * u * u)
    if kernel == "geman_mcclure":
        u = 1.0 + s / c2
        return -2.0 / (c2 * u * u * u)
    if kernel == "barron":
        b, d = _barron_bd(alpha)
        return (0.5 * d - 1.0) / (c2 * b) * np.power(s / (c2 * b) + 1.0,
                                                     0.5 * d - 2.0)
    raise ValueError(f"kernel must be one of {VALID_KERNELS!r}, got {kernel!r}")


def _blocks(J: np.ndarray, r: np.ndarray, weights: np.ndarray | None, kernel: str, scale: float,
            alpha: float, block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                np.ndarray]:
    """Shared per-block quantities: reshaped ``(J_i, r_i)``, squared block
    residuals ``s_i``, user weights ``w_i``, and kernel weights ``ω_i``."""
    n_rows, p = J.shape
    if n_rows % block:
        raise ValueError(f"J has {n_rows} rows, not a multiple of block {block}")
    n = n_rows // block
    Jp = np.asarray(J, float).reshape(n, block, p)
    rp = np.asarray(r, float).reshape(n, block)
    s = np.einsum("nk,nk->n", rp, rp)
    om = robust_weight(s, kernel, scale, alpha)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    return Jp, rp, s, w, om


def _sym_inv(A: np.ndarray, jitter: float) -> np.ndarray:
    """Inverse of a (possibly barely indefinite) symmetric matrix with a
    trace-relative Tikhonov jitter — the bread of a redescending kernel can lose
    positive-definiteness at a contaminated solve."""
    p = A.shape[0]
    A = 0.5 * (A + A.T)
    tr = abs(float(np.trace(A))) / max(p, 1)
    return np.linalg.inv(A + jitter * max(tr, 1.0) * np.eye(p))


def sandwich_covariance(J: np.ndarray, r: np.ndarray,
                        weights: np.ndarray | None = None,
                        kernel: str = "none", scale: float = 1.0, *,
                        alpha: float = -2.0, block: int = 2,
                        jitter: float = 1e-12) -> np.ndarray:
    r"""M-estimator sandwich covariance ``G⁻¹ M G⁻ᵀ`` at a converged solve.

    Parameters mirror the solver's: ``J (n_rows, p)`` and ``r (n_rows,)`` at the
    solution, per-block user ``weights (n_rows//block,)``, and the *same*
    ``kernel`` / ``scale`` / ``alpha`` the solve used (with
    ``robust_scale='auto'``, pass the final iteration's scale).

    Bread (exact estimating-equation Jacobian, per block)::

        G_i = w_i [ ω_i·J_iᵀJ_i + 2ω′_i·(J_iᵀr_i)(J_iᵀr_i)ᵀ ]

    Meat (empirical score covariance, MacKinnon–White HC1 dof correction)::

        M_i = (w_i ω_i)²·(J_iᵀr_i)(J_iᵀr_i)ᵀ,   M ← M · m/(m − p)

    Returns the symmetric ``(p, p)`` covariance of the parameter estimate.
    """
    Jp, rp, s, w, om = _blocks(J, r, weights, kernel, scale, alpha, block)
    p = J.shape[1]
    omp = robust_weight_derivative(s, kernel, scale, alpha)
    a = np.einsum("nkp,nk->np", Jp, rp)                    # J_iᵀ r_i  (n, p)
    JtJ = np.einsum("nki,nkj->nij", Jp, Jp)                # (n, p, p)
    G = (np.einsum("n,nij->ij", w * om, JtJ)
         + 2.0 * np.einsum("n,ni,nj->ij", w * omp, a, a))
    M = np.einsum("ni,n,nj->ij", a, (w * om) ** 2, a)
    m_rows = float(J.shape[0])
    M *= m_rows / max(m_rows - p, 1.0)                     # HC1
    Ginv = _sym_inv(G, jitter)
    cov = Ginv @ M @ Ginv.T
    return 0.5 * (cov + cov.T)


def clustered_sandwich_covariance(J: np.ndarray, r: np.ndarray, cluster_id: np.ndarray,
                                  weights: np.ndarray | None = None,
                                  kernel: str = "none", scale: float = 1.0, *,
                                  alpha: float = -2.0, block: int = 2,
                                  jitter: float = 1e-12,
                                  small_cluster_correction: bool = True) -> np.ndarray:
    r"""Frame/cluster-robust sandwich covariance — the fix for
    :func:`sandwich_covariance`'s measured real-data under-coverage (4-9x on a real rig's
    frame bootstrap) when per-block scores are actually correlated within a cluster (e.g.
    corners from the same synchronized board placement, across cameras) but the plain
    sandwich treats every corner as independent.

    Same bread ``G`` as :func:`sandwich_covariance` (clustering does not change it: ``G`` is
    the derivative of the *total* score equation and carries no grouping). The fix is
    entirely in the meat, following Liang & Zeger (1986) / Cameron & Miller (2015) eq. 11's
    clustered "sandwich": sum each cluster's per-block scores FIRST, then outer-product
    (instead of outer-producting every block independently)::

        ψ_i = w_i·ω_i·(J_iᵀ r_i)     (per-block score, same as the plain sandwich's meat)
        Ψ_g = Σ_{i∈g} ψ_i             (summed within cluster g)
        M   = Σ_g Ψ_g Ψ_gᵀ

    which lets correlated within-cluster scores reinforce instead of averaging away as if
    independent — verified on a synthetic known-correlation Monte-Carlo to close a 6.2x
    (std) / 38x (variance) under-coverage of the frame-level parameter direction down to a
    0.98x coverage ratio (0.98-1.10x under a contaminated/robust refit); see
    ``.ai/experiments/2026-07-17-stage-I-frame-clustered-sandwich-derivation.md``.

    With few clusters (DS-MSP's real rigs: ``G≈30-35`` frames), the cluster-robust variance
    estimator is downward-biased; ``small_cluster_correction=True`` (default) applies the
    standard scalar fix ``c = [G/(G−1)]·[(N−1)/(N−K)]`` (Cameron, Gelbach & Miller 2008,
    *Rev. Econ. Stat.* 90(3)) where ``N`` is the number of blocks (corners) and ``K = p`` —
    measured ``c ≈ 1.24`` on both this repo's real rigs, dominated by the ordinary
    ``(N−1)/(N−K)`` term (nuisance frame-pose parameters make ``K`` large relative to ``N``),
    not by ``G/(G−1)`` itself. **For inference downstream, use ``df = G − 1``, not ``N − K``.**

    Parameters
    ----------
    cluster_id : (n_blocks,) array-like
        Groups blocks into clusters, e.g. the frame id a ChArUco corner was detected in,
        **pooled across cameras** for a synchronized multi-camera rig (the independent
        replication unit is a physical board placement, not a per-camera image — see the
        derivation doc §"cluster definition" for the physical reasoning). Any hashable/
        ``np.unique``-able dtype; need not be contiguous integers.
    J, r, weights, kernel, scale, alpha, block, jitter
        Same as :func:`sandwich_covariance`.

    Returns
    -------
    (p, p) ndarray
        Symmetric clustered-sandwich covariance.

    References
    ----------
    Liang, K.-Y. & Zeger, S.L. (1986), "Longitudinal Data Analysis Using Generalized Linear
    Models," Biometrika 73(1):13-22. Cameron, A.C. & Miller, D.L. (2015), "A Practitioner's
    Guide to Cluster-Robust Inference," J. Human Resources 50(2), eq. 11. Cameron, A.C.,
    Gelbach, J.B. & Miller, D.L. (2008), "Bootstrap-Based Improvements for Inference with
    Clustered Errors," Rev. Econ. Stat. 90(3), small-cluster correction.
    """
    Jp, rp, s, w, om = _blocks(J, r, weights, kernel, scale, alpha, block)
    p = J.shape[1]
    n = Jp.shape[0]
    omp = robust_weight_derivative(s, kernel, scale, alpha)
    a = np.einsum("nkp,nk->np", Jp, rp)                    # J_iᵀ r_i  (n, p)
    JtJ = np.einsum("nki,nkj->nij", Jp, Jp)
    G = (np.einsum("n,nij->ij", w * om, JtJ)
         + 2.0 * np.einsum("n,ni,nj->ij", w * omp, a, a))

    cluster_id = np.asarray(cluster_id)
    if cluster_id.shape[0] != n:
        raise ValueError(f"cluster_id has {cluster_id.shape[0]} entries, expected {n} "
                         "(one per block)")
    psi = (w * om)[:, None] * a                             # (n, p) per-block score
    _, inv = np.unique(cluster_id, return_inverse=True)
    n_clusters = int(inv.max()) + 1 if n else 0
    Psi_g = np.zeros((n_clusters, p))
    np.add.at(Psi_g, inv, psi)                              # sum within cluster, THEN...
    M = Psi_g.T @ Psi_g                                     # ...outer-product (Liang-Zeger)

    if small_cluster_correction and n_clusters > 1:
        Gc = float(n_clusters)
        c = (Gc / max(Gc - 1.0, 1.0)) * ((float(n) - 1.0) / max(float(n) - p, 1.0))
        M = M * c

    Ginv = _sym_inv(G, jitter)
    cov = Ginv @ M @ Ginv.T
    return 0.5 * (cov + cov.T)


def naive_covariance(J: np.ndarray, r: np.ndarray,
                     weights: np.ndarray | None = None,
                     kernel: str = "none", scale: float = 1.0, *,
                     alpha: float = -2.0, block: int = 2,
                     sigma2: float | None = None,
                     jitter: float = 1e-12) -> np.ndarray:
    r"""Model-trusting covariance ``σ̂²·(Jᵀ W̃ J)⁻¹`` for comparison/reporting.

    ``W̃`` folds the kernel's IRLS weights (and user weights) exactly as the
    solver's normal equations do. ``sigma2`` is the per-row noise variance; when
    ``None`` it is estimated from the weighted residuals with the standard dof
    correction ``σ̂² = Σ w̃·r² / (m − p)``. Exact under the assumed i.i.d. noise
    model — over-confident under contamination (see :func:`sandwich_covariance`).
    """
    Jp, rp, s, w, om = _blocks(J, r, weights, kernel, scale, alpha, block)
    p = J.shape[1]
    wr = np.repeat(w * om, block)                          # per-row weights
    H = np.einsum("mi,m,mj->ij",
                  np.asarray(J, float), wr, np.asarray(J, float))
    if sigma2 is None:
        m_rows = float(J.shape[0])
        sigma2 = float((wr * np.asarray(r, float) ** 2).sum()
                       / max(m_rows - p, 1.0))
    cov = float(sigma2) * _sym_inv(H, jitter)
    return 0.5 * (cov + cov.T)
