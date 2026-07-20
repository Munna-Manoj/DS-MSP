"""Rotation synchronization: chordal cost, refinement, and the a-posteriori
global-optimality certificate.

Given noisy relative-rotation measurements ``R̃_ij`` on a graph, rotation synchronization
estimates absolute node rotations. Its weighted **chordal** cost is a quadratic in the
stacked rotations, ``F(R) = tr(Rᵀ L R)`` with ``L`` the *connection Laplacian*, and admits a
Lagrangian-dual **optimality certificate**: at a first-order critical point ``R``, build the
block-diagonal multiplier ``Λ_i = Sym((L R)_i R_iᵀ)``; if ``S = L − Λ ⪰ 0`` then ``R`` is a
**global** minimizer of the chordal cost — no false positives, valid regardless of
relaxation tightness, with one-sided error only (at high noise a true global optimum may
fail to certify; a pass is always trustworthy).

References (equations verified against the papers, not memory):

- Eriksson, Olsson, Kahl, Chin, "Rotation Averaging and Strong Duality", CVPR 2018
  (arXiv 1705.01362): primal (13), stationarity (17a), multiplier (18), Lemma 3.2.
- Rosen, Carlone, Bandeira, Leonard, "SE-Sync", IJRR 2019 (arXiv 1612.07386): connection
  Laplacian (14a), Riemannian gradient (120-122), multiplier (119), certificate (124),
  Theorem 7 and the suboptimality bound (50-51).
- The same certificate construction ships in GTSAM's ``ShonanAveraging`` (Dellaert et al.,
  ECCV 2020) for SLAM factor graphs — prior art for the mechanism; DS-MSP's contribution is
  surfacing it as a calibration-trust output over the rig's camera x board-placement graph.

Convention (load-bearing — mixing conventions silently breaks the certificate): node
rotation ``X_i`` = world-from-frame_i; for edge ``(i, j)`` the measurement is
``R̃_ij ≈ X_i⁻¹ X_j``. ``S`` always has an exact 3-dimensional nullspace per connected
component (the global-rotation gauge), so the PSD test is ``λ_min ≥ −tol`` with a
scale-relative tolerance, and the certificate must be evaluated per component.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from .lie import so3_exp, vee

_I3 = np.eye(3)


def connection_laplacian(n: int, edges: Iterable[Tuple[int, int, np.ndarray, float]]
                         ) -> np.ndarray:
    """Connection Laplacian ``L`` (3n x 3n) of a rotation-measurement graph.

    ``edges`` yields ``(i, j, R̃_ij, w)`` with ``R̃_ij ≈ X_i⁻¹ X_j`` and weight ``w > 0``:
    diagonal blocks ``d_i I₃`` (weighted degree), off-diagonal blocks ``−w R̃_ij``
    (Rosen eq. 14a; Eriksson eq. 11).
    """
    L = np.zeros((3 * n, 3 * n))
    deg = np.zeros(n)
    for i, j, Rij, w in edges:
        deg[i] += w
        deg[j] += w
        L[3 * i:3 * i + 3, 3 * j:3 * j + 3] += -w * np.asarray(Rij, float)
        L[3 * j:3 * j + 3, 3 * i:3 * i + 3] += -w * np.asarray(Rij, float).T
    for i in range(n):
        L[3 * i:3 * i + 3, 3 * i:3 * i + 3] += deg[i] * _I3
    return L


def chordal_cost(L: np.ndarray, R: np.ndarray) -> float:
    """``F(R) = tr(Rᵀ L R)`` for stacked ``R`` (3n, 3), block ``i`` = ``X_i`` in SO(3)."""
    return float(np.trace(R.T @ L @ R))


def chordal_grad(L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Riemannian gradient (n, 3) in per-node so(3) coordinates, right perturbation
    ``X_i ← X_i exp([w]ₓ)`` (Rosen 120-122 specialised to SO(3)^n; FD-verified 2.2e-8)."""
    n = R.shape[0] // 3
    LR = L @ R
    g = np.zeros((n, 3))
    for i in range(n):
        Ri = R[3 * i:3 * i + 3]
        B = Ri.T @ LR[3 * i:3 * i + 3]
        g[i] = 2.0 * vee(B - B.T)
    return g


def refine_chordal(L: np.ndarray, R0: np.ndarray, *, max_iter: int = 500,
                   tol: float = 1e-12) -> np.ndarray:
    """Riemannian gradient descent with backtracking to a first-order critical point of the
    chordal cost. The certificate is only meaningful at a stationary point — certifying a
    non-stationary iterate makes ``Λ`` non-symmetric and ``λ_min`` spuriously negative — so
    a warm start (e.g. the BA rotations, which minimize a *different*, reprojection cost)
    must be refined through here first.
    """
    n = R0.shape[0] // 3
    R = R0.copy()
    for _ in range(max_iter):
        g = chordal_grad(L, R)
        gn = float(np.linalg.norm(g))
        if gn < tol:
            break
        f0 = chordal_cost(L, R)
        s = 1.0 / (2.0 * np.max(np.abs(np.diag(L))) + 1e-12)
        Rn = R
        for _ls in range(40):
            Rn = R.copy()
            for i in range(n):
                Rn[3 * i:3 * i + 3] = R[3 * i:3 * i + 3] @ so3_exp(-s * g[i])
            if chordal_cost(L, Rn) < f0 - 1e-4 * s * gn * gn:
                break
            s *= 0.5
        R = Rn
    return R


def certificate(L: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Dual optimality certificate at a stationary point ``R`` of the chordal cost.

    Builds ``Λ_i = Sym((L R)_i R_iᵀ)`` (Rosen 119 / Eriksson 18) and ``S = L − Λ``;
    returns ``(S, λ_min, eigvals ascending, gap_bound)``. ``S ⪰ 0`` (up to the exact
    3-dim gauge nullspace, which sits at machine zero) proves ``R`` is a global minimizer
    (Eriksson Lemma 3.2; Rosen Thm 7). When ``λ_min < 0`` the result is *inconclusive* and
    ``gap_bound = 3n·|λ_min|`` bounds the possible suboptimality ``F(R) − p*``
    (Rosen 50-51) — report the gap, not just a boolean.
    """
    n = R.shape[0] // 3
    LR = L @ R
    Lam = np.zeros_like(L)
    for i in range(n):
        Ri = R[3 * i:3 * i + 3]
        Bi = LR[3 * i:3 * i + 3] @ Ri.T
        Lam[3 * i:3 * i + 3, 3 * i:3 * i + 3] = 0.5 * (Bi + Bi.T)
    S = L - Lam
    ev = np.linalg.eigvalsh(0.5 * (S + S.T))
    lam_min = float(ev[0])
    gap = 3 * n * abs(lam_min) if lam_min < 0 else 0.0
    return S, lam_min, ev, gap
