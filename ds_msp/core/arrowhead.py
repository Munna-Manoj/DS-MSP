"""Spectral quantities of a symmetric block-arrowhead matrix without forming it.

A bundle-adjustment information matrix with its pose blocks ordered last is block-arrowhead:

    H = [[A,  B],
         [Bᵀ, D]],    D = blkdiag(D_1, ..., D_F),  each D_f of size b x b,

where ``A`` (C x C) holds the parameters every observation can touch (camera extrinsics and
intrinsics) and each ``D_f`` one pose that only its own observations touch. Storing ``A``, ``B``
and the blocks costs ``C² + C·bF + b²F`` numbers instead of ``(C + bF)²``; on a 25-camera rig with
1,662 board poses that is about 25 MB instead of 843 MB.

The eigenvalues are reached without the full matrix through Sylvester's law of inertia and
Haynsworth's inertia additivity. With ``D_f = V_f diag(d_f) V_fᵀ``, ``W = B·blkdiag(V_f)`` and any
``σ`` that is not an eigenvalue of a block,

    ν(σ) = #{eigenvalues of H below σ} = #{d < σ} + #{eigenvalues of S(σ) below 0},
    S(σ) = A − σI − W diag(1/(d − σ)) Wᵀ,

so ``ν`` costs one C x C symmetric eigenvalue problem. Bisection on ``ν`` gives any eigenvalue,
like LAPACK's ``stebz`` on a tridiagonal matrix. Between poles the ordered eigenvalues of
``S(σ)`` decrease with ``σ`` (``S'(σ) = −I − W diag(1/(d−σ)²) Wᵀ`` is negative definite), and
``λ`` is an eigenvalue of ``H`` exactly when one of them crosses zero, so Newton's method on that
eigenvalue converges quadratically once the root is isolated. Eigenvectors follow from the
same elimination: ``S(λ) x_c = 0`` and ``x_d = −(D − λI)⁻¹ Bᵀ x_c``.

Every count is exact for the matrix it is given, so the computed eigenvalues carry the usual
backward error of a dense symmetric solver, ``O(ε‖H‖)`` absolutely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

_EPS = float(np.finfo(float).eps)
_TINY = float(np.finfo(float).tiny)
_CHUNK = 4096          # columns of W per GEMM chunk when building S(σ)


@dataclass
class ArrowheadSym:
    """``H`` in block-arrowhead form.

    ``A`` (C, C) and ``D`` (F, b, b) are symmetric; ``B`` (C, b·F) couples them, its column
    ``b·f + i`` belonging to block ``f``. ``core`` (C,) and ``blocks`` (b·F,) give each row of
    ``A`` and each column of ``B`` its index in the original ordering, so the dense matrix is
    ``H[np.ix_(core, core)] = A``, ``H[np.ix_(core, blocks)] = B`` and so on."""

    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    core: np.ndarray
    blocks: np.ndarray

    @property
    def n(self) -> int:
        return len(self.core) + len(self.blocks)

    def diagonal(self) -> np.ndarray:
        """``diag(H)`` in the original ordering."""
        out = np.empty(self.n)
        out[self.core] = np.diag(self.A)
        out[self.blocks] = np.diagonal(self.D, axis1=1, axis2=2).ravel()
        return out

    def to_dense(self) -> np.ndarray:
        """The full matrix (tests and small problems only)."""
        H = np.zeros((self.n, self.n))
        H[np.ix_(self.core, self.core)] = self.A
        H[np.ix_(self.core, self.blocks)] = self.B
        H[np.ix_(self.blocks, self.core)] = self.B.T
        b = self.D.shape[1]
        for f in range(self.D.shape[0]):
            ix = self.blocks[b * f:b * f + b]
            H[np.ix_(ix, ix)] = self.D[f]
        return H

    def equilibrate_inplace(self) -> np.ndarray:
        """Scale to unit diagonal in place, ``H ← diag(s) H diag(s)`` with
        ``s = 1/sqrt(diag H)`` (floored like :func:`ds_msp.core.observability.equilibrate`), and
        return ``s``. Each entry is computed as the dense equilibration computes it,
        ``((H_ij s_i) s_j + (H_ji s_j) s_i) / 2``."""
        d = self.diagonal()
        eps = max(float(np.max(d)), 1.0) * 1e-15
        s = 1.0 / np.sqrt(np.where(d > eps, d, eps))
        sc, sb = s[self.core], s[self.blocks]
        A = self.A
        A[...] = ((A * sc[:, None]) * sc[None, :] + ((A.T * sc[None, :]) * sc[:, None])) * 0.5
        for j in range(0, self.B.shape[1], _CHUNK):
            Bj = self.B[:, j:j + _CHUNK]
            sbj = sb[j:j + _CHUNK]
            Bj[...] = ((Bj * sc[:, None]) * sbj[None, :] + (Bj * sbj[None, :]) * sc[:, None]) * 0.5
        b = self.D.shape[1]
        sD = sb.reshape(-1, b)
        D = self.D
        D[...] = ((D * sD[:, :, None]) * sD[:, None, :]
                  + (np.swapaxes(D, 1, 2) * sD[:, None, :]) * sD[:, :, None]) * 0.5
        return s

    def strong_pairs(self, corr_thresh: float) -> List[Tuple[int, int, float]]:
        """``(i, j, H[i, j])`` for every ``i < j`` (original ordering) with
        ``|H[i, j]| > corr_thresh``, sorted like a row-major scan of the dense matrix."""
        out: List[Tuple[int, int, float]] = []
        r, c = np.nonzero(np.abs(self.A) > corr_thresh)
        out += [(int(self.core[i]), int(self.core[j]), float(self.A[i, j])) for i, j in zip(r, c)]
        r, c = np.nonzero(np.abs(self.B) > corr_thresh)
        for i, j in zip(r, c):
            a, b_ = int(self.core[i]), int(self.blocks[j])
            out.append((a, b_, float(self.B[i, j])) if a < b_ else (b_, a, float(self.B[i, j])))
        b = self.D.shape[1]
        f, r, c = np.nonzero(np.abs(self.D) > corr_thresh)
        out += [(int(self.blocks[b * k + i]), int(self.blocks[b * k + j]), float(self.D[k, i, j]))
                for k, i, j in zip(f, r, c)]
        uniq: Dict[Tuple[int, int], float] = {(i, j): v for i, j, v in out if i < j}
        return [(i, j, uniq[(i, j)]) for i, j in sorted(uniq)]

    def gershgorin(self) -> Tuple[float, float]:
        """Bounds ``lo <= λ_min``, ``λ_max <= hi`` from the Gershgorin discs."""
        absB = np.abs(self.B)
        rc = np.abs(self.A).sum(1) - np.abs(np.diag(self.A)) + absB.sum(1)
        b = self.D.shape[1]
        absD = np.abs(self.D)
        rb = (absD.sum(2) - np.abs(np.diagonal(self.D, axis1=1, axis2=2))).ravel() + absB.sum(0)
        dc, db = np.diag(self.A), np.diagonal(self.D, axis1=1, axis2=2).ravel()
        lo = min(float(np.min(dc - rc)), float(np.min(db - rb))) if b else float(np.min(dc - rc))
        hi = max(float(np.max(dc + rc)), float(np.max(db + rb))) if b else float(np.max(dc + rc))
        return lo, hi


class ArrowheadSpectrum:
    """Eigenvalues (by inertia counts) and eigenvectors of an :class:`ArrowheadSym`.

    The blocks are rotated to their eigenbases, ``D_f = V_f diag(d_f) V_fᵀ`` (an orthogonal
    similarity: the spectrum is unchanged), so the pose part is ``diag(d)`` coupled to the core
    by ``W = B·blkdiag(V_f)``. Eliminating a coordinate ``j`` divides by ``d_j − σ`` and adds
    ``w_j w_jᵀ/(d_j − σ)`` to the Schur complement; relative to the core diagonal that is a
    growth of ``γ_j = Σ_c (w_cj² / A_cc) / |d_j − σ|`` (unit-free). When ``γ_j`` would exceed
    :attr:`max_growth` the coordinate is not eliminated but kept beside the core (pivoting), so
    the computed Schur complement stays within a small multiple of ``ε‖H‖`` of the exact one
    and its inertia, hence every count, is reliable even next to a near-singular pose block.
    Every solve uses ``O((C + r)² + C·b·F)`` memory, ``r`` the few kept coordinates."""

    #: largest relative Schur-complement growth ``γ_j`` accepted per eliminated coordinate
    max_growth: float = 100.0

    def __init__(self, H: ArrowheadSym):
        self.H = H
        F, b = H.D.shape[0], H.D.shape[1]
        self.b = b
        dv, V = np.linalg.eigh(H.D) if F else (np.zeros((0, b)), np.zeros((0, b, b)))
        self.V = V
        self.d = dv.ravel()
        C = H.A.shape[0]
        self.W = np.einsum("cfi,fij->cfj", H.B.reshape(C, F, b), V).reshape(C, b * F)
        acc = np.maximum(np.diag(H.A), _TINY)
        self.w2 = np.einsum("cj,cj->j", self.W / acc[:, None], self.W)   # Σ_c w_cj² / A_cc
        self.C = C
        self._counts: Dict[float, int] = {}

    #: absolute resolution: eigenvalues closer than this to each other or to 0 are not split
    #: further (set by :func:`spectrum_bounds` to 1e-3·ε·‖H‖, finer than a dense solver's error)
    atol: float = _TINY

    # --- S(σ) and the count ν(σ) -------------------------------------------------------
    def _kept(self, lo: float, hi: float) -> np.ndarray:
        """Coordinates whose elimination anywhere in ``[lo, hi]`` would grow the Schur
        complement by more than :attr:`max_growth` (including every ``d_j`` inside)."""
        dist = np.maximum(np.maximum(lo - self.d, self.d - hi), 0.0)
        return np.nonzero(self.w2 >= self.max_growth * dist)[0]

    def schur(self, sigma: float, kept: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(S, g)``: the Schur complement after eliminating every pose coordinate except
        ``kept``, which are appended after the core as ``[[S_c, W_k], [W_kᵀ, diag(d_k) − σ]]``;
        ``g = 1/(d − σ)`` with 0 at the kept coordinates."""
        with np.errstate(divide="ignore"):
            g = 1.0 / (self.d - sigma)
        g[kept] = 0.0
        S = self.H.A - sigma * np.eye(self.C)
        for j in range(0, self.W.shape[1], _CHUNK):
            Wj = self.W[:, j:j + _CHUNK]
            S -= (Wj * g[j:j + _CHUNK]) @ Wj.T
        S = 0.5 * (S + S.T)
        if len(kept):
            Wk = self.W[:, kept]
            S = np.block([[S, Wk], [Wk.T, np.diag(self.d[kept] - sigma)]])
        return S, g

    def count(self, sigma: float) -> int:
        """``ν(σ)``: the number of eigenvalues of ``H`` below ``σ`` (Sylvester + Haynsworth)."""
        sigma = float(sigma)
        if sigma in self._counts:
            return self._counts[sigma]
        kept = self._kept(sigma, sigma)
        S, _g = self.schur(sigma, kept)
        elim = np.ones(len(self.d), bool)
        elim[kept] = False
        nu = (int(np.count_nonzero(self.d[elim] < sigma))
              + int(np.count_nonzero(np.linalg.eigvalsh(S) < 0)))
        self._counts[sigma] = nu
        return nu

    def _bracket(self, k: int) -> Tuple[float, float]:
        """The tightest known ``lo < hi`` with ``ν(lo) <= k < ν(hi)``."""
        lo = max((s for s, c in self._counts.items() if c <= k), default=None)
        hi = min((s for s, c in self._counts.items() if c > k), default=None)
        if lo is None or hi is None:
            raise ValueError("spectrum bounds not counted yet")
        return lo, hi

    @staticmethod
    def _mid(lo: float, hi: float) -> float:
        if lo > 0:
            return float(np.sqrt(lo) * np.sqrt(hi))          # geometric: tiny eigenvalues
        if hi < 0:
            return -float(np.sqrt(-lo) * np.sqrt(-hi))
        return 0.5 * (lo + hi)

    def init_bounds(self, lo: float, hi: float) -> None:
        """Record ``ν(lo) = 0`` and ``ν(hi) = n`` (checked) for valid outer bounds."""
        if self.count(lo) != 0 or self.count(hi) != self.H.n:
            raise ValueError("bounds do not enclose the spectrum")
        if lo < 0 < hi:
            self.count(0.0)
            self.count(_TINY)

    def _done(self, lo: float, hi: float) -> bool:
        return hi - lo <= max(4 * _EPS * max(abs(lo), abs(hi)), self.atol)

    def kth(self, k: int, *, newton: bool = True) -> float:
        """The ``k``-th smallest eigenvalue (0-based) to full double precision: bisection on
        the count until ``λ_k`` is alone in its bracket, then safeguarded Newton."""
        lo, hi = self._bracket(k)
        for _ in range(4000):
            if self._done(lo, hi):
                return 0.5 * (lo + hi)
            if newton and self.count(lo) == k and self.count(hi) == k + 1:
                return self._newton(k, lo, hi)
            mid = self._mid(lo, hi)
            if not lo < mid < hi:
                return 0.5 * (lo + hi)
            if self.count(mid) > k:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)                                # pragma: no cover

    def _newton(self, k: int, lo: float, hi: float) -> float:
        """Safeguarded Newton inside a bracket holding exactly ``λ_k``. With the kept set fixed
        for the whole bracket, the eliminated poles all lie outside it, so the ordered
        eigenvalues of ``S(σ)`` are smooth and decreasing there and the ``m``-th one crosses
        zero exactly at ``λ_k``."""
        kept = self._kept(lo, hi)
        elim = np.ones(len(self.d), bool)
        elim[kept] = False
        m = k - int(np.count_nonzero(self.d[elim] < lo))
        sigma = self._mid(lo, hi)
        for _ in range(100):
            S, g = self.schur(sigma, kept)
            mu, U = np.linalg.eigh(S)
            if mu[m] > 0:
                lo = sigma
            elif mu[m] < 0:
                hi = sigma
            else:
                return sigma
            t = (self.W.T @ U[:self.C, m]) * g
            dmu = -1.0 - float(t @ t)                          # < 0: S(σ) decreases
            step = sigma - mu[m] / dmu
            if self._done(lo, hi):
                return 0.5 * (lo + hi)
            if not lo < step < hi:
                step = self._mid(lo, hi)
            if abs(step - sigma) <= 2 * _EPS * max(abs(sigma), _TINY):
                return step
            sigma = step
        return 0.5 * (lo + hi)                                # pragma: no cover

    def eigvecs_at(self, lam: float, mult: int) -> np.ndarray:
        """``mult`` orthonormal eigenvectors of ``H`` (original ordering, as columns) for the
        eigenvalue ``λ`` of multiplicity ``mult``: null vectors ``[x_c; x_k]`` of the pivoted
        Schur complement at ``λ``, with the eliminated pose coordinates recovered as
        ``−(d_j − λ)⁻¹ w_jᵀ x_c`` and rotated back by ``V``."""
        kept = self._kept(lam, lam)
        S, g = self.schur(lam, kept)
        mu, U = np.linalg.eigh(S)
        idx = np.argsort(np.abs(mu))[:mult]
        Xc, Xk = U[:self.C, idx], U[self.C:, idx]
        Y = -(self.W.T @ Xc) * g[:, None]                      # pose part in the V basis
        Y[kept] = Xk
        b = self.b
        F = len(self.d) // b if b else 0
        Xd = np.einsum("fij,fjk->fik", self.V, Y.reshape(F, b, -1)).reshape(b * F, -1)
        X = np.zeros((self.H.n, mult))
        X[self.H.core] = Xc
        X[self.H.blocks] = Xd
        Q, _ = np.linalg.qr(X)
        return Q


def lowest_eigenvalues(sp: ArrowheadSpectrum, below: float) -> np.ndarray:
    """All eigenvalues ``< below``, ascending, each to full precision."""
    n_low = sp.count(below)
    return np.array([sp.kth(k) for k in range(n_low)])


def spectrum_bounds(H: ArrowheadSym, sp: Optional[ArrowheadSpectrum] = None
                    ) -> ArrowheadSpectrum:
    """An :class:`ArrowheadSpectrum` with its outer counts recorded from Gershgorin bounds
    (widened slightly so rounding cannot put an eigenvalue outside)."""
    sp = sp or ArrowheadSpectrum(H)
    lo, hi = H.gershgorin()
    norm = max(abs(lo), abs(hi), _TINY)
    pad = 16 * _EPS * max(norm, 1.0) * max(H.n, 1)
    sp.atol = 1e-3 * _EPS * norm
    sp.init_bounds(lo - pad - _TINY, hi + pad + _TINY)
    return sp
