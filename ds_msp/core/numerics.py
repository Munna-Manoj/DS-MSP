r"""Numerical guard rails shared by every layer (pure NumPy).

NumPy's dense linear algebra fails in three different ways, and only one of them is loud:

* ``inv`` / ``solve`` raise only on an *exactly* zero pivot. A near-singular matrix returns a
  large, meaningless answer, and a NaN input returns NaN, both silently.
* ``eigh`` / ``eigvalsh`` / ``svd`` raise ``LinAlgError("... did not converge")`` on any NaN or
  inf entry, with no hint of which matrix or why.
* Every comparison with NaN is ``False``, so a gate such as ``if sigma > limit`` lets a NaN
  through, and Python's ``max(nan, x)`` depends on argument order.

The helpers here turn all three into one explicit, attributable :class:`NumericalError`.
Rank and conditioning are judged *relative* to the largest eigenvalue after Jacobi
equilibration (the NumPy ``matrix_rank`` rule), never against an absolute ``1e-12``, so the
verdict does not change with the units of the problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


class NumericalError(ValueError):
    """A numerical precondition failed: a non-finite input, or a matrix that is not
    (numerically) symmetric positive definite.

    ``where`` names the calling function, ``what`` the offending array. ``cond`` and
    ``rank`` are filled in when the failure is a conditioning one.
    """

    def __init__(self, where: str, what: str, reason: str, *,
                 cond: Optional[float] = None, rank: Optional[int] = None,
                 size: Optional[int] = None) -> None:
        self.where, self.what, self.reason = where, what, reason
        self.cond, self.rank, self.size = cond, rank, size
        detail = ""
        if rank is not None and size is not None:
            detail += f" (numerical rank {rank} of {size}"
            detail += f", cond {cond:.3g})" if cond is not None else ")"
        super().__init__(f"{where}: {what} {reason}{detail}")


@dataclass(frozen=True)
class Conditioning:
    """Conditioning of a symmetric matrix after Jacobi equilibration
    ``D^{-1/2} A D^{-1/2}`` with ``D = diag(A)``.

    ``cond`` is the ratio of the largest to the smallest eigenvalue (``inf`` when the
    smallest is not positive). ``rank`` counts eigenvalues above ``n·eps·λ_max``.
    """
    cond: float
    rank: int
    size: int

    @property
    def full_rank(self) -> bool:
        return self.rank == self.size


def require_finite(where: str, **arrays: object) -> None:
    """Raise :class:`NumericalError` if any named array holds a NaN or inf.

    >>> require_finite("demo", x=np.array([1.0, 2.0]))         # passes silently
    """
    for name, a in arrays.items():
        if not np.isfinite(np.asarray(a, dtype=float)).all():
            raise NumericalError(where, name, "is not finite")


def _symmetric(where: str, what: str, A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise NumericalError(where, what, f"is not square (shape {A.shape})")
    require_finite(where, **{what: A})
    scale = float(np.max(np.abs(A))) if A.size else 0.0
    if A.size and float(np.max(np.abs(A - A.T))) > 1e-9 * scale:
        raise NumericalError(where, what, "is not symmetric")
    return 0.5 * (A + A.T)


def _equilibrated_eigh(where: str, what: str, A: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Conditioning]:
    """``(d, w, V, info)``: the Jacobi scaling ``d = diag(A)^{-1/2}`` and the eigenpairs of the
    equilibrated matrix ``d A d``."""
    A = _symmetric(where, what, A)
    n = A.shape[0]
    diag = np.diag(A)
    if np.any(diag <= 0.0):
        return (np.zeros(n), np.zeros(n), np.eye(n),
                Conditioning(cond=float("inf"), rank=int(np.sum(diag > 0.0)), size=n))
    d = 1.0 / np.sqrt(diag)
    w, V = np.linalg.eigh(d[:, None] * A * d[None, :])
    tol = n * np.finfo(float).eps * max(float(w[-1]), 0.0)
    rank = int(np.sum(w > tol))
    cond = float(w[-1] / w[0]) if w[0] > 0.0 else float("inf")
    return d, w, V, Conditioning(cond=cond, rank=rank, size=n)


def conditioning(A: np.ndarray, *, where: str = "conditioning", what: str = "A") -> Conditioning:
    """Equilibrated condition number and numerical rank of a finite symmetric matrix."""
    return _equilibrated_eigh(where, what, A)[3]


def spd_inverse(A: np.ndarray, *, where: str = "spd_inverse", what: str = "A"
                ) -> Tuple[np.ndarray, Conditioning]:
    """Inverse of a symmetric positive-definite matrix, with its :class:`Conditioning`.

    Raises :class:`NumericalError` when ``A`` is non-finite, not symmetric, or numerically
    rank deficient, instead of returning the meaningless inverse ``np.linalg.inv`` would.
    """
    d, w, V, info = _equilibrated_eigh(where, what, A)
    if not info.full_rank or w[0] <= 0.0:
        raise NumericalError(where, what, "is not positive definite",
                             cond=info.cond, rank=info.rank, size=info.size)
    inv = d[:, None] * ((V / w) @ V.T) * d[None, :]
    return 0.5 * (inv + inv.T), info


def spd_solve(A: np.ndarray, b: np.ndarray, *, where: str = "spd_solve", what: str = "A"
              ) -> Tuple[np.ndarray, Conditioning]:
    """Solve ``A x = b`` for symmetric positive-definite ``A`` (see :func:`spd_inverse`)."""
    require_finite(where, b=b)
    d, w, V, info = _equilibrated_eigh(where, what, A)
    if not info.full_rank or w[0] <= 0.0:
        raise NumericalError(where, what, "is not positive definite",
                             cond=info.cond, rank=info.rank, size=info.size)
    b = np.asarray(b, dtype=float)
    y = V.T @ (d[:, None] * b if b.ndim == 2 else d * b)
    y = (y / w[:, None]) if b.ndim == 2 else (y / w)
    x = V @ y
    return (d[:, None] * x if b.ndim == 2 else d * x), info


def safe_eigh(A: np.ndarray, *, where: str = "safe_eigh", what: str = "A"
              ) -> Tuple[np.ndarray, np.ndarray]:
    """``np.linalg.eigh`` that names the matrix on failure: a non-finite or non-symmetric
    input raises :class:`NumericalError` instead of an anonymous ``LinAlgError``."""
    A = _symmetric(where, what, A)
    try:
        return np.linalg.eigh(A)
    except np.linalg.LinAlgError as exc:                 # pragma: no cover - finite input
        raise NumericalError(where, what, f"eigendecomposition failed ({exc})") from exc


def psd_eigvalsh(A: np.ndarray, *, where: str = "psd_eigvalsh", what: str = "A",
                 rtol: Optional[float] = None) -> np.ndarray:
    """Eigenvalues of a covariance-like matrix that must be positive semi-definite.

    Eigenvalues within roundoff of zero (``|λ| ≤ rtol·λ_max``, default ``n·eps``) are returned
    as exactly 0. A clearly negative eigenvalue raises :class:`NumericalError`: an indefinite
    covariance is a defect to report, not a variance to clamp to zero.
    """
    w, _ = safe_eigh(A, where=where, what=what)
    n = w.size
    tol = (n * np.finfo(float).eps if rtol is None else rtol) * max(float(np.abs(w).max(initial=0.0)), 0.0)
    if n and w[0] < -tol:
        raise NumericalError(where, what, f"is indefinite (smallest eigenvalue {w[0]:.3g}, "
                                          f"largest {w[-1]:.3g})")
    return np.where(np.abs(w) <= tol, 0.0, w)


def se3_inverse(T: np.ndarray, *, where: str = "se3_inverse", what: str = "T") -> np.ndarray:
    """Closed-form inverse of a rigid 4x4 transform, ``[Rᵀ, -Rᵀt; 0, 1]``.

    Exact for a rotation block (no LU), and validated: a non-finite entry, a bottom row other
    than ``[0, 0, 0, 1]``, or a rotation block more than ``1e-6`` from orthonormal raises
    :class:`NumericalError`.
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise NumericalError(where, what, f"is not 4x4 (shape {T.shape})")
    require_finite(where, **{what: T})
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-12):
        raise NumericalError(where, what, "bottom row is not [0, 0, 0, 1]")
    R, t = T[:3, :3], T[:3, 3]
    if float(np.max(np.abs(R.T @ R - np.eye(3)))) > 1e-6:
        raise NumericalError(where, what, "rotation block is not orthonormal")
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def max_or_nan(values: Iterable[float]) -> float:
    """Largest value, or NaN if any value is NaN (or there are none).

    Unlike Python's ``max``, the answer does not depend on the order of the values: a NaN
    anywhere always wins, so a later finite-ness check cannot be bypassed.
    """
    a = np.asarray(list(values), dtype=float)
    return float(np.max(a)) if a.size else float("nan")
