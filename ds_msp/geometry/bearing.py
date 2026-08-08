"""Shared full-sphere bearing residuals.

The chordal residual compares unit directions directly.  Unlike a two-component
tangent-plane projection, its norm cannot mistake the antipodal ray for a perfect
fit: ``||d - f||² = 2 (1 - cos(theta))`` grows monotonically from zero at 0 degrees
to four at 180 degrees.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def chordal_bearing_residual_jacobian(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``normalize(predicted) - normalize(observed)`` and its Jacobian.

    Parameters
    ----------
    predicted, observed : (N, 3) ndarray
        Predicted vectors and observed bearings.  Both are normalized internally.
    eps : float, default 1e-12
        Vectors with norm at most ``eps`` are invalid.

    Returns
    -------
    residual : (N, 3) ndarray
        Chordal bearing residual.  Invalid rows are zero.
    jacobian : (N, 3, 3) ndarray
        Derivative with respect to ``predicted``.  Invalid rows are zero.
    valid : (N,) ndarray of bool
        Rows for which both input vectors have usable norm.

    Notes
    -----
    For ``d = y / ||y||``, ``d(d)/d(y) = (I - d d.T) / ||y||``.  The
    observed bearing is constant, so this is also the residual Jacobian.
    """
    y = np.asarray(predicted, float)
    f = np.asarray(observed, float)
    if y.ndim != 2 or y.shape[1:] != (3,) or f.shape != y.shape:
        raise ValueError("predicted and observed must have matching shape (N, 3)")

    y_norm = np.linalg.norm(y, axis=1)
    f_norm = np.linalg.norm(f, axis=1)
    valid = (
        np.isfinite(y).all(axis=1)
        & np.isfinite(f).all(axis=1)
        & np.isfinite(y_norm)
        & np.isfinite(f_norm)
        & (y_norm > eps)
        & (f_norm > eps)
    )

    d = np.zeros_like(y)
    f_unit = np.zeros_like(f)
    d[valid] = y[valid] / y_norm[valid, None]
    f_unit[valid] = f[valid] / f_norm[valid, None]

    residual = np.zeros_like(y)
    residual[valid] = d[valid] - f_unit[valid]

    jacobian = np.zeros((len(y), 3, 3), dtype=float)
    eye = np.eye(3)
    jacobian[valid] = (
        eye[None] - np.einsum("ni,nj->nij", d[valid], d[valid])
    ) / y_norm[valid, None, None]
    return residual, jacobian, valid
