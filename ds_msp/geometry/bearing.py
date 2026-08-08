"""Shared full-sphere bearing residuals.

The chordal residual compares unit directions directly.  Unlike a two-component
tangent-plane projection, its norm cannot mistake the antipodal ray for a perfect
fit: ``||d - f||² = 2 (1 - cos(theta))`` grows monotonically from zero at 0 degrees
to four at 180 degrees.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def projection_bearing_whiteners(
    observed: np.ndarray,
    projection_jacobian: np.ndarray,
    projection_valid: Optional[np.ndarray] = None,
    *,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Build a local pixel metric for full-sphere chordal bearing residuals.

    For an observed unit ray ``f`` and the camera contract's analytic point Jacobian
    ``J_pi = d(pixel) / d(point)`` evaluated at ``f``, a small tangent bearing error
    ``delta`` produces the pixel displacement ``J_pi delta``. This function returns a
    symmetric ``W`` such that, to first order on the sphere,

    ``||W delta||² = ||J_pi delta||²``.

    The projective Jacobian has the radial ray direction in its null space. A pure tangent
    metric would therefore also assign zero cost to the antipodal chord ``-f - f``. To retain
    the full-sphere signed-ray semantics, the metric is completed along ``f`` with
    ``s_rad = sqrt(s_max s_min)``:

    ``W.T W = J_pi.T J_pi + s_rad² f f.T``.

    This radial completion does not change first-order tangent/pixel errors, but keeps the
    antipode far from zero. It lets a caller express a GNC-TLS noise bound in actual local
    pixel units without inserting concrete camera equations into the optimizer.

    Parameters
    ----------
    observed : (N, 3) ndarray
        Observed bearings; normalized internally.
    projection_jacobian : (N, 2, 3) ndarray
        Analytic ``d(pixel)/d(point)`` from ``CameraModel.project_jacobian(observed)``.
    projection_valid : (N,) ndarray of bool, optional
        Camera-projection validity at each ray.
    eps : float, default 1e-12
        Minimum usable bearing norm and tangent singular value.

    Returns
    -------
    whiteners : (N, 3, 3) ndarray
        Symmetric square-root pixel metrics. Invalid rows are zero.
    valid : (N,) ndarray of bool
        Rows with finite bearings and a rank-two local projection Jacobian.
    """
    f = np.asarray(observed, float)
    J = np.asarray(projection_jacobian, float)
    if f.ndim != 2 or f.shape[1:] != (3,) or J.shape != (len(f), 2, 3):
        raise ValueError("observed must be (N, 3) and projection_jacobian (N, 2, 3)")
    if projection_valid is None:
        valid = np.ones(len(f), dtype=bool)
    else:
        valid = np.asarray(projection_valid, bool).copy()
        if valid.shape != (len(f),):
            raise ValueError("projection_valid must have shape (N,)")

    norm = np.linalg.norm(f, axis=1)
    valid &= (
        np.isfinite(f).all(axis=1)
        & np.isfinite(J).all(axis=(1, 2))
        & np.isfinite(norm)
        & (norm > eps)
    )
    f_unit = np.zeros_like(f)
    f_unit[valid] = f[valid] / norm[valid, None]
    whiteners = np.zeros((len(f), 3, 3), dtype=float)
    for i in np.flatnonzero(valid):
        singular = np.linalg.svd(J[i], compute_uv=False)
        if not np.isfinite(singular).all() or singular[-1] <= eps:
            valid[i] = False
            continue
        radial_scale = float(np.sqrt(singular[0] * singular[-1]))
        metric = J[i].T @ J[i] + radial_scale ** 2 * np.outer(f_unit[i], f_unit[i])
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (metric + metric.T))
        if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= eps ** 2:
            valid[i] = False
            continue
        root = (eigenvectors * np.sqrt(eigenvalues)[None, :]) @ eigenvectors.T
        whiteners[i] = 0.5 * (root + root.T)
    return whiteners, valid


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
