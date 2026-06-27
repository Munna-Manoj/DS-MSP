"""Angular reprojection error and two-view bundle refinement (Tier-1).

The eight-point + cheirality estimate is *algebraic*; a nonlinear refinement that
minimizes the **geometric** reprojection error tightens it. For a wide-FOV camera that error must
be measured as an **angle on the sphere**, not a pixel distance: pixel error is anisotropic and
ill-defined past 90°, whereas the angle between the observed ray and the predicted ray is the
honest, model-free residual everywhere. This refines relative pose **and** structure together by
least-squares on the tangent-plane residual.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core.lie import hat, so3_exp
from ..core.optimize import lm_solve
from .ransac import ransac_relative_pose
from .two_view import _as_rays, triangulate_rays


def _tangent_basis(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the tangent plane at each unit ray ``f`` (N,3)."""
    helper = np.tile(np.array([0.0, 1.0, 0.0]), (f.shape[0], 1))
    near = np.abs(f @ np.array([0.0, 1.0, 0.0])) > 0.9
    helper[near] = np.array([1.0, 0.0, 0.0])
    e_u = np.cross(helper, f)
    e_u /= np.linalg.norm(e_u, axis=1, keepdims=True)
    e_v = np.cross(f, e_u)
    return e_u, e_v


def angular_reprojection_error(f1: np.ndarray, f2: np.ndarray,
                               R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Per-point reprojection error in **degrees**: max of the two view angles.

    View 1 predicts direction ``X``; view 2 predicts ``R X + t``. Each is compared (as an angle)
    to the observed ray.
    """
    f1, f2 = _as_rays(f1), _as_rays(f2)
    R = np.asarray(R, float)
    t = np.asarray(t, float).reshape(3)
    d1 = X / np.linalg.norm(X, axis=1, keepdims=True)
    d2 = X @ R.T + t
    d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
    a1 = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", d1, f1), -1, 1)))
    a2 = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", d2, f2), -1, 1)))
    return np.maximum(a1, a2)


def refine_two_view(f1: np.ndarray, f2: np.ndarray,
                    R0: np.ndarray, t0: np.ndarray, X0: np.ndarray, *,
                    max_nfev: int = 100, robust_kernel: str = "none",
                    robust_scale: float | str = 1.0
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nonlinear refinement of ``(R, t, X)`` minimizing the tangent-plane angular residual.

    **Manifold-correct, in-house solver.** Driven by :func:`ds_msp.core.optimize.lm_solve`,
    which *re-bases* the rotation every accepted step (``R ← R · exp([δω]_×)``, ``δω`` reset to 0)
    instead of letting a fixed-base perturbation drift toward the ``‖δω‖ = π`` singularity — so it
    stays stable at large inter-frame rotation where a flat axis-angle / fixed-base solve wobbles.
    The Jacobian is **analytic** (the angular residual differentiated through the tangent basis),
    replacing the old finite-difference ``scipy`` path that stalled at large rotation.

    Camera 1 is fixed at the identity (reference frame) and ``t`` is kept unit-length, pinning the
    7-DOF similarity gauge (the angular error is scale-invariant). The translation gauge makes the
    ``δt`` block rank-deficient by design; the solver's damped-Cholesky floor absorbs it.

    ``robust_kernel`` / ``robust_scale`` optionally turn on IRLS down-weighting of mismatched
    correspondences (e.g. ``"cauchy"``, ``"auto"``); the default reproduces plain L2. Returns the
    refined ``(R, t, X)``.
    """
    f1, f2 = _as_rays(f1), _as_rays(f2)
    n = f1.shape[0]
    eu1, ev1 = _tangent_basis(f1)
    eu2, ev2 = _tangent_basis(f2)
    R0 = np.asarray(R0, float)
    t0 = np.asarray(t0, float).reshape(3)
    t0 = t0 / np.linalg.norm(t0)
    X0 = np.asarray(X0, float)

    def residual(state):
        R, t, X = state
        d1 = X / np.linalg.norm(X, axis=1, keepdims=True)
        d2 = X @ R.T + t
        d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
        r1 = np.stack([np.einsum("ij,ij->i", d1, eu1), np.einsum("ij,ij->i", d1, ev1)], 1)
        r2 = np.stack([np.einsum("ij,ij->i", d2, eu2), np.einsum("ij,ij->i", d2, ev2)], 1)
        return np.concatenate([r1.ravel(), r2.ravel()])

    def jacobian(state):
        R, t, X = state
        # View 1 (camera at identity) sees only X; view 2 sees R·exp(δω)·X + normalize(t+δt).
        J = np.zeros((4 * n, 6 + 3 * n))
        y1 = X
        n1 = np.linalg.norm(y1, axis=1, keepdims=True)
        d1 = y1 / n1
        y2 = X @ R.T + t
        n2 = np.linalg.norm(y2, axis=1, keepdims=True)
        d2 = y2 / n2
        Pt = np.eye(3) - np.outer(t, t)                  # ∂ normalize(t+δt)/∂δt at δt=0
        for i in range(n):
            E1 = np.stack([eu1[i], ev1[i]])              # (2,3)
            P1 = (np.eye(3) - np.outer(d1[i], d1[i])) / n1[i, 0]
            J[2 * i:2 * i + 2, 6 + 3 * i:9 + 3 * i] = E1 @ P1          # ∂r1/∂X_i

            E2 = np.stack([eu2[i], ev2[i]])
            P2 = (np.eye(3) - np.outer(d2[i], d2[i])) / n2[i, 0]
            G = E2 @ P2                                  # (2,3): ∂r2/∂y2
            row = 2 * n + 2 * i
            J[row:row + 2, 0:3] = G @ (-R @ hat(X[i]))   # ∂r2/∂δω
            J[row:row + 2, 3:6] = G @ Pt                 # ∂r2/∂δt
            J[row:row + 2, 6 + 3 * i:9 + 3 * i] = G @ R  # ∂r2/∂X_i
        return J

    def retract(state, delta):
        R, t, X = state
        R = R @ so3_exp(delta[:3])
        t = t + delta[3:6]
        t = t / np.linalg.norm(t)
        X = X + delta[6:].reshape(n, 3)
        return (R, t, X)

    out = lm_solve((R0, t0, X0), residual, jacobian, retract, block=2,
                   max_iter=max_nfev, robust_kernel=robust_kernel,
                   robust_scale=robust_scale)
    return out.state


def estimate_relative_pose(f1: np.ndarray, f2: np.ndarray, *,
                           threshold: float = 0.005, max_iters: int = 1000, seed: int = 0,
                           refine: bool = True, robust_kernel: str = "none",
                           robust_scale: float | str = "auto"):
    """End-to-end **robust** two-view relative pose from ray correspondences.

    Ties the RANSAC consensus and angular-BA pieces into one call: RANSAC consensus over the eight-point gives an
    outlier-free inlier set and a well-conditioned initial ``(R₀, t₀)`` (a far better seed than a
    single least-squares eight-point on contaminated data, especially at large rotation where one
    bad ray skews the essential matrix); the inliers are triangulated and handed to the
    manifold-correct :func:`refine_two_view` for a final geometric (angular) bundle adjustment.

    ``robust_kernel`` lets the refinement *additionally* down-weight any soft mismatches that
    slipped under the RANSAC threshold. Returns ``(R, t, X, inliers)`` — pose, triangulated points
    (camera-1 frame, inliers only), and the boolean inlier mask over the input correspondences.
    """
    f1, f2 = _as_rays(f1), _as_rays(f2)
    R0, t0, inliers = ransac_relative_pose(f1, f2, threshold=threshold,
                                           max_iters=max_iters, seed=seed)
    fin1, fin2 = f1[inliers], f2[inliers]
    X0, _, _ = triangulate_rays(fin1, fin2, R0, t0)
    if refine:
        R, t, X = refine_two_view(fin1, fin2, R0, t0, X0,
                                  robust_kernel=robust_kernel, robust_scale=robust_scale)
    else:
        R, t, X = R0, t0, X0
    return R, t, X, inliers
