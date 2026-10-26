"""Model-agnostic single-camera bundle-adjustment driver (pure NumPy).

The shared BA core consumed by both single-camera calibration (``calib.bundle``) and the
rig per-camera intrinsics front-end (``rig``). Given an initial parameter vector and
**already-seeded** per-image poses, it jointly refines intrinsics + extrinsics by manifold
Levenberg-Marquardt with Schur-complemented per-image poses, using the model's analytic
projection Jacobian. Pose seeding and result formatting live in the caller; this layer is
geometry-only (depends on ``core`` — no cv2, no concrete models).

Manifold state: each rotation is kept as a base matrix re-based every accepted step by the
solver (``R ← R·exp([δω]_×)``, ``δω`` reset to 0). Because ``δω`` is linearized at 0 the
retraction Jacobian ``J_r(0) = I`` drops out — the extrinsics Jacobian is the cheap
``-R[Xw]_×`` and ``δω`` never drifts toward the ``‖r‖=π`` singularity.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from ..core.covariance import pixel_whiteners
from ..core.lie import hat_batch, so3_exp
from ..core.optimize import OptResult, box_active, gnc_tls_schur_solve, schur_lm


def bundle_adjust(model_cls,
                  params0: np.ndarray,
                  R0: np.ndarray,
                  t0: np.ndarray,
                  X_world_list: Sequence[np.ndarray],
                  keypoints_list: Sequence[np.ndarray],
                  visibility_list: Sequence[np.ndarray],
                  *, kernel: str = "none", scale: "float | str" = 1.0,
                  gnc_start: float = 0.0, gnc_iters: int = 0,
                  noise_bound: "float | None" = None,
                  max_iter: int = 200, pixel_covariances=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, OptResult]:
    """Refine ``model_cls`` intrinsics + per-image poses from seeded extrinsics.

    Parameters
    ----------
    model_cls : type[CameraModel]
        The camera-model class (the parameter vector is in ``model_cls.param_names`` order).
    params0 : (P,) ndarray
        Initial intrinsics (clipped into ``model_cls.param_bounds()``).
    R0, t0 : (n,3,3), (n,3) ndarray
        Seeded per-image rotation matrices and translations (object->camera).
    kernel, scale : robust IRLS kernel name + inlier scale (``"none"`` ⇒ plain L2).
    gnc_start, gnc_iters : graduated-non-convexity schedule (0 ⇒ off).
    noise_bound : expected per-corner reprojection σ in pixels (e.g. ``~0.3``). When set, run a
        median-free **GNC-TLS** solve (:func:`core.optimize.gnc_tls_schur_solve`) against the
        explicit ``barc2 = (3.03·σ)²`` inlier band instead of the MAD-capped kernel path — it
        rejects **past 50%** gross corner-detection outliers and returns a hard inlier set.
        ``kernel``/``scale``/``gnc_*`` are ignored when ``noise_bound`` is given.

    Returns
    -------
    tuple
        ``(params, Rb, t, OptResult)`` — refined intrinsics, refined base rotations,
        translations, and the raw solver result.
    """
    cls = model_cls
    P = len(cls.param_names)
    n_img = len(X_world_list)
    sizes = [len(X) for X in X_world_list]
    masks = [np.asarray(v, bool) for v in visibility_list]
    if pixel_covariances is None:
        pixel_covariances = [None] * n_img
    if len(pixel_covariances) != n_img:
        raise ValueError("pixel_covariances must have one entry per image")
    whiteners = [pixel_whiteners(c, n) for c, n in zip(pixel_covariances, sizes)]
    lb_i, ub_i = cls.param_bounds()
    state0 = (np.clip(np.asarray(params0, float), lb_i, ub_i).copy(),
              np.asarray(R0, float).copy(), np.asarray(t0, float).copy())

    # Every view's points flattened once (image-major, the residual order): each evaluation
    # is a single projection call over all points instead of a Python loop over views.
    img = np.repeat(np.arange(n_img), sizes).astype(int)
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    Xw_all = (np.concatenate([np.asarray(X, float) for X in X_world_list]) if n_img
              else np.zeros((0, 3)))
    uv_all = (np.concatenate([np.asarray(u, float) for u in keypoints_list]) if n_img
              else np.zeros((0, 2)))
    vis_all = np.concatenate(masks) if n_img else np.zeros(0, bool)
    W_all = None
    if any(w is not None for w in whiteners):
        W_all = np.concatenate([w if w is not None else np.tile(np.eye(2), (n, 1, 1))
                                for w, n in zip(whiteners, sizes)])
    hat_Xw = hat_batch(Xw_all)

    def residual(state):
        """Flat ``(2 * sum(sizes),)`` pixel residual ``uv_predicted - uv_observed``, image-major."""
        params, Rb, t = state
        m = cls.from_params(params)
        Xc = np.einsum('nij,nj->ni', Rb[img], Xw_all) + t[img]
        uvp, valid = m.project(Xc)
        mask = vis_all & np.asarray(valid, bool).ravel()
        diff = np.zeros_like(uv_all)
        diff[mask] = uvp[mask] - uv_all[mask]
        if W_all is not None:
            diff = np.einsum('nij,nj->ni', W_all, diff)
        return diff.ravel()

    def linearize(state):
        """Per-image residual + split Jacobian (shared intrinsics A_i, local pose B_i).

        Feeding the blocks separately lets the solver Schur-complement out the (block-
        diagonal) per-image poses, so the work scales linearly in image count rather than
        cubically in the full ``P + 6·n_img`` dimension.
        """
        params, Rb, t = state
        m = cls.from_params(params)
        R = Rb[img]
        Xc = np.einsum('nij,nj->ni', R, Xw_all) + t[img]
        uvp, J_point, J_param, valid = m.project_jacobian(Xc)
        mask = vis_all & np.asarray(valid, bool).ravel()
        r = np.zeros_like(uv_all)
        r[mask] = uvp[mask] - uv_all[mask]
        mask3 = mask[:, None, None].astype(np.float64)
        # δω linearized at 0 ⇒ J_r = I, so ∂Xc/∂δω = -R[Xw]_×; ∂Xc/∂δt = I.
        J_rvec = -np.einsum('nij,njk,nkl->nil', J_point, R, hat_Xw)
        J_ext = np.concatenate([J_rvec, J_point], axis=-1) * mask3
        J_intr = J_param * mask3
        if W_all is not None:
            r = np.einsum('nij,nj->ni', W_all, r)
            J_intr = np.einsum('nij,njk->nik', W_all, J_intr)
            J_ext = np.einsum('nij,njk->nik', W_all, J_ext)
        r_list, A_list, B_list = [], [], []
        for i in range(n_img):
            s0, s1 = starts[i], starts[i + 1]
            r_list.append(r[s0:s1].ravel())
            A_list.append(J_intr[s0:s1].reshape(2 * (s1 - s0), P))
            B_list.append(J_ext[s0:s1].reshape(2 * (s1 - s0), 6))
        return r_list, A_list, B_list

    def retract(state, d_shared, d_local):
        """Manifold update for ``schur_lm``/``gnc_tls_schur_solve``: intrinsics clipped to
        bounds, each per-image rotation updated by SO(3) exp, translation updated flat."""
        params, Rb, t = state
        params = np.clip(params + d_shared, lb_i, ub_i)          # keep intrinsics valid
        Rb, t = Rb.copy(), t.copy()
        for i in range(n_img):
            Rb[i] = Rb[i] @ so3_exp(d_local[i, :3])
            t[i] = t[i] + d_local[i, 3:]
        return (params, Rb, t)

    def bound_active(state, g):
        """Intrinsics held at a bound against the gradient (see ``box_active``)."""
        return box_active(state[0], lb_i, ub_i, g)

    if noise_bound is not None:
        out = gnc_tls_schur_solve(state0, residual, linearize, retract,
                                  noise_bound=3.03 * float(noise_bound),   # 2-DoF 99% χ² band
                                  n_groups=n_img, shared_dim=P, local_dim=6, block=2,
                                  inner_max_iter=max_iter)
    else:
        out = schur_lm(state0, residual, linearize, retract,
                       n_groups=n_img, shared_dim=P, local_dim=6, block=2,
                       max_iter=max_iter, robust_kernel=kernel, robust_scale=scale,
                       gnc_start=gnc_start, gnc_iters=gnc_iters, bound_active=bound_active)
    params, Rb, t = out.state
    return params, Rb, t, out
