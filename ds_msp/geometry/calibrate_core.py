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

from ..core.lie import hat_batch, so3_exp
from ..core.optimize import OptResult, gnc_tls_schur_solve, schur_lm


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
                  max_iter: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, OptResult]:
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
    total = 2 * sum(sizes)
    masks = [np.asarray(v, bool) for v in visibility_list]
    lb_i, ub_i = cls.param_bounds()
    state0 = (np.clip(np.asarray(params0, float), lb_i, ub_i).copy(),
              np.asarray(R0, float).copy(), np.asarray(t0, float).copy())

    def residual(state):
        """Flat ``(2 * sum(sizes),)`` pixel residual ``uv_predicted - uv_observed``, image-major."""
        params, Rb, t = state
        m = cls.from_params(params)
        out = np.zeros((total,))
        row = 0
        for i, (Xw, uv) in enumerate(zip(X_world_list, keypoints_list)):
            N = sizes[i]
            Xc = (Rb[i] @ Xw.T).T + t[i]
            uvp, valid = m.project(Xc)
            mask = masks[i] & valid
            diff = np.zeros_like(uv, dtype=np.float64)
            diff[mask] = uvp[mask] - uv[mask]
            out[row:row + 2 * N] = diff.ravel()
            row += 2 * N
        return out

    def linearize(state):
        """Per-image residual + split Jacobian (shared intrinsics A_i, local pose B_i).

        Feeding the blocks separately lets the solver Schur-complement out the (block-
        diagonal) per-image poses, so the work scales linearly in image count rather than
        cubically in the full ``P + 6·n_img`` dimension.
        """
        params, Rb, t = state
        m = cls.from_params(params)
        r_list, A_list, B_list = [], [], []
        for i, (Xw, uv) in enumerate(zip(X_world_list, keypoints_list)):
            N = sizes[i]
            Xc = (Rb[i] @ Xw.T).T + t[i]
            uvp, J_point, J_param, valid = m.project_jacobian(Xc)
            mask = (masks[i] & valid)
            r_i = np.zeros((N, 2))
            r_i[mask] = uvp[mask] - uv[mask]
            mask3 = mask[:, None, None].astype(np.float64)
            # δω linearized at 0 ⇒ J_r = I, so ∂Xc/∂δω = -R[Xw]_×; ∂Xc/∂δt = I.
            dXc_dw = -np.einsum('ij,njk->nik', Rb[i], hat_batch(Xw))
            J_rvec = np.einsum('nij,njc->nic', J_point, dXc_dw)
            J_ext = np.concatenate([J_rvec, J_point], axis=-1) * mask3
            r_list.append(r_i.ravel())
            A_list.append((J_param * mask3).reshape(2 * N, P))
            B_list.append(J_ext.reshape(2 * N, 6))
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

    if noise_bound is not None:
        out = gnc_tls_schur_solve(state0, residual, linearize, retract,
                                  noise_bound=3.03 * float(noise_bound),   # 2-DoF 99% χ² band
                                  n_groups=n_img, shared_dim=P, local_dim=6, block=2,
                                  inner_max_iter=max_iter)
    else:
        out = schur_lm(state0, residual, linearize, retract,
                       n_groups=n_img, shared_dim=P, local_dim=6, block=2,
                       max_iter=max_iter, robust_kernel=kernel, robust_scale=scale,
                       gnc_start=gnc_start, gnc_iters=gnc_iters)
    params, Rb, t = out.state
    return params, Rb, t, out
