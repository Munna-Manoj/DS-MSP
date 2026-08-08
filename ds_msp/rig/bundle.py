"""Staged global bundle adjustment for the rig.

Residual composition mirrors MC-Calib's ``UniversalReprojectionError``
(OptimizationCeres.h:681): ``X_cam = T_c_g @ T_g_o @ X_obj``, then ``model.project``.
The board-in-object poses are baked into ``Object3D.pts_3d`` (fixed here); the optimizer
refines the camera extrinsics ``T_c_g`` (ref camera held fixed), the per-frame object
poses ``T_g_o``, and — when ``fix_intrinsics=False`` — the per-camera intrinsics.

Pose retraction follows ``calib.bundle``: ``R <- R @ so3_exp(δω)``, ``t <- t + δt`` with
the local tangent ordered ``[δω(3), δt(3)]``, so the projection Jacobian chains cleanly
through ``-R[X]_×`` and ``I`` with no ``J_r`` term. Huber loss, δ=1.0 px
(CameraGroup.cpp:288). Solved densely via :func:`core.optimize.lm_solve` — the parameter
count is small at calibration scale (a handful of cameras, hundreds of object poses).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.covariance import clustered_sandwich_covariance
from ..core.lie import hat_batch as _skew_batch
from ..core.lie import so3_exp
from ..core.optimize import gnc_tls_schur_solve, gnc_tls_solve, lm_solve, schur_lm
from ..core.robust import auto_kernel_scale
from ..geometry.bearing import chordal_bearing_residual_jacobian
from .types import ObjectObs, RigState

Key = Tuple[int, int]


def _state_from_rig(rig: RigState) -> dict:
    return {
        "cam_R": {c: rig.T_c_g[c][:3, :3].copy() for c in rig.T_c_g},
        "cam_t": {c: rig.T_c_g[c][:3, 3].copy() for c in rig.T_c_g},
        "obj_R": {k: rig.object_poses[k][:3, :3].copy() for k in rig.object_poses},
        "obj_t": {k: rig.object_poses[k][:3, 3].copy() for k in rig.object_poses},
        "intr": {c: np.asarray(rig.cameras[c].params, float).copy() for c in rig.cameras},
    }


def _rig_from_state(rig: RigState, state: dict) -> RigState:
    out = copy.copy(rig)
    out.T_c_g = {}
    for c in state["cam_R"]:
        T = np.eye(4)
        T[:3, :3] = state["cam_R"][c]
        T[:3, 3] = state["cam_t"][c]
        out.T_c_g[c] = T
    out.object_poses = {}
    for k in state["obj_R"]:
        T = np.eye(4)
        T[:3, :3] = state["obj_R"][k]
        T[:3, 3] = state["obj_t"][k]
        out.object_poses[k] = T
    out.cameras = {c: type(rig.cameras[c]).from_params(state["intr"][c]) for c in rig.cameras}
    return out


def build_problem(rig: RigState, object_obs: List[ObjectObs], *,
                  fix_intrinsics: bool = True, fix_extrinsics: bool = False,
                  residual_mode: str = "pixel"):
    """Assemble the BA callbacks ``(state0, residual, jacobian, retract, K)`` for one
    pass, without solving. ``residual``/``jacobian``/``retract`` follow the
    :func:`core.optimize.lm_solve` contract; ``K`` is the tangent dimension. Exposed so
    tests can finite-difference-check the analytic Jacobian (the chain in §9.1 of the
    implementation doc).

    ``fix_extrinsics=True`` holds all camera poses fixed and refines only the per-frame
    object poses (the per-object intermediate refinement); combined with
    ``fix_intrinsics=True`` it is the pure object-pose stage."""
    ref_cam = rig.ref_cam_id
    cam_ids = [] if fix_extrinsics else [c for c in sorted(rig.cameras) if c != ref_cam]
    obj_keys = sorted(rig.object_poses)
    classes = {c: type(rig.cameras[c]) for c in rig.cameras}
    Pn = {c: len(classes[c].param_names) for c in rig.cameras}
    bounds = {c: classes[c].param_bounds() for c in rig.cameras}

    # tangent column layout
    col, cam_col, obj_col, intr_col = 0, {}, {}, {}
    for c in cam_ids:
        cam_col[c] = col
        col += 6
    for k in obj_keys:
        obj_col[k] = col
        col += 6
    if not fix_intrinsics:
        for c in sorted(rig.cameras):
            intr_col[c] = col
            col += Pn[c]
    K = col

    angular = residual_mode == "angular"
    if angular and not fix_intrinsics:
        raise ValueError("angular residual requires fix_intrinsics=True "
                         "(the observed bearing is intrinsics-dependent)")

    # precompute per-observation object points (board poses baked into pts_3d). For the angular
    # (bearing) residual, also precompute each corner's observed unit bearing f = unproject(uv)
    # — the model-agnostic ray the predicted ray is compared against.
    obs_data = []
    total_rows = 0
    for o in object_obs:
        if o.cam_id not in rig.cameras:
            continue
        if (o.object_id, o.frame_id) not in rig.object_poses:  # frame never got a pose
            continue
        Xo = rig.objects[o.object_id].pts_3d[o.point_rows]      # (N,3) object frame
        tb = None
        if angular:
            f, fv = rig.cameras[o.cam_id].unproject(o.pts_2d)
            f = np.asarray(f, float)
            tb = (f, np.asarray(fv).ravel().astype(bool) if fv is not None
                  else np.ones(len(f), bool))
        obs_data.append((o, Xo, tb))
        total_rows += (3 if angular else 2) * len(Xo)

    def _project_all(state):
        out = []
        for o, Xo, tb in obs_data:
            key = (o.object_id, o.frame_id)
            Xg = (state["obj_R"][key] @ Xo.T).T + state["obj_t"][key]
            Xc = (state["cam_R"][o.cam_id] @ Xg.T).T + state["cam_t"][o.cam_id]
            out.append((o, Xo, key, Xg, Xc, tb))
        return out

    def _ang_resid(Xc, tb):
        """Full-sphere chordal residual and ``d(residual)/d(Xc)`` (N,3)/(N,3,3)."""
        f, fv = tb
        r, Jp, valid = chordal_bearing_residual_jacobian(Xc, f)
        valid &= fv
        r[~valid] = 0.0
        Jp[~valid] = 0.0
        return r, Jp

    def residual(state):
        """Stacked reprojection (or angular) residual vector ``(total_rows,)`` for ``state``.

        Part of the :func:`core.optimize.lm_solve` callback contract; see
        :func:`build_problem`'s docstring for the state layout.
        """
        m_cache = {c: classes[c].from_params(state["intr"][c]) for c in rig.cameras}
        r = np.zeros(total_rows)
        row = 0
        for o, Xo, key, Xg, Xc, tb in _project_all(state):
            if angular:
                diff, _Jp = _ang_resid(Xc, tb)
            else:
                uv, valid = m_cache[o.cam_id].project(Xc)
                diff = np.zeros_like(o.pts_2d, float)
                diff[valid] = uv[valid] - o.pts_2d[valid]
            rows = diff.size
            r[row:row + rows] = diff.ravel()
            row += rows
        return r

    def jacobian(state):
        """Dense analytic Jacobian ``(total_rows, K)`` of :func:`residual` at ``state``.

        Column layout is ``[camera extrinsics (6 each, ref camera excluded), object
        poses (6 each), intrinsics (P_c each, only if ``fix_intrinsics=False``)]``,
        matching :func:`retract`'s tangent ordering ``[δω(3), δt(3)]`` per block.
        """
        m_cache = {c: classes[c].from_params(state["intr"][c]) for c in rig.cameras}
        J = np.zeros((total_rows, K))
        row = 0
        for o, Xo, key, Xg, Xc, tb in _project_all(state):
            N = len(Xo)
            cam = o.cam_id
            R_cam = state["cam_R"][cam]
            R_obj = state["obj_R"][key]
            if angular:
                # ∂(normalize(Xc)-f)/∂Xc; no intrinsic columns (intrinsics fixed).
                _r, Jp = _ang_resid(Xc, tb)                    # (N,3,3)
                J_param = None
            else:
                uv, J_point, J_param, valid = m_cache[cam].project_jacobian(Xc)
                mask = valid[:, None, None].astype(float)
                Jp = J_point * mask                              # (N,2,3)
            # object pose: dXc/dω = R_cam @ (-R_obj[Xo]_x); dXc/dt = R_cam
            dXc_dw_o = -np.einsum('ij,njk->nik', R_cam @ R_obj, _skew_batch(Xo))
            Jw_o = np.einsum('nij,njc->nic', Jp, dXc_dw_o)
            Jt_o = np.einsum('nij,jc->nic', Jp, R_cam)
            point_dim = 3 if angular else 2
            c0 = obj_col[key]
            J[row:row + point_dim * N, c0:c0 + 3] = Jw_o.reshape(point_dim * N, 3)
            J[row:row + point_dim * N, c0 + 3:c0 + 6] = Jt_o.reshape(point_dim * N, 3)
            # camera extrinsic (skip ref): dXc/dω = -R_cam[Xg]_x; dXc/dt = I
            if cam in cam_col:
                dXc_dw_c = -np.einsum('ij,njk->nik', R_cam, _skew_batch(Xg))
                Jw_c = np.einsum('nij,njc->nic', Jp, dXc_dw_c)
                Jt_c = Jp                                        # J_point @ I
                cc = cam_col[cam]
                J[row:row + point_dim * N, cc:cc + 3] = Jw_c.reshape(point_dim * N, 3)
                J[row:row + point_dim * N, cc + 3:cc + 6] = Jt_c.reshape(point_dim * N, 3)
            # intrinsics
            if not fix_intrinsics:
                ic = intr_col[cam]
                J[row:row + 2 * N, ic:ic + Pn[cam]] = (J_param * mask).reshape(2 * N, Pn[cam])
            row += point_dim * N
        return J

    def retract(state, d):
        """Apply a tangent-space update ``d`` (K,) to ``state``: SO(3) retraction
        ``R <- R @ so3_exp(δω)`` for every pose block, additive for translations and
        (bounds-clipped) intrinsics. Returns a new state dict; does not mutate ``state``.
        """
        s = {k: (v.copy() if isinstance(v, np.ndarray) else dict(v)) for k, v in state.items()}
        s["cam_R"], s["cam_t"] = dict(state["cam_R"]), dict(state["cam_t"])
        s["obj_R"], s["obj_t"] = dict(state["obj_R"]), dict(state["obj_t"])
        s["intr"] = dict(state["intr"])
        for c in cam_ids:
            o = cam_col[c]
            s["cam_R"][c] = state["cam_R"][c] @ so3_exp(d[o:o + 3])
            s["cam_t"][c] = state["cam_t"][c] + d[o + 3:o + 6]
        for k in obj_keys:
            o = obj_col[k]
            s["obj_R"][k] = state["obj_R"][k] @ so3_exp(d[o:o + 3])
            s["obj_t"][k] = state["obj_t"][k] + d[o + 3:o + 6]
        if not fix_intrinsics:
            for c in sorted(rig.cameras):
                ic = intr_col[c]
                lb, ub = bounds[c]
                s["intr"][c] = np.clip(state["intr"][c] + d[ic:ic + Pn[c]], lb, ub)
        return s

    return _state_from_rig(rig), residual, jacobian, retract, K


def _report_column_layout(rig: RigState, *, fix_intrinsics: bool, fix_extrinsics: bool):
    """The same tangent column layout :func:`build_problem` computes internally, exposed
    standalone for :func:`parameter_covariance` (which needs to label ``J``'s columns by
    camera, unlike the solver which only needs ``K``). Kept in sync with ``build_problem``'s
    own layout logic by construction (same rig, same flags -> same columns); duplicated
    rather than plumbed through the solver's hot path, matching this module's existing
    per-function re-derivation of ``classes``/``Pn``/``bounds`` (see :func:`build_schur_problem`).
    """
    ref_cam = rig.ref_cam_id
    cam_ids = [] if fix_extrinsics else [c for c in sorted(rig.cameras) if c != ref_cam]
    classes = {c: type(rig.cameras[c]) for c in rig.cameras}
    Pn = {c: len(classes[c].param_names) for c in rig.cameras}
    col, cam_col, intr_col = 0, {}, {}
    for c in cam_ids:
        cam_col[c] = col
        col += 6
    n_obj_cols = 6 * len(rig.object_poses)
    col += n_obj_cols
    if not fix_intrinsics:
        for c in sorted(rig.cameras):
            intr_col[c] = col
            col += Pn[c]
    return cam_col, intr_col, Pn


def parameter_covariance(rig: RigState, object_obs: List[ObjectObs], *,
                         fix_intrinsics: bool = True, fix_extrinsics: bool = False,
                         kernel: str = "cauchy", scale: Optional[float] = None,
                         small_cluster_correction: bool = True) -> Dict:
    r"""Parameter-uncertainty report for a converged rig fit: per-camera extrinsic/intrinsic
    std from the **frame-clustered** M-estimator sandwich covariance
    (:func:`ds_msp.core.covariance.clustered_sandwich_covariance`).

    **Why the clustered estimator, and only it.** A 200-refit cluster bootstrap of a real
    2-camera rig session (resampling its 33 board placements, refitting, and comparing each
    estimator's predicted std to the measured scatter) gave predicted/measured coverage:
    naive ``σ̂²(JᵀW̃J)⁻¹`` **0.147** (claims ~7x more certainty than reality), unclustered
    sandwich **0.275** (~3.6x), frame-clustered **1.136** (honest within 14%, on the
    conservative side). The under-coverage mechanism: corners from the same synchronized
    board placement (across all cameras that saw it that frame) share correlated board-pose
    noise, so per-corner independence assumptions overstate the information content.
    Clustering by frame — pooled across cameras, since the shared board-pose realization is
    the correlated quantity — is the fix. The disproven estimators are deliberately NOT
    included in this report's output (a knowingly-wrong error bar is worse than none); they
    remain available in :mod:`ds_msp.core.covariance` as test baselines.

    This evaluates the dense reprojection Jacobian/residual **once** at the given (already
    converged) ``rig`` — it does not re-solve. ``kernel``/``scale`` should match what the
    final BA pass used (default ``"cauchy"``, matching :func:`refine`'s stage-(c) global
    joint BA default); ``scale=None`` auto-estimates the MAD inlier scale from the final
    residuals, the same estimator ``robust_scale="auto"`` uses during solving.

    Returns a dict: ``{"n_clusters": G, "n_blocks": N, "K": ..., "kernel": ..., "scale": ...,
    "clustered": {cam_id: {"extrinsic_std": (6,) or None, "intrinsic_std": (P_c,) or None}}}``.
    Extrinsic std order is the tangent ``[δω(3) rad, δt(3) in the dataset's length unit]`` —
    **rotation first, radians**, matching :func:`build_problem`'s state layout; intrinsic std
    order matches the camera model's ``param_names``. ``extrinsic_std`` is ``None`` for the
    reference camera (gauge-fixed, not a free parameter) and (if ``fix_extrinsics``) for all
    cameras; ``intrinsic_std`` is ``None`` when ``fix_intrinsics``.

    Not yet wired into the HTML report (``report.py``) — that is tracked as a fast-follow;
    this function is the complete, tested computation a report layer can call.
    """
    state0, residual, jacobian, _retract, K = build_problem(
        rig, object_obs, fix_intrinsics=fix_intrinsics, fix_extrinsics=fix_extrinsics)
    r = np.asarray(residual(state0), float)
    J = np.asarray(jacobian(state0), float)

    cluster_id = []
    for o in object_obs:
        if o.cam_id not in rig.cameras:
            continue
        if (o.object_id, o.frame_id) not in rig.object_poses:
            continue
        cluster_id.extend([o.frame_id] * len(o.point_rows))
    cluster_id = np.asarray(cluster_id)
    if cluster_id.shape[0] != r.size // 2:
        raise ValueError(f"cluster_id has {cluster_id.shape[0]} blocks, expected "
                         f"{r.size // 2} (mismatched filtering vs build_problem)")

    if scale is None:
        bn = np.linalg.norm(r.reshape(-1, 2), axis=1)
        scale = auto_kernel_scale(bn, kernel) if kernel != "none" else 1.0

    cov_clu = clustered_sandwich_covariance(
        J, r, cluster_id, kernel=kernel, scale=scale,
        small_cluster_correction=small_cluster_correction)

    cam_col, intr_col, Pn = _report_column_layout(
        rig, fix_intrinsics=fix_intrinsics, fix_extrinsics=fix_extrinsics)

    def _cameras(cov: np.ndarray) -> Dict:
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        out = {}
        for c in sorted(rig.cameras):
            ext = std[cam_col[c]:cam_col[c] + 6] if c in cam_col else None
            intr = std[intr_col[c]:intr_col[c] + Pn[c]] if c in intr_col else None
            out[c] = {"extrinsic_std": ext, "intrinsic_std": intr}
        return out

    n_clusters = int(np.unique(cluster_id).shape[0]) if cluster_id.size else 0
    return {"n_clusters": n_clusters, "n_blocks": int(r.size // 2), "K": K,
           "kernel": kernel, "scale": float(scale),
           "clustered": _cameras(cov_clu)}


def _obs_blocks(model, R_cam, R_obj, Xo, Xg, Xc, pts_2d, want_intr):
    """Per-observation residual + Jacobian blocks, the single source of the BA chain
    (board baked into Xo): returns ``(r (2N,), Jw_o, Jt_o (2N,3), Jw_c, Jt_c (2N,3),
    J_param (2N,P) or None)``. Object pose: ∂Xc/∂ω = R_cam R_obj(-[Xo]_x), ∂Xc/∂t = R_cam.
    Camera: ∂Xc/∂ω = -R_cam[Xg]_x, ∂Xc/∂t = I."""
    N = len(Xo)
    uv, J_point, J_param, valid = model.project_jacobian(Xc)
    mask = valid[:, None, None].astype(float)
    Jp = J_point * mask
    r = np.zeros((N, 2))
    r[valid] = uv[valid] - pts_2d[valid]
    dXc_dw_o = -np.einsum('ij,njk->nik', R_cam @ R_obj, _skew_batch(Xo))
    Jw_o = np.einsum('nij,njc->nic', Jp, dXc_dw_o).reshape(2 * N, 3)
    Jt_o = np.einsum('nij,jc->nic', Jp, R_cam).reshape(2 * N, 3)
    dXc_dw_c = -np.einsum('ij,njk->nik', R_cam, _skew_batch(Xg))
    Jw_c = np.einsum('nij,njc->nic', Jp, dXc_dw_c).reshape(2 * N, 3)
    Jt_c = Jp.reshape(2 * N, 3)
    Jpar = (J_param * mask).reshape(2 * N, -1) if want_intr else None
    return r.ravel(), Jw_o, Jt_o, Jw_c, Jt_c, Jpar


def build_schur_problem(rig: RigState, object_obs: List[ObjectObs], *,
                        fix_intrinsics: bool = True, fix_extrinsics: bool = False):
    """Assemble the rig BA for :func:`core.optimize.schur_lm`, mapping the **per-frame
    object poses to the eliminated block-diagonal ``local`` blocks** and
    ``{camera extrinsics, intrinsics}`` to the ``shared`` block.

    Each reprojection residual touches exactly one object pose (its frame) plus a slice of
    the shared state, so the Hessian is the block-arrow that the Schur trick collapses:
    eliminate every 6-DoF object pose with a 6×6 inverse, solve the small shared system,
    back-substitute. This is the sparse analogue of the dense :func:`build_problem` and
    the source of the speed-up on rigs with many frames (v-slam Ch.8 / lio-slam SMW).

    Returns ``(state0, residual, linearize, retract, shared_dim, n_groups)``.
    """
    ref_cam = rig.ref_cam_id
    cam_ids = [] if fix_extrinsics else [c for c in sorted(rig.cameras) if c != ref_cam]
    classes = {c: type(rig.cameras[c]) for c in rig.cameras}
    Pn = {c: len(classes[c].param_names) for c in rig.cameras}
    bounds = {c: classes[c].param_bounds() for c in rig.cameras}

    # shared layout: non-ref camera extrinsics, then (optionally) per-camera intrinsics
    col, cam_col, intr_col = 0, {}, {}
    for c in cam_ids:
        cam_col[c] = col
        col += 6
    if not fix_intrinsics:
        for c in sorted(rig.cameras):
            intr_col[c] = col
            col += Pn[c]
    shared_dim = col

    # groups = object poses; gather each group's observations once
    groups = sorted(rig.object_poses)
    gobs = {k: [] for k in groups}
    for o in object_obs:
        k = (o.object_id, o.frame_id)
        if o.cam_id in rig.cameras and k in rig.object_poses:
            gobs[k].append((o, rig.objects[o.object_id].pts_3d[o.point_rows]))
    n_groups = len(groups)

    def _models(state):
        return {c: classes[c].from_params(state["intr"][c]) for c in rig.cameras}

    def _xc(state, o, Xo, key):
        Xg = (state["obj_R"][key] @ Xo.T).T + state["obj_t"][key]
        Xc = (state["cam_R"][o.cam_id] @ Xg.T).T + state["cam_t"][o.cam_id]
        return Xg, Xc

    def residual(state):
        """Stacked reprojection residual, grouped by object pose (frame), for :func:`core.optimize.schur_lm`."""
        mc = _models(state)
        out = []
        for k in groups:
            for o, Xo in gobs[k]:
                _, Xc = _xc(state, o, Xo, k)
                uv, valid = mc[o.cam_id].project(Xc)
                d = np.zeros_like(o.pts_2d)
                d[valid] = uv[valid] - o.pts_2d[valid]
                out.append(d.ravel())
        return np.concatenate(out) if out else np.zeros(0)

    def linearize(state):
        """Per-group ``(r, A, B)`` blocks for :func:`core.optimize.schur_lm`.

        Returns three lists (one entry per object-pose group): ``r_list[i]`` the
        group's stacked residual, ``A_list[i]`` its Jacobian w.r.t. the shared block
        (camera extrinsics/intrinsics), ``B_list[i]`` its Jacobian w.r.t. that
        group's own 6-DoF local (object-pose) block.
        """
        mc = _models(state)
        r_list, A_list, B_list = [], [], []
        for k in groups:
            R_obj = state["obj_R"][k]
            rs, As, Bs = [], [], []
            for o, Xo in gobs[k]:
                cam = o.cam_id
                Xg, Xc = _xc(state, o, Xo, k)
                r, Jw_o, Jt_o, Jw_c, Jt_c, Jpar = _obs_blocks(
                    mc[cam], state["cam_R"][cam], R_obj, Xo, Xg, Xc, o.pts_2d,
                    not fix_intrinsics)
                m = len(r)
                A = np.zeros((m, shared_dim))
                if cam in cam_col:
                    cc = cam_col[cam]
                    A[:, cc:cc + 3] = Jw_c
                    A[:, cc + 3:cc + 6] = Jt_c
                if not fix_intrinsics:
                    ic = intr_col[cam]
                    A[:, ic:ic + Pn[cam]] = Jpar
                B = np.empty((m, 6))
                B[:, :3] = Jw_o
                B[:, 3:] = Jt_o
                rs.append(r)
                As.append(A)
                Bs.append(B)
            r_list.append(np.concatenate(rs))
            A_list.append(np.vstack(As))
            B_list.append(np.vstack(Bs))
        return r_list, A_list, B_list

    def retract(state, d_shared, d_local):
        """Apply the shared-block update ``d_shared`` (shared_dim,) and per-group local
        updates ``d_local`` (n_groups, 6) to ``state`` via SO(3) retraction on rotations
        and additive updates on translations/intrinsics. Returns a new state dict.
        """
        s = dict(state)
        s["cam_R"], s["cam_t"] = dict(state["cam_R"]), dict(state["cam_t"])
        s["obj_R"], s["obj_t"] = dict(state["obj_R"]), dict(state["obj_t"])
        s["intr"] = dict(state["intr"])
        for c in cam_ids:
            o = cam_col[c]
            s["cam_R"][c] = state["cam_R"][c] @ so3_exp(d_shared[o:o + 3])
            s["cam_t"][c] = state["cam_t"][c] + d_shared[o + 3:o + 6]
        if not fix_intrinsics:
            for c in sorted(rig.cameras):
                ic = intr_col[c]
                lb, ub = bounds[c]
                s["intr"][c] = np.clip(state["intr"][c] + d_shared[ic:ic + Pn[c]], lb, ub)
        for i, k in enumerate(groups):
            s["obj_R"][k] = state["obj_R"][k] @ so3_exp(d_local[i, :3])
            s["obj_t"][k] = state["obj_t"][k] + d_local[i, 3:]
        return s

    return _state_from_rig(rig), residual, linearize, retract, shared_dim, n_groups


def refine(rig: RigState, object_obs: List[ObjectObs], *,
           fix_intrinsics: bool = True, fix_extrinsics: bool = False, max_iter: int = 60,
           robust_kernel: str = "huber", robust_scale="auto", gnc_iters: int = 0,
           gnc_start: float = 0.0, noise_bound: Optional[float] = None,
           verbose: bool = False, sparse: bool = True,
           residual_mode: str = "pixel",
           on_iter=None) -> RigState:
    """One BA pass. Returns a refined copy of ``rig``.

    ``fix_intrinsics=True`` reproduces ``refineCameraGroupAndObjects`` (poses only);
    ``False`` reproduces ``refineCameraGroupAndObjectsAndIntrinsics`` (full joint).
    ``fix_extrinsics=True`` (cameras held) refines only the object poses — the per-object
    intermediate stage (``estimatePoseAllObjects`` / ``computeAllObjPoseInCameraGroup``).

    **Robust weighting, no rejection (default).** Every observation is kept; outliers are
    down-weighted by IRLS (``w = ρ'(r)/r``). ``robust_scale="auto"`` re-estimates the
    inlier scale by MAD each iteration so the kernel adapts to the actual noise instead of
    a hand-set pixel threshold. A redescending ``cauchy`` kernel mutes gross outliers
    smoothly; ``gnc_iters>0`` anneals the scale from ``gnc_start`` down (graduated
    non-convexity) so the redescending fit cannot get trapped by a bad initial residual.

    **High-breakdown rejection (opt-in).** The MAD auto-scale above is capped at 50%
    contamination (the median's breakdown). Pass ``noise_bound`` (the expected per-corner
    reprojection σ in pixels, e.g. ``~0.3``) to instead run a median-free **GNC-TLS** solve
    (:func:`core.optimize.gnc_tls_solve` / :func:`~core.optimize.gnc_tls_schur_solve`): it
    graduates a truncated surrogate against the explicit ``barc2 = (3.03·σ)²`` inlier band,
    recovers **past 50%** gross-outlier contamination, and returns a hard inlier set. When set,
    ``robust_kernel``/``robust_scale``/``gnc_*`` are ignored.

    ``sparse=True`` (default) Schur-eliminates the per-frame object poses
    (:func:`build_schur_problem` + :func:`core.optimize.schur_lm`); ``sparse=False`` uses
    the dense solver (kept for tests).

    ``on_iter(it, max_iter, rms, cost, rig_snapshot)`` — optional live-progress callback. The
    solver calls it with its own opaque ``state``; this function wraps it so callers see a
    real ``RigState`` snapshot of the *current* (mid-solve) cameras/extrinsics/object poses
    instead — reconstructed via :func:`_rig_from_state`, which is cheap (a handful of 4x4s
    per camera/frame, no large-array copies), so this does not measurably slow the solve.
    """
    rk = dict(robust_kernel=robust_kernel, robust_scale=robust_scale,
              gnc_iters=gnc_iters, gnc_start=gnc_start)
    wrapped_on_iter = (
        (lambda it, max_iter, rms, cost, state:
         on_iter(it, max_iter, rms, cost, _rig_from_state(rig, state)))
        if on_iter is not None else None)
    barc = None if noise_bound is None else 3.03 * float(noise_bound)   # 2-DoF 99% χ² band
    if residual_mode == "angular":
        # The bearing (angular) residual is the model-agnostic, pinhole/fisheye-uniform error:
        # compare predicted and observed unit rays with a full-sphere chordal residual.
        # Implemented on the dense analytic path with intrinsics held fixed (the observed bearing
        # is intrinsics-dependent), used as a geometry/structure polish.
        state0, residual, jacobian, retract, Kdim = build_problem(
            rig, object_obs, fix_intrinsics=True, fix_extrinsics=fix_extrinsics,
            residual_mode="angular")
        if Kdim == 0:
            return rig
        if barc is not None:
            res = gnc_tls_solve(state0, residual, jacobian, retract, noise_bound=barc,
                                block=3, inner_max_iter=max_iter, on_iter=wrapped_on_iter)
        else:
            res = lm_solve(state0, residual, jacobian, retract, block=3, max_iter=max_iter,
                           on_iter=wrapped_on_iter, **rk)
        return _rig_from_state(rig, res.state)
    if sparse:
        state0, residual, linearize, retract, shared_dim, n_groups = build_schur_problem(
            rig, object_obs, fix_intrinsics=fix_intrinsics, fix_extrinsics=fix_extrinsics)
        if shared_dim == 0 or n_groups == 0:           # nothing shared to solve -> dense
            return refine(rig, object_obs, fix_intrinsics=fix_intrinsics,
                          fix_extrinsics=fix_extrinsics, max_iter=max_iter,
                          noise_bound=noise_bound, verbose=verbose, sparse=False,
                          on_iter=on_iter, **rk)
        if barc is not None:
            res = gnc_tls_schur_solve(state0, residual, linearize, retract, noise_bound=barc,
                                      n_groups=n_groups, shared_dim=shared_dim, local_dim=6,
                                      block=2, inner_max_iter=max_iter,
                                      on_iter=wrapped_on_iter)
        else:
            res = schur_lm(state0, residual, linearize, retract, n_groups=n_groups,
                           shared_dim=shared_dim, local_dim=6, block=2, max_iter=max_iter,
                           on_iter=wrapped_on_iter, **rk)
    else:
        state0, residual, jacobian, retract, K = build_problem(
            rig, object_obs, fix_intrinsics=fix_intrinsics, fix_extrinsics=fix_extrinsics)
        if K == 0:
            return rig
        if barc is not None:
            res = gnc_tls_solve(state0, residual, jacobian, retract, noise_bound=barc,
                                block=2, inner_max_iter=max_iter, on_iter=wrapped_on_iter)
        else:
            res = lm_solve(state0, residual, jacobian, retract, block=2, max_iter=max_iter,
                           on_iter=wrapped_on_iter, **rk)
    if verbose:
        print(f"  BA: rms {res.rms:.4f}px iters={res.iterations} "
              f"intr={'free' if not fix_intrinsics else 'fixed'} "
              f"{'sparse' if sparse else 'dense'} kernel={robust_kernel}")
    return _rig_from_state(rig, res.state)


def refine_groups(rig: RigState, object_obs: List[ObjectObs], groups: List[List[int]],
                  **kw) -> RigState:
    """Per-camera-group intermediate refinement (``calibrateCameraGroup`` /
    ``refineAllCameraGroupAndObjects``): refine **each group independently** — its camera
    extrinsics (anchored at the group's first camera) plus the object poses its cameras
    observe — holding intrinsics fixed, with the same analytic-Jacobian BA. For a single
    connected group this refines the whole rig; for several groups it warm-starts each one
    before they are fused, exactly MC-Calib's hierarchy. ``kw`` forwards robust-kernel opts.

    ``on_iter``, if present in ``kw``, is called with a **full-rig** snapshot even though each
    group is solved on a cameras-only-in-that-group sub-``RigState`` internally: cameras and
    object poses outside the group currently being solved are filled in from ``out`` (already-
    refined groups' latest values; not-yet-processed groups' pre-refine values) so a live
    viewer never sees cameras vanish just because they aren't part of the group being updated
    this instant.
    """
    if len(groups) <= 1:
        return refine(rig, object_obs, fix_intrinsics=True, **kw)
    out = copy.copy(rig)
    out.T_c_g = dict(rig.T_c_g)
    out.object_poses = dict(rig.object_poses)
    out.cameras = dict(rig.cameras)
    caller_on_iter = kw.pop("on_iter", None)
    for grp in groups:
        grp_set = set(grp)
        sub_obs = [o for o in object_obs if o.cam_id in grp_set]
        sub_keys = {(o.object_id, o.frame_id) for o in sub_obs} & set(out.object_poses)
        if len(grp) < 2 or not sub_keys:
            continue
        sub = copy.copy(out)
        sub.cameras = {c: out.cameras[c] for c in grp}
        sub.T_c_g = {c: out.T_c_g[c] for c in grp}
        sub.object_poses = {k: out.object_poses[k] for k in sub_keys}
        sub.ref_cam_id = grp[0]                         # anchor the group at its first camera

        def _merged_on_iter(it, max_iter, rms, cost, partial_rig, _out=out):
            merged = copy.copy(_out)
            merged.cameras = dict(_out.cameras)
            merged.cameras.update(partial_rig.cameras)
            merged.T_c_g = dict(_out.T_c_g)
            merged.T_c_g.update(partial_rig.T_c_g)
            merged.object_poses = dict(_out.object_poses)
            merged.object_poses.update(partial_rig.object_poses)
            caller_on_iter(it, max_iter, rms, cost, merged)

        ref = refine(sub, sub_obs, fix_intrinsics=True,
                    on_iter=(_merged_on_iter if caller_on_iter is not None else None), **kw)
        for c in grp:
            out.T_c_g[c] = ref.T_c_g[c]
        for k in sub_keys:
            out.object_poses[k] = ref.object_poses[k]
    return out


def _robust_w(r2: np.ndarray, scale: float) -> np.ndarray:
    """Cauchy IRLS weight ``ρ'(r)/r`` for a squared-residual-per-point ``r2`` (down-weight,
    never reject)."""
    return 1.0 / (1.0 + r2 / (scale * scale))


def refine_object_structure(rig: RigState, object_obs: List[ObjectObs], *,
                            free_rows=None, iters: int = 10, robust_scale: float = 1.5
                            ) -> RigState:
    """Refine the fused object's 3-D point positions (MC-Calib's ``refineObject``).

    With the camera intrinsics, extrinsics and per-frame object poses held fixed, each free
    object point is re-triangulated from **all** its observations by a small robust
    Gauss-Newton (analytic Jacobian ``J_point · R_c · R_o``, Cauchy-weighted). Reconstructed
    nominal board geometry — and any inter-board pose / physical board imperfection baked into
    it — is corrected to what the corners actually imply, the single largest reprojection
    win on a real multi-board target.

    ``free_rows`` is the set of object-point rows to move; the rest are held to **anchor the
    gauge** (scale + frame). Default: every point except 3 spanning corners of the reference
    board, so the metric scale and object frame stay pinned while all real structure (both
    boards) is free. Returns a refined copy of ``rig``."""
    obj = rig.objects[next(iter(rig.objects))] if rig.objects else None
    if obj is None:
        return rig
    if free_rows is None:
        ref_rows = [i for i, (b, _c) in enumerate(obj.pts_obj_2_board)
                    if int(b) == obj.ref_board_id]
        anchor = {ref_rows[0], ref_rows[len(ref_rows) // 2], ref_rows[-1]} if ref_rows else set()
        free_rows = set(range(len(obj.pts_3d))) - anchor
    else:
        free_rows = set(int(r) for r in free_rows)

    # Index every free point to a dense 0..F-1 slot, and group observations by (camera, frame)
    # so each block's Jacobian is computed once per iteration for ALL its free points at once
    # (vectorised over points) and scatter-added into the per-point 3x3 normal equations —
    # ~10x faster than the per-point Python loop, since the work is now a handful of batched
    # matmuls per object pose, not one tiny solve per (point, view).
    free_sorted = sorted(free_rows)
    slot = {r: i for i, r in enumerate(free_sorted)}
    F = len(free_sorted)
    blocks = []                                    # (cam, key, slots(int[]), gidx(int[]), uv)
    for o in object_obs:
        key = (o.object_id, o.frame_id)
        if o.cam_id not in rig.cameras or key not in rig.object_poses:
            continue
        sel = [i for i, r in enumerate(o.point_rows) if int(r) in free_rows]
        if not sel:
            continue
        sel = np.asarray(sel, int)
        gidx = o.point_rows[sel].astype(int)
        slots = np.array([slot[int(r)] for r in gidx], int)
        blocks.append((o.cam_id, key, slots, gidx, o.pts_2d[sel].astype(float)))

    out = copy.copy(rig)
    pts = obj.pts_3d.copy()
    if F == 0 or not blocks:
        return out
    X = pts[free_sorted].astype(float).copy()       # (F,3) free points, working copy
    for _ in range(iters):
        H = np.zeros((F, 3, 3))
        g = np.zeros((F, 3))
        for cam, key, slots, gidx, uv in blocks:
            Tcg, Tgo = rig.T_c_g[cam], rig.object_poses[key]
            M = Tcg[:3, :3] @ Tgo[:3, :3]
            t = Tcg[:3, :3] @ Tgo[:3, 3] + Tcg[:3, 3]
            Xc = X[slots] @ M.T + t
            uvp, J_point, _Jp, valid = rig.cameras[cam].project_jacobian(Xc)
            J = np.einsum('nij,jk->nik', J_point, M)            # (n,2,3) ∂uv/∂X
            r = uvp - uv                                        # (n,2)
            w = _robust_w(np.einsum('ni,ni->n', r, r), robust_scale) * valid.astype(float)
            np.add.at(H, slots, w[:, None, None] * np.einsum('nij,nik->njk', J, J))
            np.add.at(g, slots, w[:, None] * np.einsum('nij,ni->nj', J, r))
        H += 1e-9 * np.eye(3)[None]
        try:
            dX = np.linalg.solve(H, -g[..., None])[..., 0]      # (F,3) all points at once
        except np.linalg.LinAlgError:
            break
        X += dX
        if float(np.max(np.einsum('ni,ni->n', dX, dX))) < 1e-14:
            break
    pts[free_sorted] = X
    new_obj = copy.copy(obj)
    new_obj.pts_3d = pts
    out.objects = dict(rig.objects)
    out.objects[obj.object_id] = new_obj
    return out


def _per_obs_errors(rig: RigState, object_obs: List[ObjectObs]) -> Dict[int, np.ndarray]:
    """Per-camera array of per-point reprojection errors (px)."""
    errs: Dict[int, list] = {}
    for o in object_obs:
        if o.cam_id not in rig.cameras:
            continue
        key = (o.object_id, o.frame_id)
        if key not in rig.object_poses:
            continue
        Xo = rig.objects[o.object_id].pts_3d[o.point_rows]
        Xg = (rig.object_poses[key][:3, :3] @ Xo.T).T + rig.object_poses[key][:3, 3]
        Xc = (rig.T_c_g[o.cam_id][:3, :3] @ Xg.T).T + rig.T_c_g[o.cam_id][:3, 3]
        uv, valid = rig.cameras[o.cam_id].project(Xc)
        errs.setdefault(o.cam_id, []).append(
            np.linalg.norm(uv[valid] - o.pts_2d[valid], axis=1))
    return {c: np.concatenate(v) if v else np.zeros(0) for c, v in errs.items()}


def per_observation_errors(rig: RigState, object_obs: List[ObjectObs]) -> Dict[int, np.ndarray]:
    """Public per-camera array of per-point reprojection errors (px) — the raw distribution
    behind :func:`reprojection_rms` / :func:`reprojection_metrics`, for callers (e.g.
    :mod:`.report`) that need the full mean/median/p95/max picture, not just one number."""
    return _per_obs_errors(rig, object_obs)


def reprojection_rms(rig: RigState, object_obs: List[ObjectObs]) -> Dict[int, float]:
    """Per-camera reprojection RMS (px) over all observations."""
    return {c: float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")
            for c, e in _per_obs_errors(rig, object_obs).items()}


def reprojection_metrics(rig: RigState, object_obs: List[ObjectObs],
                         inlier_px: float = 1.0) -> Dict[int, dict]:
    """Per-camera **robust** reprojection metrics. Naive all-corner RMS lies on a robust
    fit — it scores the size of the outliers the model deliberately down-weighted
    (docs/learn/robust_losses_and_evaluation.md). Report instead: ``median`` (50% break-
    down), ``inlier_rms`` (RMS over corners under ``inlier_px``), and ``inlier_frac``."""
    out = {}
    for c, e in _per_obs_errors(rig, object_obs).items():
        if not len(e):
            out[c] = dict(median=float("nan"), inlier_rms=float("nan"), inlier_frac=0.0,
                          rms=float("nan"))
            continue
        inl = e < inlier_px
        out[c] = dict(
            median=float(np.median(e)),
            inlier_rms=float(np.sqrt(np.mean(e[inl] ** 2))) if inl.any() else float("nan"),
            inlier_frac=float(inl.mean()),
            rms=float(np.sqrt(np.mean(e ** 2))),
        )
    return out
