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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.covariance import (
    clustered_sandwich_covariance,
    pixel_whiteners,
    psd_curvature_derivative,
    robust_weight_derivative,
)
from ..core.lie import hat_batch as _skew_batch
from ..core.lie import so3_exp
from ..core.numerics import NumericalError, psd_eigvalsh, spd_inverse
from ..core.optimize import box_active, gnc_tls_schur_solve, gnc_tls_solve, lm_solve, schur_lm
from ..core.robust import auto_kernel_scale, robust_weight
from ..geometry.bearing import chordal_bearing_residual_jacobian
from .types import ObjectObs, RigState

Key = Tuple[int, int]

# Largest diag(G)-scaled eigenvalue an absorbed board direction may keep after frame-pose
# elimination (rounding leaves <= 1e-12 on MC-Calib rigs); larger means the board
# co-observation graph and S disagree, and the covariance fails closed.
_ABSORBED_RTOL = 1e-9


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
    if state.get("board_base"):
        out.objects = dict(rig.objects)
        for oid, base in state["board_base"].items():
            obj = copy.copy(rig.objects[oid])
            obj.pts_3d = _board_points(base, state, oid)
            obj.T_co_b = dict(rig.objects[oid].T_co_b)
            for b in base["movable"]:
                R, t = state["board_R"][(oid, b)], state["board_t"][(oid, b)]
                if b in obj.T_co_b:
                    D = np.eye(4)
                    D[:3, :3] = R
                    D[:3, 3] = base["cent"][b] + t - R @ base["cent"][b]
                    obj.T_co_b[b] = D @ rig.objects[oid].T_co_b[b]
            out.objects[oid] = obj
    return out


def _board_bases(rig: RigState, bend: bool = False) -> dict:
    """Per fused object: starting points plus, per board, its rows, centroid and in-plane
    frame. Each non-reference board moves rigidly about its centroid,
    ``X = R_b (X0 - c_b + n_b w_b) + c_b + t_b``; the reference board holds the gauge. With
    ``bend``, every board also gets an out-of-plane quadratic bow
    ``w_b = k_b · [u², v², uv]`` (``u, v`` in-plane, normalised to the board half-extent, each
    term mean-centred so a bow cannot shift the board along its normal)."""
    bases = {}
    for oid, obj in rig.objects.items():
        rows_of: Dict[int, List[int]] = {}
        for i, (b, _c) in enumerate(obj.pts_obj_2_board):
            rows_of.setdefault(int(b), []).append(i)
        movable = [b for b in sorted(rows_of) if b != int(obj.ref_board_id)]
        bent = sorted(rows_of) if bend else []
        if not movable and not bent:
            continue
        P0 = np.asarray(obj.pts_3d, float).copy()
        base = {"P0": P0, "movable": movable, "bent": bent, "boards": sorted(rows_of),
                "rows": {b: np.asarray(r) for b, r in rows_of.items()},
                "cent": {b: P0[r].mean(0) for b, r in rows_of.items()},
                "normal": {}, "basis": np.zeros((len(P0), 3)),
                "board_of_row": np.asarray(obj.pts_obj_2_board, int)[:, 0]}
        for b in bent:
            r = base["rows"][b]
            q = P0[r] - base["cent"][b]
            _u, _s, Vt = np.linalg.svd(q, full_matrices=False)
            uv = q @ Vt[:2].T
            uv /= max(float(np.abs(uv).max()), 1e-12)
            F = np.c_[uv[:, 0] ** 2, uv[:, 1] ** 2, uv[:, 0] * uv[:, 1]]
            base["basis"][r] = F - F.mean(0)
            base["normal"][b] = Vt[2]
        bases[oid] = base
    return bases


def _board_layout(rig: RigState, col: int, *, refine_boards: bool, bend_boards: bool):
    """Shared columns of the board poses (``refine_boards``) and bows (``bend_boards``),
    starting at ``col``. Returns ``(bases, board_col, bend_col, end_col)``."""
    bases = _board_bases(rig, bend=bend_boards) if (refine_boards or bend_boards) else {}
    board_col, bend_col = {}, {}
    for oid in sorted(bases):
        for b in (bases[oid]["movable"] if refine_boards else []):
            board_col[(oid, b)] = col
            col += 6
        for b in bases[oid]["bent"]:
            bend_col[(oid, b)] = col
            col += 3
    return bases, board_col, bend_col, col


def _absorbed_boards(rig: RigState, object_obs: List[ObjectObs], bases: dict,
                     board_col: dict, bend_col: dict) -> List[Tuple[List[int], int]]:
    """Board directions the frame poses absorb, as ``[(columns, nullity)]``.

    Boards of one object that appear in the same frame are linked. A linked component
    without the reference board can move rigidly against it, and the frame poses follow, so 6
    directions of its board poses carry no information. A board never observed has none in its
    pose or bow."""
    seen: Dict[Key, set] = {}
    for o in object_obs:
        if (o.cam_id in rig.cameras and o.object_id in bases
                and (o.object_id, o.frame_id) in rig.object_poses):
            rows = np.asarray(o.point_rows, int)
            seen.setdefault((o.object_id, o.frame_id), set()).update(
                bases[o.object_id]["board_of_row"][rows].tolist())
    gauges = []
    for oid in sorted(bases):
        parent = {b: b for b in bases[oid]["boards"]}

        def root(b):
            while parent[b] != b:
                b = parent[b]
            return b
        observed = set()
        for (o2, _frame), boards in seen.items():
            if o2 == oid:
                boards = sorted(boards)
                observed |= set(boards)
                for b in boards[1:]:
                    parent[root(b)] = root(boards[0])
        ref = root(int(rig.objects[oid].ref_board_id))
        components: Dict[int, List[int]] = {}
        for b in bases[oid]["boards"]:
            components.setdefault(root(b), []).append(b)
        for r, members in sorted(components.items()):
            pose = [j for b in members if (oid, b) in board_col
                    for j in range(board_col[(oid, b)], board_col[(oid, b)] + 6)]
            if not observed & set(members):
                bows = [j for b in members if (oid, b) in bend_col
                        for j in range(bend_col[(oid, b)], bend_col[(oid, b)] + 3)]
                if pose or bows:
                    gauges.append((pose + bows, len(pose) + len(bows)))
            elif r != ref and pose:
                gauges.append((pose, 6))
    return gauges


def _board_points(base: dict, state: dict, oid: int) -> np.ndarray:
    P = base["P0"].copy()
    for b in base["boards"]:
        r, c = base["rows"][b], base["cent"][b]
        q = base["P0"][r] - c
        if b in base["normal"]:
            q = q + np.outer(base["basis"][r] @ state["board_k"][(oid, b)], base["normal"][b])
        if (oid, b) in state.get("board_R", {}):
            P[r] = q @ state["board_R"][(oid, b)].T + c + state["board_t"][(oid, b)]
        else:
            P[r] = q + c
    return P


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
    ``fix_intrinsics=True`` it is the pure object-pose stage. Optional observation pixel
    covariances whiten residuals and Jacobians together; robust scales then use whitened
    units. Angular residuals reject pixel covariance until bearing propagation is supported.
    """
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

    if angular and any(o.pixel_covariances is not None for o in object_obs):
        raise ValueError("pixel_covariances require pixel residuals; angular covariance propagation is unsupported")
    whiteners = {id(o): pixel_whiteners(o.pixel_covariances, len(o.pts_2d)) for o in object_obs}

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
            W = whiteners[id(o)]
            if W is not None:
                diff = np.einsum('nij,nj->ni', W, diff)
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
            W = whiteners[id(o)]
            if W is not None:
                block = J[row:row + 2 * N].reshape(N, 2, K)
                J[row:row + 2 * N] = np.einsum('nij,njk->nik', W, block).reshape(2 * N, K)
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

    def bound_active(state, g):
        """Intrinsic coordinates the clipping retraction holds at a bound (see
        :func:`core.optimize.box_active`)."""
        act = np.zeros(K, bool)
        if not fix_intrinsics:
            for c in sorted(rig.cameras):
                ic = intr_col[c]
                act[ic:ic + Pn[c]] = box_active(state["intr"][c], *bounds[c], g[ic:ic + Pn[c]])
        return act

    retract.bound_active = bound_active
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

    Unless ``fix_extrinsics``, this is computed by :func:`shared_covariance` -- the same
    estimator marginalised through the Schur structure, exact, without the dense
    Jacobian over every frame pose and with a well-conditioned (Jacobi-scaled) inverse.
    """
    if not fix_extrinsics:
        unc = shared_covariance(rig, object_obs, fix_intrinsics=fix_intrinsics, kernel=kernel,
                                scale=scale, small_cluster_correction=small_cluster_correction)
        std = _std_from_cov(unc["cov"])
        Pn = {c: len(type(rig.cameras[c]).param_names) for c in rig.cameras}
        cams = {}
        for c in sorted(rig.cameras):
            col, icol = unc["cam_col"].get(c), unc["intr_col"].get(c)
            cams[c] = {"extrinsic_std": None if col is None else std[col:col + 6],
                       "intrinsic_std": None if icol is None else std[icol:icol + Pn[c]]}
        K = len(std) + 6 * len(rig.object_poses)
        return {"n_clusters": unc["n_clusters"], "n_blocks": unc["n_blocks"], "K": K,
                "kernel": kernel, "scale": unc["scale"], "clustered": cams}
    return _dense_parameter_covariance(rig, object_obs, fix_intrinsics=fix_intrinsics,
                                       fix_extrinsics=fix_extrinsics, kernel=kernel,
                                       scale=scale,
                                       small_cluster_correction=small_cluster_correction)


def _std_from_cov(cov: np.ndarray) -> np.ndarray:
    """Per-parameter σ from a covariance diagonal. A variance below ``-n·eps·max|diag|`` (or
    non-finite) cannot come from a valid covariance and is reported as ``inf`` (unidentified),
    never clamped to a falsely confident 0; roundoff-small negatives are 0."""
    d = np.diag(np.asarray(cov, float)).copy()
    finite = np.isfinite(d)
    tol = d.size * np.finfo(float).eps * (float(np.max(np.abs(d[finite]))) if finite.any() else 0.0)
    std = np.full(d.shape, np.inf)
    ok = finite & (d >= -tol)
    std[ok] = np.sqrt(np.maximum(d[ok], 0.0))
    return std


def _dense_parameter_covariance(rig: RigState, object_obs: List[ObjectObs], *,
                                fix_intrinsics: bool, fix_extrinsics: bool, kernel: str,
                                scale: Optional[float], small_cluster_correction: bool) -> Dict:
    """Dense-Jacobian form of :func:`parameter_covariance` (every frame pose as a column);
    used for ``fix_extrinsics`` and as the reference :func:`shared_covariance` is tested
    against."""
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
        std = _std_from_cov(cov)
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


def shared_covariance(rig: RigState, object_obs: List[ObjectObs], *,
                      fix_intrinsics: bool = True, refine_boards: bool = False,
                      bend_boards: bool = False, kernel: str = "cauchy",
                      scale: Optional[float] = None,
                      small_cluster_correction: bool = True) -> Dict:
    r"""Frame-clustered M-estimator sandwich covariance of the **shared** rig parameters
    (camera extrinsics, intrinsics, board poses/bows), computed through the Schur structure.

    Exactly :func:`ds_msp.core.covariance.clustered_sandwich_covariance` (same bread with the
    Triggs curvature term, same cluster = frame pooled across cameras, same small-cluster
    correction) marginalised onto the shared block -- without forming the dense Jacobian over
    every frame pose, which :func:`parameter_covariance` does and which does not fit real rigs.
    With bread ``G`` partitioned into shared ``s`` and per-frame-pose ``k`` blocks
    (``G_kk`` 6x6, block diagonal) and per-cluster score ``Ψ_g``:

    ``S = G_ss − Σ_k G_sk G_kk⁻¹ G_ks``,  ``v_g = Ψ_g,s − Σ_{k∈g} G_sk G_kk⁻¹ Ψ_g,k``,
    ``Cov_ss = c · S⁻¹ (Σ_g v_g v_gᵀ) S⁻¹``.

    Frame poses (and, with ``refine_boards``/``bend_boards``, board poses/bows) are nuisance
    parameters marginalised in. Free per-point structure is not modelled. Returns
    ``{"cov", "identifiable", "unidentified_structure", "cam_col", "intr_col", "n_clusters",
    "n_blocks", "scale", "kernel"}`` with the
    shared column layout of :func:`build_schur_problem` (extrinsic tangent ``[δω rad, δt]``,
    right-multiplied, reference camera gauge-fixed)."""
    state, _residual, linearize, _retract, shared_dim, n_groups = build_schur_problem(
        rig, object_obs, fix_intrinsics=fix_intrinsics, refine_boards=refine_boards,
        bend_boards=bend_boards)
    groups = sorted(rig.object_poses)
    r_list, A_list, B_list = linearize(state)
    if scale is None:
        norms = np.concatenate([np.linalg.norm(r.reshape(-1, 2), axis=1) for r in r_list])
        scale = auto_kernel_scale(norms, kernel) if kernel != "none" else 1.0

    p_s = shared_dim
    G_ss = np.zeros((p_s, p_s))
    S = np.zeros((p_s, p_s))
    v_by_cluster: Dict[int, np.ndarray] = {}
    n_blocks = 0
    for k, r, A, B in zip(groups, r_list, A_list, B_list):
        n = len(r) // 2
        n_blocks += n
        rp = r.reshape(n, 2)
        Ap = A.reshape(n, 2, p_s)
        Bp = B.reshape(n, 2, 6)
        sq = np.einsum("nk,nk->n", rp, rp)
        om = robust_weight(sq, kernel, scale)
        omp = psd_curvature_derivative(sq, om, robust_weight_derivative(sq, kernel, scale))
        a_s = np.einsum("nkp,nk->np", Ap, rp)                  # J_iᵀ r_i, shared part
        a_l = np.einsum("nkp,nk->np", Bp, rp)                  # J_iᵀ r_i, local part
        Gss = (np.einsum("n,nki,nkj->ij", om, Ap, Ap)
               + 2.0 * np.einsum("n,ni,nj->ij", omp, a_s, a_s))
        Gsl = (np.einsum("n,nki,nkj->ij", om, Ap, Bp)
               + 2.0 * np.einsum("n,ni,nj->ij", omp, a_s, a_l))
        Gll = (np.einsum("n,nki,nkj->ij", om, Bp, Bp)
               + 2.0 * np.einsum("n,ni,nj->ij", omp, a_l, a_l))
        Gll = 0.5 * (Gll + Gll.T)
        if not np.isfinite(Gll).all():
            Gll_inv = np.full((6, 6), np.nan)              # S goes non-finite: fails closed below
        else:
            try:
                Gll_inv, _ = spd_inverse(Gll, where="shared_covariance", what="G_kk")
            except NumericalError:                         # a frame pose with a null direction:
                Gll_inv = np.linalg.pinv(Gll, hermitian=True)  # no information along it
        G_ss += Gss
        S += Gss - Gsl @ Gll_inv @ Gsl.T
        psi_s = om @ a_s                                       # Σ_i ω_i J_iᵀ r_i
        psi_l = om @ a_l
        v = psi_s - Gsl @ (Gll_inv @ psi_l)
        frame = k[1]
        v_by_cluster[frame] = v_by_cluster.get(frame, 0.0) + v
    V = np.array(list(v_by_cluster.values())) if v_by_cluster else np.zeros((0, p_s))
    M = V.T @ V
    n_clusters = len(v_by_cluster)
    if small_cluster_correction and n_clusters > 1:
        p_total = p_s + 6 * n_groups
        M *= (n_clusters / (n_clusters - 1.0)) * ((n_blocks - 1.0) / max(n_blocks - p_total, 1.0))
    S = 0.5 * (S + S.T)
    cam_col, intr_col, Pn = _report_column_layout(rig, fix_intrinsics=fix_intrinsics,
                                                  fix_extrinsics=False)
    # _report_column_layout interleaves frame-pose columns before intrinsics; the shared
    # Schur layout has them removed, so intrinsic columns shift down by 6 per frame pose.
    intr_col = {c: col - 6 * len(rig.object_poses) for c, col in intr_col.items()}
    start = 6 * len(cam_col) + sum(Pn[c] for c in intr_col)
    bases, board_col, bend_col, end = _board_layout(
        rig, start, refine_boards=refine_boards, bend_boards=bend_boards)
    assert end == p_s, (end, p_s)
    gauges = _absorbed_boards(rig, object_obs, bases, board_col, bend_col)
    cov, identifiable, absorbed = _sandwich_from_schur(S, M, G_ss, gauges)
    return {"cov": cov, "identifiable": identifiable, "unidentified_structure": absorbed,
            "cam_col": cam_col, "intr_col": intr_col, "n_clusters": n_clusters,
            "n_blocks": n_blocks, "scale": float(scale), "kernel": kernel}


def _sandwich_from_schur(S: np.ndarray, M: np.ndarray, G: Optional[np.ndarray] = None,
                         gauges: Sequence[Tuple[List[int], int]] = ()
                         ) -> Tuple[np.ndarray, bool, List[int]]:
    """``S⁻¹ M S⁻¹`` through a Jacobi-scaled inverse, failing closed.

    ``gauges`` lists board directions the frame poses absorb (:func:`_absorbed_boards`):
    ``(columns, nullity)``. Each one's ``nullity`` least-informative directions, found on
    ``S`` scaled by ``diag(G)`` (the information before frame-pose elimination; ``diag(S)``
    would blow their rounding noise up to O(1)), are projected out, and the sandwich is taken
    on the rest. They carry no information or score, so the remaining parameters'
    covariance does not depend on that choice. The gauge columns are returned as ``inf`` and
    listed (third value). Without gauges this is the plain inverse.

    No jitter: ``S`` mixes pixel-scale focal columns with radian and distortion columns
    (cond ~1e11 on a real rig), where even a 1e-12 trace-relative Tikhonov term perturbs the
    small-eigenvalue directions by percents. ``S`` that is non-finite, not positive definite
    after equilibration, numerically singular, or not null along a listed gauge
    (``_ABSORBED_RTOL``) leaves the shared parameters unidentified: every entry is returned
    as ``inf`` with ``identifiable=False`` instead of overflowing into NaN."""
    p = S.shape[0]
    G = S if G is None else G
    unidentified = np.full((p, p), np.inf)
    if not (np.isfinite(S).all() and np.isfinite(M).all() and np.isfinite(G).all()):
        return unidentified, False, list(range(p))
    absorbed = sorted(j for cols, _n in gauges for j in cols)
    # basis of the identified subspace: identity outside the gauges, and inside each gauge its
    # scaled eigenvectors with the `nullity` least-informative ones left out
    B = np.eye(p)
    null = []
    for cols, nullity in gauges:
        g = np.diag(G)[cols]
        Dc = np.where(g > 0.0, 1.0 / np.sqrt(np.where(g > 0.0, g, 1.0)), 1.0)
        ev, V = np.linalg.eigh(Dc[:, None] * S[np.ix_(cols, cols)] * Dc[None, :])
        if nullity and abs(ev[nullity - 1]) > _ABSORBED_RTOL:
            return unidentified, False, absorbed
        B[np.ix_(cols, cols)] = Dc[:, None] * V
        null.extend(cols[:nullity])
    B = np.delete(B, null, axis=1)
    Sr, Mr = B.T @ S @ B, B.T @ M @ B
    d = np.diag(Sr)
    if np.any(d <= 0.0):
        return unidentified, False, absorbed
    D = 1.0 / np.sqrt(d)
    Se = D[:, None] * Sr * D[None, :]
    ev = np.linalg.eigvalsh(Se)
    if ev[0] <= Sr.shape[0] * np.finfo(float).eps * ev[-1]:
        return unidentified, False, absorbed
    Sr_inv = D[:, None] * np.linalg.inv(Se) * D[None, :]
    cov = B @ (Sr_inv @ Mr @ Sr_inv.T) @ B.T
    cov = 0.5 * (cov + cov.T)
    cov[absorbed, :] = np.inf
    cov[:, absorbed] = np.inf
    keep = np.setdiff1d(np.arange(p), absorbed)
    if not np.isfinite(cov[np.ix_(keep, keep)]).all():
        return unidentified, False, absorbed
    return cov, True, absorbed


def pair_uncertainty(rig: RigState, unc: Dict) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Propagate :func:`shared_covariance` to every camera pair ``(a, b)``: 1-sigma relative
    rotation (deg) and camera-``b``-centre-in-``a`` (scene length units), each the square
    root of the largest eigenvalue of its 3x3 covariance (worst direction), plus the RMS
    over directions. Propagation is a central-difference Jacobian of the relative pose in
    the extrinsic tangents (cross-covariance between the two cameras included)."""
    cov, cam_col = unc["cov"], unc["cam_col"]
    cams = sorted(rig.T_c_g)

    def rel(Ta, Tb):
        T = Tb @ np.linalg.inv(Ta)                      # a -> b
        return T[:3, :3], -T[:3, :3].T @ T[:3, 3]       # rotation, centre of b in a

    def perturb(T, d):
        out = T.copy()
        out[:3, :3] = T[:3, :3] @ so3_exp(d[:3])
        out[:3, 3] = T[:3, 3] + d[3:]
        return out

    out = {}
    eps = 1e-6
    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            cols = [cam_col[c] + j for c in (a, b) if c in cam_col for j in range(6)]
            if not cols:
                continue
            R0, C0 = rel(rig.T_c_g[a], rig.T_c_g[b])
            Jf = np.zeros((6, len(cols)))
            for m, col in enumerate(cols):
                for sign in (1.0, -1.0):
                    d = {c: np.zeros(6) for c in (a, b)}
                    for c in (a, b):
                        if c in cam_col and cam_col[c] <= col < cam_col[c] + 6:
                            d[c][col - cam_col[c]] = sign * eps
                    R, C = rel(perturb(rig.T_c_g[a], d[a]), perturb(rig.T_c_g[b], d[b]))
                    w = R0.T @ R
                    dw = 0.5 * np.array([w[2, 1] - w[1, 2], w[0, 2] - w[2, 0], w[1, 0] - w[0, 1]])
                    Jf[:, m] += sign * np.r_[dw, C - C0] / (2 * eps)
            block = cov[np.ix_(cols, cols)]
            P = Jf @ block @ Jf.T if np.isfinite(block).all() else None
            ev_r = ev_c = None
            if P is not None and np.isfinite(P).all():
                P = 0.5 * (P + P.T)
                try:
                    # A clearly negative variance is a defect, not a zero: report the pair
                    # as unidentified rather than clamping it to a falsely tight sigma.
                    # (rtol 1e-8: the propagation's roundoff, across mixed rotation and
                    # length units, can exceed n*eps of a 3x3 block.)
                    ev_r = psd_eigvalsh(P[:3, :3], where="pair_uncertainty",
                                        what="rotation cov", rtol=1e-8)
                    ev_c = psd_eigvalsh(P[3:, 3:], where="pair_uncertainty",
                                        what="centre cov", rtol=1e-8)
                except NumericalError:
                    ev_r = ev_c = None
            if ev_r is None or ev_c is None:
                out[(a, b)] = {"rotation_sigma_deg": np.inf, "rotation_sigma_rms_deg": np.inf,
                               "centre_sigma": np.inf, "centre_sigma_rms": np.inf,
                               "covariance": None, "identifiable": False}
                continue
            out[(a, b)] = {
                "rotation_sigma_deg": float(np.degrees(np.sqrt(ev_r[-1]))),
                "rotation_sigma_rms_deg": float(np.degrees(np.sqrt(ev_r.sum() / 3))),
                "centre_sigma": float(np.sqrt(ev_c[-1])),
                "centre_sigma_rms": float(np.sqrt(ev_c.sum() / 3)),
                "covariance": P.tolist(),
                "identifiable": True,
            }
    return out


def build_schur_problem(rig: RigState, object_obs: List[ObjectObs], *,
                        fix_intrinsics: bool = True, fix_extrinsics: bool = False,
                        refine_boards: bool = False, bend_boards: bool = False):
    """Assemble the rig BA for :func:`core.optimize.schur_lm`, mapping the **per-frame
    object poses to the eliminated block-diagonal ``local`` blocks** and
    ``{camera extrinsics, intrinsics}`` to the ``shared`` block.

    Each reprojection residual touches exactly one object pose (its frame) plus a slice of
    the shared state, so the Hessian is the block-arrow that the Schur trick collapses:
    eliminate every 6-DoF object pose with a 6×6 inverse, solve the small shared system,
    back-substitute. This is the sparse analogue of the dense :func:`build_problem` and
    the source of the speed-up on rigs with many frames (v-slam Ch.8 / lio-slam SMW).

    ``refine_boards=True`` adds each non-reference board's rigid 6-DOF pose inside its fused
    object to the shared block (tangent ``[δω, δt]`` about the board centroid, left-multiplied:
    ``∂X_o/∂ω = -[R_b(X0 - c_b)]_×``, ``∂X_o/∂t = I``). Board shape is held; this refines the
    inter-board poses :func:`~.object3d.build_objects` seeds from averaged PnP jointly with the
    rig instead of freezing them. ``bend_boards=True`` adds a 3-parameter out-of-plane quadratic
    bow per board (``∂X_o/∂k = R_b n_b [u², v², uv]``) for physically warped targets.

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
    bases, board_col, bend_col, shared_dim = _board_layout(
        rig, col, refine_boards=refine_boards, bend_boards=bend_boards)

    # groups = object poses. Every observation's points are laid out once, flat and in the
    # residual order (group -> observation -> point -> (u, v)), so each evaluation is one
    # projection call per camera plus array algebra; per-group blocks are contiguous slices.
    groups = sorted(rig.object_poses)
    gindex = {k: i for i, k in enumerate(groups)}
    gobs = {k: [] for k in groups}
    for o in object_obs:
        k = (o.object_id, o.frame_id)
        if o.cam_id in rig.cameras and k in rig.object_poses:
            gobs[k].append(o)
    n_groups = len(groups)
    seq = [o for k in groups for o in gobs[k]]
    counts = np.array([len(o.point_rows) for o in seq], int)
    N = int(counts.sum())
    p_group = np.repeat([gindex[(o.object_id, o.frame_id)] for o in seq], counts).astype(int)
    p_cam = np.repeat([o.cam_id for o in seq], counts).astype(int)
    p_oid = np.repeat([o.object_id for o in seq], counts).astype(int)
    p_row = (np.concatenate([np.asarray(o.point_rows, int) for o in seq]) if seq
             else np.zeros(0, int))
    p_uv = (np.concatenate([np.asarray(o.pts_2d, float) for o in seq]) if seq
            else np.zeros((0, 2)))
    Xo_static = (np.concatenate([rig.objects[o.object_id].pts_3d[o.point_rows] for o in seq])
                 if seq else np.zeros((0, 3)))
    p_W = None
    if any(o.pixel_covariances is not None for o in seq):
        p_W = np.concatenate([pixel_whiteners(o.pixel_covariances, len(o.pts_2d))
                              if o.pixel_covariances is not None
                              else np.tile(np.eye(2), (len(o.pts_2d), 1, 1)) for o in seq])
    gstart = np.zeros(n_groups + 1, int)
    np.add.at(gstart, p_group + 1, 1)
    gstart = np.cumsum(gstart)
    cam_pts = {c: np.flatnonzero(p_cam == c) for c in rig.cameras}
    board_masks = {}                                  # (oid, b) -> point indices on that board
    for oid, base in bases.items():
        on = p_oid == oid
        bids = np.full(N, -1)
        bids[on] = base["board_of_row"][p_row[on]]
        for bb in base["boards"]:
            board_masks[(oid, bb)] = np.flatnonzero(on & (bids == bb))

    def _models(state):
        return {c: classes[c].from_params(state["intr"][c]) for c in rig.cameras}

    def _object_points(state):
        return {oid: _board_points(base, state, oid) for oid, base in bases.items()}

    def _geometry(state):
        """Flat ``Xo, Xg, Xc`` and the per-point object/camera rotations."""
        Xo = Xo_static.copy()
        for oid, P in _object_points(state).items():
            m = p_oid == oid
            Xo[m] = P[p_row[m]]
        R_obj = np.stack([state["obj_R"][k] for k in groups])[p_group] if N else np.zeros((0, 3, 3))
        t_obj = np.stack([state["obj_t"][k] for k in groups])[p_group] if N else np.zeros((0, 3))
        cams = sorted(rig.cameras)
        cpos = {c: i for i, c in enumerate(cams)}
        ci = np.array([cpos[c] for c in p_cam], int)
        R_cam = np.stack([state["cam_R"][c] for c in cams])[ci]
        t_cam = np.stack([state["cam_t"][c] for c in cams])[ci]
        Xg = np.einsum('nij,nj->ni', R_obj, Xo) + t_obj
        Xc = np.einsum('nij,nj->ni', R_cam, Xg) + t_cam
        return Xo, Xg, Xc, R_obj, R_cam

    def residual(state):
        """Stacked reprojection residual, grouped by object pose (frame), for :func:`core.optimize.schur_lm`."""
        mc = _models(state)
        _Xo, _Xg, Xc, _Ro, _Rc = _geometry(state)
        d = np.zeros((N, 2))
        for c, idx in cam_pts.items():
            if not len(idx):
                continue
            uv, valid = mc[c].project(Xc[idx])
            valid = np.asarray(valid, bool).ravel()
            d[idx[valid]] = uv[valid] - p_uv[idx[valid]]
        if p_W is not None:
            d = np.einsum('nij,nj->ni', p_W, d)
        return d.ravel()

    def linearize(state):
        """Per-group ``(r, A, B)`` blocks for :func:`core.optimize.schur_lm`.

        Returns three lists (one entry per object-pose group): ``r_list[i]`` the
        group's stacked residual, ``A_list[i]`` its Jacobian w.r.t. the shared block
        (camera extrinsics/intrinsics, board poses/bows), ``B_list[i]`` its Jacobian w.r.t.
        that group's own 6-DoF local (object-pose) block. Object pose: ∂Xc/∂ω =
        R_cam R_obj(-[Xo]_×), ∂Xc/∂t = R_cam; camera: ∂Xc/∂ω = -R_cam[Xg]_×, ∂Xc/∂t = I.
        """
        mc = _models(state)
        Xo, Xg, Xc, R_obj, R_cam = _geometry(state)
        r = np.zeros((N, 2))
        Jp = np.zeros((N, 2, 3))
        A = np.zeros((N, 2, shared_dim))
        for c, idx in cam_pts.items():
            if not len(idx):
                continue
            uv, J_point, J_param, valid = mc[c].project_jacobian(Xc[idx])
            valid = np.asarray(valid, bool).ravel()
            r[idx[valid]] = uv[valid] - p_uv[idx[valid]]
            Jp[idx] = J_point * valid[:, None, None]
            if not fix_intrinsics:
                ic = intr_col[c]
                A[idx, :, ic:ic + Pn[c]] = J_param * valid[:, None, None]
        M = np.einsum('nij,njk->nik', R_cam, R_obj)
        B = np.empty((N, 2, 6))
        B[:, :, :3] = -np.einsum('nij,njk,nkl->nil', Jp, M, _skew_batch(Xo))
        B[:, :, 3:] = np.einsum('nij,njk->nik', Jp, R_cam)
        for c in cam_col:
            idx = cam_pts[c]
            if not len(idx):
                continue
            cc = cam_col[c]
            A[idx, :, cc:cc + 3] = -np.einsum('nij,njk,nkl->nil', Jp[idx], R_cam[idx],
                                              _skew_batch(Xg[idx]))
            A[idx, :, cc + 3:cc + 6] = Jp[idx]
        if board_col or bend_col:
            Jx = np.einsum('nij,njk->nik', Jp, M)              # ∂uv/∂X_o
            for key, idx in board_masks.items():
                if not len(idx):
                    continue
                oid, bb = key
                base = bases[oid]
                if key in board_col:
                    q = Xo[idx] - base["cent"][bb] - state["board_t"][key]
                    cb = board_col[key]
                    A[idx, :, cb:cb + 3] = -np.einsum('nij,njk->nik', Jx[idx], _skew_batch(q))
                    A[idx, :, cb + 3:cb + 6] = Jx[idx]
                if key in bend_col:
                    Rb = state["board_R"][key] if key in board_col else np.eye(3)
                    ck = bend_col[key]
                    F = base["basis"][p_row[idx]]
                    A[idx, :, ck:ck + 3] = np.einsum('nij,j,nk->nik', Jx[idx],
                                                     Rb @ base["normal"][bb], F)
        if p_W is not None:
            r = np.einsum('nij,nj->ni', p_W, r)
            A = np.einsum('nij,njk->nik', p_W, A)
            B = np.einsum('nij,njk->nik', p_W, B)
        r_list, A_list, B_list = [], [], []
        for g in range(n_groups):
            s0, s1 = gstart[g], gstart[g + 1]
            r_list.append(r[s0:s1].ravel())
            A_list.append(A[s0:s1].reshape(2 * (s1 - s0), shared_dim))
            B_list.append(B[s0:s1].reshape(2 * (s1 - s0), 6))
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
        if board_col:
            s["board_R"], s["board_t"] = dict(state["board_R"]), dict(state["board_t"])
            for key, o in board_col.items():
                s["board_R"][key] = so3_exp(d_shared[o:o + 3]) @ state["board_R"][key]
                s["board_t"][key] = state["board_t"][key] + d_shared[o + 3:o + 6]
        if bend_col:
            s["board_k"] = dict(state["board_k"])
            for key, o in bend_col.items():
                s["board_k"][key] = state["board_k"][key] + d_shared[o:o + 3]
        return s

    def bound_active(state, g):
        """Shared intrinsic coordinates held at a bound (see :func:`core.optimize.box_active`)."""
        act = np.zeros(shared_dim, bool)
        if not fix_intrinsics:
            for c in sorted(rig.cameras):
                ic = intr_col[c]
                act[ic:ic + Pn[c]] = box_active(state["intr"][c], *bounds[c], g[ic:ic + Pn[c]])
        return act

    retract.bound_active = bound_active

    state0 = _state_from_rig(rig)
    if bases:
        state0["board_R"] = {key: np.eye(3) for key in board_col}
        state0["board_t"] = {key: np.zeros(3) for key in board_col}
        state0["board_k"] = {key: np.zeros(3) for key in bend_col}
        state0["board_base"] = bases
    return state0, residual, linearize, retract, shared_dim, n_groups


def solve_record(res, solver: str) -> Dict:
    """JSON-safe summary of one solver run (:class:`~ds_msp.core.optimize.OptResult`):
    ``solver``, termination ``status`` and ``stop_reason``, ``iterations``, final ``rms``
    (block residual units, pixels here) and ``gradient_ratio``. Non-finite numbers are
    ``None`` so the record can go into strict JSON."""
    def num(x):
        return float(x) if x is not None and np.isfinite(x) else None
    return {"solver": solver, "status": res.status, "stop_reason": res.stop_reason,
            "iterations": int(res.iterations), "rms": num(res.rms),
            "gradient_ratio": num(getattr(res, "gradient_ratio", None))}


def refine(rig: RigState, object_obs: List[ObjectObs], *,
           fix_intrinsics: bool = True, fix_extrinsics: bool = False, max_iter: int = 60,
           robust_kernel: str = "huber", robust_scale="auto", gnc_iters: int = 0,
           gnc_start: float = 0.0, noise_bound: Optional[float] = None,
           verbose: bool = False, sparse: bool = True,
           residual_mode: str = "pixel", refine_boards: bool = False,
           bend_boards: bool = False, on_iter=None,
           log: Optional[List[Dict]] = None) -> RigState:
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

    ``refine_boards=True`` also refines each non-reference board's rigid pose inside every
    fused multi-board object, and ``bend_boards=True`` a quadratic out-of-plane bow per board
    (see :func:`build_schur_problem`); sparse pixel path only.

    ``sparse=True`` (default) Schur-eliminates the per-frame object poses
    (:func:`build_schur_problem` + :func:`core.optimize.schur_lm`); ``sparse=False`` uses
    the dense solver (kept for tests).

    ``on_iter(it, max_iter, rms, cost, rig_snapshot)`` — optional live-progress callback. The
    solver calls it with its own opaque ``state``; this function wraps it so callers see a
    real ``RigState`` snapshot of the *current* (mid-solve) cameras/extrinsics/object poses
    instead — reconstructed via :func:`_rig_from_state`, which is cheap (a handful of 4x4s
    per camera/frame, no large-array copies), so this does not measurably slow the solve.

    ``log``, when given, gets one :func:`solve_record` dict appended per solver run (none when
    there is nothing to refine), so a caller can report every stage's termination status.
    """
    rk = dict(robust_kernel=robust_kernel, robust_scale=robust_scale,
              gnc_iters=gnc_iters, gnc_start=gnc_start)
    wrapped_on_iter = (
        (lambda it, max_iter, rms, cost, state:
         on_iter(it, max_iter, rms, cost, _rig_from_state(rig, state)))
        if on_iter is not None else None)
    barc = None if noise_bound is None else 3.03 * float(noise_bound)   # 2-DoF 99% χ² band
    if (refine_boards or bend_boards) and (residual_mode != "pixel" or not sparse):
        raise ValueError("refine_boards/bend_boards require the sparse pixel-residual solver")
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
                                block=3, inner_max_iter=max_iter, on_iter=wrapped_on_iter,
                                bound_active=retract.bound_active)
        else:
            res = lm_solve(state0, residual, jacobian, retract, block=3, max_iter=max_iter,
                           on_iter=wrapped_on_iter, bound_active=retract.bound_active, **rk)
        if log is not None:
            log.append(solve_record(res, "gnc_tls" if barc is not None else "lm"))
        return _rig_from_state(rig, res.state)
    if sparse:
        state0, residual, linearize, retract, shared_dim, n_groups = build_schur_problem(
            rig, object_obs, fix_intrinsics=fix_intrinsics, fix_extrinsics=fix_extrinsics,
            refine_boards=refine_boards, bend_boards=bend_boards)
        # No shared parameters (cameras and intrinsics both fixed) is the pure per-frame pose
        # stage: block-diagonal, so the Schur solver handles it as independent 6x6 solves. The
        # dense fallback here was O((6·frames)³) per iteration -- ~5,900 columns on a 985-frame
        # camera -- and dominated runtime on full-length captures. Dense only without groups.
        if n_groups == 0:
            return refine(rig, object_obs, fix_intrinsics=fix_intrinsics,
                          fix_extrinsics=fix_extrinsics, max_iter=max_iter,
                          noise_bound=noise_bound, verbose=verbose, sparse=False,
                          on_iter=on_iter, log=log, **rk)
        if barc is not None:
            res = gnc_tls_schur_solve(state0, residual, linearize, retract, noise_bound=barc,
                                      n_groups=n_groups, shared_dim=shared_dim, local_dim=6,
                                      block=2, inner_max_iter=max_iter,
                                      on_iter=wrapped_on_iter,
                                      bound_active=retract.bound_active)
        else:
            res = schur_lm(state0, residual, linearize, retract, n_groups=n_groups,
                           shared_dim=shared_dim, local_dim=6, block=2, max_iter=max_iter,
                           on_iter=wrapped_on_iter, bound_active=retract.bound_active, **rk)
    else:
        state0, residual, jacobian, retract, K = build_problem(
            rig, object_obs, fix_intrinsics=fix_intrinsics, fix_extrinsics=fix_extrinsics)
        if K == 0:
            return rig
        if barc is not None:
            res = gnc_tls_solve(state0, residual, jacobian, retract, noise_bound=barc,
                                block=2, inner_max_iter=max_iter, on_iter=wrapped_on_iter,
                                bound_active=retract.bound_active)
        else:
            res = lm_solve(state0, residual, jacobian, retract, block=2, max_iter=max_iter,
                           on_iter=wrapped_on_iter, bound_active=retract.bound_active, **rk)
    if log is not None:
        kind = "gnc_tls" if barc is not None else "lm"
        log.append(solve_record(res, kind + ("_schur" if sparse else "")))
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


def _structure_anchors(points: np.ndarray, rows) -> set:
    """Choose a well-spread noncollinear triangle, independent of point ordering."""
    rows = np.asarray(rows, int)
    if len(rows) < 3:
        raise ValueError("structure refinement needs three noncollinear reference points")
    xyz = np.asarray(points)[rows]
    b = int(np.argmax(np.linalg.norm(xyz - xyz[0], axis=1)))
    edge = xyz[b] - xyz[0]
    areas = np.linalg.norm(np.cross(xyz - xyz[0], edge), axis=1)
    c = int(np.argmax(areas))
    if areas[c] <= 1e-10 * max(float(edge @ edge), np.finfo(float).tiny):
        raise ValueError("structure refinement reference points are collinear")
    return {int(rows[0]), int(rows[b]), int(rows[c])}


def refine_object_structure(rig: RigState, object_obs: List[ObjectObs], *,
                            free_rows=None, iters: int = 10, robust_scale: float = 1.5,
                            prior_sigma: Optional[float] = None, prior_points=None,
                            pixel_sigma: Optional[float] = None) -> RigState:
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
    boards) is free. ``prior_sigma`` adds an isotropic Gaussian prior in target length units;
    ``prior_points`` can retain the original geometry across repeated calls. Without a prior,
    movement can absorb camera/detector bias and does not establish physical target shape.
    The prior is weighted against the reprojection term at the corner noise ``pixel_sigma``
    (whitened units when observations carry covariances; default: a MAD estimate from the
    current residuals), i.e. the MAP weight ``σ_px² / σ_target²`` -- so ``prior_sigma`` means
    the same length whatever the detector noise.
    Returns a refined copy of ``rig``."""
    obj = rig.objects[next(iter(rig.objects))] if rig.objects else None
    if obj is None:
        return rig
    if free_rows is None:
        ref_rows = [i for i, (b, _c) in enumerate(obj.pts_obj_2_board)
                    if int(b) == obj.ref_board_id]
        anchor = _structure_anchors(obj.pts_3d, ref_rows)
        free_rows = set(range(len(obj.pts_3d))) - anchor
    else:
        free_rows = set(int(r) for r in free_rows)
        _structure_anchors(obj.pts_3d, sorted(set(range(len(obj.pts_3d))) - free_rows))
    if prior_sigma is not None and (not np.isfinite(prior_sigma) or prior_sigma <= 0):
        raise ValueError("structure prior_sigma must be positive in target length units")
    prior = np.asarray(obj.pts_3d if prior_points is None else prior_points, float)
    if prior.shape != obj.pts_3d.shape or not np.isfinite(prior).all():
        raise ValueError("structure prior_points must match finite target geometry")

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
        if o.object_id != obj.object_id or o.cam_id not in rig.cameras or key not in rig.object_poses:
            continue
        sel = [i for i, r in enumerate(o.point_rows) if int(r) in free_rows]
        if not sel:
            continue
        sel = np.asarray(sel, int)
        gidx = o.point_rows[sel].astype(int)
        slots = np.array([slot[int(r)] for r in gidx], int)
        cov = None if o.pixel_covariances is None else np.asarray(o.pixel_covariances)[sel]
        blocks.append((o.cam_id, key, slots, gidx, o.pts_2d[sel].astype(float),
                       pixel_whiteners(cov, len(sel))))

    out = copy.copy(rig)
    pts = obj.pts_3d.copy()
    if F == 0 or not blocks:
        return out
    X = pts[free_sorted].astype(float).copy()       # (F,3) free points, working copy
    prior_w = 0.0
    if prior_sigma is not None:
        if pixel_sigma is None:
            pixel_sigma = _residual_sigma(rig, blocks, X)
        prior_w = (float(pixel_sigma) / float(prior_sigma)) ** 2
    for _ in range(iters):
        H = np.zeros((F, 3, 3))
        g = np.zeros((F, 3))
        for cam, key, slots, gidx, uv, W in blocks:
            Tcg, Tgo = rig.T_c_g[cam], rig.object_poses[key]
            M = Tcg[:3, :3] @ Tgo[:3, :3]
            t = Tcg[:3, :3] @ Tgo[:3, 3] + Tcg[:3, 3]
            Xc = X[slots] @ M.T + t
            uvp, J_point, _Jp, valid = rig.cameras[cam].project_jacobian(Xc)
            J = np.einsum('nij,jk->nik', J_point, M)            # (n,2,3) ∂uv/∂X
            r = uvp - uv                                        # (n,2)
            if W is not None:
                r = np.einsum('nij,nj->ni', W, r)
                J = np.einsum('nij,njk->nik', W, J)
            # Drop invalid or non-finite rows by selection: multiplying a NaN by a zero
            # weight is still NaN, and one NaN would poison that point's normal equations.
            use = (np.asarray(valid, bool) & np.isfinite(r).all(axis=1)
                   & np.isfinite(J).all(axis=(1, 2)))
            r = np.where(use[:, None], r, 0.0)
            J = np.where(use[:, None, None], J, 0.0)
            w = np.where(use, _robust_w(np.einsum('ni,ni->n', r, r), robust_scale), 0.0)
            np.add.at(H, slots, w[:, None, None] * np.einsum('nij,nik->njk', J, J))
            np.add.at(g, slots, w[:, None] * np.einsum('nij,ni->nj', J, r))
        if prior_w:
            H += prior_w * np.eye(3)[None]
            g += prior_w * (X - prior[free_sorted])
        H += 1e-9 * np.eye(3)[None]
        try:
            dX = np.linalg.solve(H, -g[..., None])[..., 0]      # (F,3) all points at once
        except np.linalg.LinAlgError:
            break
        # A point whose update is not finite keeps its position (never write NaN into the
        # target); every other point moves.
        dX = np.where(np.isfinite(dX).all(axis=1)[:, None], dX, 0.0)
        X += dX
        if float(np.max(np.einsum('ni,ni->n', dX, dX))) < 1e-14:
            break
    pts[free_sorted] = X
    new_obj = copy.copy(obj)
    new_obj.pts_3d = pts
    out.objects = dict(rig.objects)
    out.objects[obj.object_id] = new_obj
    return out


def _residual_sigma(rig: RigState, blocks, X: np.ndarray) -> float:
    """Robust per-coordinate corner-noise σ (MAD) of the whitened residuals of the free points."""
    comps = []
    for cam, key, slots, _gidx, uv, W in blocks:
        Tcg, Tgo = rig.T_c_g[cam], rig.object_poses[key]
        M = Tcg[:3, :3] @ Tgo[:3, :3]
        t = Tcg[:3, :3] @ Tgo[:3, 3] + Tcg[:3, 3]
        uvp, valid = rig.cameras[cam].project(X[slots] @ M.T + t)
        r = uvp - uv
        if W is not None:
            r = np.einsum('nij,nj->ni', W, r)
        comps.append(r[valid].ravel())
    r = np.concatenate(comps) if comps else np.zeros(0)
    r = r[np.isfinite(r)]
    if not len(r):
        return 1.0
    return max(float(1.4826 * np.median(np.abs(r - np.median(r)))), 1e-6)


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
