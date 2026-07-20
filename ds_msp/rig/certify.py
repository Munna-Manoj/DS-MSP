"""A-posteriori global-optimality certificate for the rig's extrinsic-rotation backbone.

After calibration, the per-view robust-PnP poses (``ObjectObs.T_c_o``) define a rotation-
synchronization problem over the bipartite camera x board-placement co-observation graph.
**Convention (load-bearing, verified by the zero-noise sign test):** node rotations are
frame-from-world, ``Z_c = R_c_g`` and ``Z_t = R_g_oᵀ``; each observation contributes the
measurement ``R̃_ct = R_c_o = R_c_g · R_g_o = Z_c Z_tᵀ`` — the ``R̃_ij ≈ Z_i Z_j⁻¹`` form
the column-stacked connection Laplacian in :mod:`ds_msp.core.rotsync` encodes. (Using the
transposed world-from-frame convention mis-poses the problem *silently* — the certificate
then certifies the wrong cost's optimum ~58 deg away; the ``d_ba_chordal_deg`` diagnostic
is what catches it, which is why it is a mandatory output.) Refining the BA rotations to a
stationary point of the weighted chordal cost and checking the dual certificate there
answers, with a proof rather than a heuristic: *is this rotation configuration the global
optimum of the rotation backbone implied by the per-view measurements, or could the solver
have converged to a different, wrong basin?*

**What a PASS proves and does not prove** (soundness audit, 2026-07-18): it proves global
optimality of the *weighted chordal rotation-averaging cost on this measurement graph* —
with one-sided error (a FAIL is inconclusive, never evidence of a wrong solution). It does
NOT certify: the correctness of the input PnP rotations (mutually-consistent wrong
measurements certify cleanly — garbage in, certified garbage out), translations, any
intrinsics (the fisheye projection is not polynomial), or the reprojection-BA optimum
itself (the chordal and reprojection optima differ slightly; the rotation distance between
them is reported as its own diagnostic). Prior art for the mechanism: SE-Sync, Shonan
averaging (shipped in GTSAM for SLAM factor graphs), TEASER++. DS-MSP's contribution is
surfacing it as a user-facing calibration-trust output.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..core.rotsync import certificate, chordal_grad, connection_laplacian, refine_chordal
from .types import ObjectObs, RigState

# Dimensionless PSD tolerance: λ_min normalized by the mean node degree must exceed −TOL.
# The 3-per-component gauge zeros sit at machine precision (measured ~1e-15 relative), and
# a planted wrong-basin critical point measures η ≈ −0.13 — eight orders below/above, so
# the tolerance is uncritical inside that gap.
_ETA_TOL = 1e-6

# Gross-outlier edge trim: an edge whose rotation residual against the refined chordal
# solution exceeds max(6 * median, 15 deg) is a wrong *measurement* (measured on both real
# datasets 2026-07-18: inlier edges 0.6-2.3 deg median with p90 <= 11 deg, versus discrete
# gross outliers at 80-179 deg — near-antipodal fisheye-PnP flips, the ADR-0013 failure
# mode). Certifying with them in would certify a contaminated problem (the documented
# garbage-in caveat); instead they are trimmed transparently and REPORTED as findings — the
# outlier list is itself one of the audit's most valuable outputs.
_TRIM_MED_FACTOR = 6.0
_TRIM_FLOOR_DEG = 15.0


def _rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def certify_rotations(rig: RigState, object_obs: List[ObjectObs], *,
                      max_refine_iter: int = 500, grad_tol: float = 1e-10) -> Dict:
    """Certify the rig's rotation backbone against its per-view PnP measurement graph.

    Two passes. Pass 1 refines the BA rotations to a chordal critical point and scores
    every measurement edge by the *minimum* of its rotation residual at the BA
    configuration and at the refined solution (outlier only if inconsistent with both —
    see the inline rationale); edges beyond ``max(6 x median, 15 deg)`` are gross
    measurement outliers (measured on both real datasets: inlier edges cluster at
    0.6-2.3 deg median while wrong PnP poses sit at 80-179 deg — near-antipodal
    fisheye-PnP flips) and are **trimmed and reported**, never
    silently kept (they would contaminate the certified problem) nor silently dropped (the
    outlier list is a first-class diagnostic). Pass 2 re-refines on the cleaned graph and
    certifies it.

    Returns a dict: ``certified`` (bool, or ``None`` when no usable measurement graph
    exists), ``eta`` (scale-relative minimum eigenvalue of the certificate matrix — the
    trust number; ≥ −1e-6 passes), ``gap_bound`` (chordal suboptimality bound when not
    certified), ``grad_norm`` (stationarity actually reached — the certificate's
    precondition), ``d_cam_deg``/``d_frame_deg`` (max rotation distance between the BA
    solution and the certified chordal solution, split by node type: **cameras are the
    calibration output, so the wrong-basin verdict uses** ``d_cam_deg``; a large
    ``d_frame_deg`` with small ``d_cam_deg`` means a board-placement node was dragged by a
    bad measurement, not a wrong calibration), ``ba_consistent``
    (``d_cam_deg <= max(3 x median edge residual, 0.5 deg)``), ``resid_med_deg``,
    ``outlier_edges`` (list of ``(cam_id, (object_id, frame_id), residual_deg)``, worst
    first), ``n_outlier_edges``, ``n_components``, ``n_nodes``/``n_edges``, ``message``.

    Edge weights are each observation's corner count (normalized to mean 1) — a natural
    information proxy; the certificate then certifies exactly that weighted problem on the
    trimmed graph.
    """
    cams = [c for c in sorted(rig.cameras) if c in rig.T_c_g]
    keys = sorted(rig.object_poses)
    cam_idx = {c: i for i, c in enumerate(cams)}
    key_idx = {k: len(cams) + i for i, k in enumerate(keys)}
    n = len(cams) + len(keys)

    raw_edges = []
    for o in object_obs:
        key = (o.object_id, o.frame_id)
        if o.T_c_o is None or o.cam_id not in cam_idx or key not in key_idx:
            continue
        raw_edges.append((cam_idx[o.cam_id], key_idx[key],
                          np.asarray(o.T_c_o[:3, :3], float), float(len(o.point_rows)),
                          (o.cam_id, key)))
    if not raw_edges:
        return {"certified": None, "eta": float("nan"), "gap_bound": float("nan"),
                "grad_norm": float("nan"), "d_cam_deg": float("nan"),
                "d_frame_deg": float("nan"), "ba_consistent": None,
                "resid_med_deg": float("nan"), "outlier_edges": [], "n_outlier_edges": 0,
                "n_components": 0, "n_nodes": n, "n_edges": 0,
                "message": "no per-view PnP rotations available — certificate skipped"}
    w_mean = float(np.mean([e[3] for e in raw_edges]))
    edges = [(i, j, R, w / w_mean, ident) for i, j, R, w, ident in raw_edges]

    # stacked frame-from-world rotations from the BA solution (Z_c = R_c_g, Z_t = R_g_o^T).
    X = np.zeros((3 * n, 3))
    for c, i in cam_idx.items():
        X[3 * i:3 * i + 3] = rig.T_c_g[c][:3, :3]
    for k, i in key_idx.items():
        X[3 * i:3 * i + 3] = rig.object_poses[k][:3, :3].T

    def _solve(edge_set):
        """Refine + certify per connected component of ``edge_set``. Returns
        (eta, gap, grad, d_cam, d_frame, Rc_full, resid list aligned with edge_set)."""
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j, *_ in edge_set:
            parent[find(i)] = find(j)
        comp_of = {v: find(v) for v in range(n)}
        touched = sorted({comp_of[e[0]] for e in edge_set} |
                         {comp_of[e[1]] for e in edge_set})
        Rc_full = X.copy()
        eta_all, gap_all, grad_all = [], [], []
        d_cam, d_frame = 0.0, 0.0
        for comp in touched:
            nodes = [v for v in range(n) if comp_of[v] == comp]
            remap = {v: q for q, v in enumerate(nodes)}
            sub = [(remap[i], remap[j], R, w) for i, j, R, w, _id in edge_set
                   if comp_of[i] == comp]
            m = len(nodes)
            L = connection_laplacian(m, sub)
            R0 = np.concatenate([X[3 * v:3 * v + 3] for v in nodes], axis=0)
            Rc = refine_chordal(L, R0, max_iter=max_refine_iter, tol=grad_tol)
            grad_all.append(float(np.linalg.norm(chordal_grad(L, Rc))))
            _S, lam_min, _ev, gap = certificate(L, Rc)
            eta_all.append(lam_min / max(float(np.mean(np.diag(L))), 1e-12))
            gap_all.append(gap)
            for q, v in enumerate(nodes):
                Rc_full[3 * v:3 * v + 3] = Rc[3 * q:3 * q + 3]
                d = _rot_angle_deg(R0[3 * q:3 * q + 3].T @ Rc[3 * q:3 * q + 3])
                if v < len(cams):
                    d_cam = max(d_cam, d)
                else:
                    d_frame = max(d_frame, d)
        resids = [_rot_angle_deg(np.asarray(R).T @ Rc_full[3 * i:3 * i + 3]
                                 @ Rc_full[3 * j:3 * j + 3].T)
                  for i, j, R, _w, _id in edge_set]
        return (min(eta_all), max(gap_all), max(grad_all), d_cam, d_frame,
                len(touched), resids)

    # pass 1: find gross measurement outliers. The trim statistic is the MINIMUM of the
    # edge's residual at the BA configuration and at the refined chordal solution — an
    # edge is a gross outlier only if it is inconsistent with BOTH. This keeps a
    # wrong-basin camera's edges (small residual at the refined optimum — trimming them
    # would eat the wrong-basin evidence) and keeps the good edges of a frame node that a
    # bad sibling measurement dragged during refinement (small residual at the BA
    # configuration), while a truly wrong PnP pose fails both (measured: 92-179 deg
    # against 0.6-2.3 deg medians on both real datasets).
    *_p1, resids1 = _solve(edges)
    resids_ba = [_rot_angle_deg(np.asarray(R).T @ X[3 * i:3 * i + 3]
                                @ X[3 * j:3 * j + 3].T)
                 for i, j, R, _w, _id in edges]
    stat = [min(a, b) for a, b in zip(resids_ba, resids1)]
    med = float(np.median(stat))
    thresh = max(_TRIM_MED_FACTOR * med, _TRIM_FLOOR_DEG)
    outliers = sorted(((r, e[4]) for r, e in zip(stat, edges) if r > thresh),
                      reverse=True)
    kept = [e for r, e in zip(stat, edges) if r <= thresh]
    if not kept:   # pathological: everything trimmed — report, don't divide by zero
        return {"certified": False, "eta": float("nan"), "gap_bound": float("nan"),
                "grad_norm": float("nan"), "d_cam_deg": float("nan"),
                "d_frame_deg": float("nan"), "ba_consistent": None,
                "resid_med_deg": med,
                "outlier_edges": [(c, k, float(r)) for r, (c, k) in outliers],
                "n_outlier_edges": len(outliers), "n_components": 0, "n_nodes": n,
                "n_edges": len(edges),
                "message": "every measurement edge is a gross outlier — measurement graph "
                           "unusable; inspect the per-view PnP poses"}

    # pass 2: certify the cleaned graph.
    eta, gap_bound, grad_norm, d_cam, d_frame, n_comp, resids2 = _solve(kept)
    resid_med = float(np.median(resids2))
    certified = bool(eta >= -_ETA_TOL)
    # Cameras are the calibration output: the wrong-basin verdict compares their distance
    # from the certified optimum against the (robust, median) measurement wobble. Measured:
    # d tracks the residual scale ~1:1 under pure noise, while a planted wrong-basin camera
    # sits at ~57 deg against ~1 deg residuals; on real data a single 92-deg outlier PnP
    # measurement drags only its *frame* node (d_frame 26.8 deg, d_cam 1.8 deg) — which is
    # an input-quality finding, not a wrong calibration.
    ba_consistent = bool(d_cam <= max(3.0 * resid_med, 0.5))
    out_note = ""
    if outliers:
        worst = ", ".join(f"cam{c} frame{k[1]} {r:.0f} deg"
                          for r, (c, k) in outliers[:3])
        out_note = (f"; {len(outliers)} gross PnP measurement outlier(s) detected and "
                    f"excluded ({worst}{', …' if len(outliers) > 3 else ''}) — the robust "
                    "BA down-weighted these, but their per-view poses are wrong")
    if certified and ba_consistent:
        msg = ("rotation backbone CERTIFIED globally optimal for the weighted chordal "
               "rotation-averaging problem on this measurement graph, and the calibrated "
               "rotations agree with that optimum (certifies neither the PnP inputs, "
               f"translations, nor intrinsics){out_note}")
    elif certified:
        msg = (f"WRONG-BASIN WARNING: the global optimum of the rotation backbone was "
               f"found and certified, and the calibrated camera rotations are "
               f"{d_cam:.1f} deg away from it (median measurement residual "
               f"{resid_med:.2f} deg) — the BA likely converged to a wrong rotation "
               f"basin; re-run with a different init or inspect the per-view poses"
               f"{out_note}")
    else:
        msg = (f"rotation backbone NOT certified (eta={eta:.2e}); inconclusive — the "
               "solution may still be optimal (high measurement noise loosens the "
               f"relaxation); chordal suboptimality bounded by {gap_bound:.3g}{out_note}")
    return {"certified": certified, "ba_consistent": ba_consistent, "eta": float(eta),
           "gap_bound": float(gap_bound), "grad_norm": float(grad_norm),
           "d_cam_deg": float(d_cam), "d_frame_deg": float(d_frame),
           "resid_med_deg": resid_med,
           "outlier_edges": [(c, k, float(r)) for r, (c, k) in outliers],
           "n_outlier_edges": len(outliers), "n_components": n_comp, "n_nodes": n,
           "n_edges": len(kept), "message": msg}
