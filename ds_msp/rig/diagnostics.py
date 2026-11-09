"""Per-corner and per-board diagnostics of a solved rig, written for inspection tools.

The acceptance file holds verdicts and summaries; this module keeps the evidence behind them,
which image-space views need and summaries cannot carry:

* :func:`corner_residuals` -- one row per observed corner: where it was detected, the signed
  residual ``q_pred - q_obs`` (px), whether it projects validly, and whether it lies within the
  GNC-TLS inlier band (the same rule :func:`bundle.camera_coverage` counts). Residual arrows,
  regional bias and worst-frame lists are all computed from this table.
* :func:`outlier_summary` -- per camera, the share of validly projected corners outside the
  band. mrcal's guidance is that more than about 3% "should raise questions"; no limit is
  applied here.
* :func:`board_flatness` -- per board of each fused target, the peak-to-valley and RMS distance
  of the solved corners from their best-fitting plane, in target units. Zero for a rigidly
  refined target; under ``structure_mode: bend`` it is the part of the fitted bow that a tilted
  plane cannot absorb, which is what a straightedge laid on the board would show.
* :func:`write_diagnostics` -- saves the residual table and the lens sigma grids
  (:func:`ds_msp.rig.lens_check.lens_sigma_grids`) as compressed ``.npz`` files.

Nothing here changes the solve: every quantity is read from the final rig state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .types import ObjectObs, RigState

#: Columns of :func:`corner_residuals`, in order.
RESIDUAL_COLUMNS = ("cam", "frame", "object", "row", "board", "corner", "u", "v", "dx", "dy",
                    "valid", "within_band")


def corner_residuals(rig: RigState, object_obs: List[ObjectObs],
                     band_px: float) -> Dict[str, np.ndarray]:
    """Signed residual of every observed corner at the final estimate.

    Returns a dict of equal-length arrays keyed by :data:`RESIDUAL_COLUMNS`: ``cam``,
    ``frame``, ``object``, ``row`` (row into the object's ``pts_3d``), ``board`` and ``corner``
    (board-local id), the detection ``u, v`` (px), ``dx, dy = q_pred - q_obs`` (px; NaN when the
    corner does not project validly or its frame has no pose), ``valid`` and ``within_band``
    (``valid`` and ``|r| <= band_px``). Observations hard-dropped before the solve
    (``rig.rejected``) are not in ``object_obs`` and so are not listed."""
    cols: Dict[str, list] = {k: [] for k in RESIDUAL_COLUMNS}
    for o in object_obs:
        k = len(o.point_rows)
        if not k:
            continue
        rows = np.asarray(o.point_rows, int)
        uv_obs = np.asarray(o.pts_2d, float).reshape(-1, 2)
        obj = rig.objects.get(o.object_id)
        if obj is not None:
            bc = np.asarray(obj.pts_obj_2_board, int)[rows]
        else:
            bc = np.full((k, 2), -1, int)
        d = np.full((k, 2), np.nan)
        valid = np.zeros(k, bool)
        key = (o.object_id, o.frame_id)
        if obj is not None and o.cam_id in rig.cameras and key in rig.object_poses:
            T = rig.T_c_g[o.cam_id] @ rig.object_poses[key]
            Xc = obj.pts_3d[rows] @ T[:3, :3].T + T[:3, 3]
            uv, ok = rig.cameras[o.cam_id].project(Xc)
            valid = np.asarray(ok, bool).ravel() & np.isfinite(uv).all(axis=1)
            d[valid] = np.asarray(uv, float)[valid] - uv_obs[valid]
        within = valid & (np.hypot(d[:, 0], d[:, 1]) <= band_px)
        cols["cam"].append(np.full(k, o.cam_id))
        cols["frame"].append(np.full(k, o.frame_id))
        cols["object"].append(np.full(k, o.object_id))
        cols["row"].append(rows)
        cols["board"].append(bc[:, 0])
        cols["corner"].append(bc[:, 1])
        cols["u"].append(uv_obs[:, 0])
        cols["v"].append(uv_obs[:, 1])
        cols["dx"].append(d[:, 0])
        cols["dy"].append(d[:, 1])
        cols["valid"].append(valid)
        cols["within_band"].append(within)
    types = {"cam": np.int32, "frame": np.int32, "object": np.int32, "row": np.int32,
             "board": np.int32, "corner": np.int32, "u": np.float64, "v": np.float64,
             "dx": np.float64, "dy": np.float64, "valid": bool, "within_band": bool}
    return {c: (np.concatenate(cols[c]).astype(types[c]) if cols[c]
                else np.zeros(0, types[c])) for c in RESIDUAL_COLUMNS}


def outlier_summary(res: Dict[str, np.ndarray], band_px: float) -> Dict[str, dict]:
    """Per camera (``str(cam)``): corners observed, validly projected and within the band,
    and ``outlier_frac`` = share of validly projected corners outside the band (``None`` when
    none project). JSON-safe."""
    out: Dict[str, dict] = {}
    for c in np.unique(res["cam"]):
        m = res["cam"] == c
        n_valid = int(res["valid"][m].sum())
        n_within = int(res["within_band"][m].sum())
        out[str(int(c))] = {"n_observed": int(m.sum()), "n_valid": n_valid,
                            "n_within": n_within, "band_px": float(band_px),
                            "outlier_frac": (1.0 - n_within / n_valid) if n_valid else None}
    return out


def board_flatness(rig: RigState) -> Dict[str, Dict[str, dict]]:
    """Per fused object and board: ``n_corners``, ``sag_ptp`` and ``rms`` -- the peak-to-valley
    and RMS distance of the board's solved corners from their least-squares plane, in target
    units (those of ``square_size``). Keys are ``str(object_id)`` then ``str(board_id)``.
    ``None`` for a board with fewer than four corners. Measures the solved geometry, so it is
    zero up to rounding unless the target was bent (``structure_mode: bend``) or freed."""
    out: Dict[str, Dict[str, dict]] = {}
    for oid, obj in sorted(rig.objects.items()):
        pts = np.asarray(obj.pts_3d, float)
        boards = np.asarray(obj.pts_obj_2_board, int)[:, 0]
        per: Dict[str, dict] = {}
        for b in sorted(set(boards.tolist())):
            P = pts[boards == b]
            if len(P) < 4:
                per[str(b)] = {"n_corners": int(len(P)), "sag_ptp": None, "rms": None}
                continue
            q = P - P.mean(0)
            dist = q @ np.linalg.svd(q, full_matrices=False)[2][2]
            per[str(b)] = {"n_corners": int(len(P)), "sag_ptp": float(np.ptp(dist)),
                           "rms": float(np.sqrt(np.mean(dist ** 2)))}
        out[str(oid)] = per
    return out


def write_diagnostics(save_dir, res: Dict[str, np.ndarray],
                      lens_grids: Optional[Dict[int, Dict[str, np.ndarray]]] = None) -> Dict[str, str]:
    """Write ``diagnostics/residuals.npz`` (the :func:`corner_residuals` table) and, when given,
    ``diagnostics/lens_sigma.npz`` (per camera ``cam<id>_<field>`` arrays from
    :func:`ds_msp.rig.lens_check.lens_sigma_grids`) under ``save_dir``. Returns the written
    paths relative to ``save_dir``."""
    d = Path(save_dir) / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    np.savez_compressed(d / "residuals.npz", **res)
    paths["residuals"] = "diagnostics/residuals.npz"
    if lens_grids:
        flat = {f"cam{c}_{k}": v for c, g in sorted(lens_grids.items()) for k, v in g.items()}
        np.savez_compressed(d / "lens_sigma.npz", **flat)
        paths["lens_sigma"] = "diagnostics/lens_sigma.npz"
    return paths
