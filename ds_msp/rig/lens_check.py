"""Per-camera lens checks after a rig calibration: is each camera's lens actually known?

Three checks, each reported per camera:

1. **Lens uncertainty in pixels.** The classical covariance ``σ² S⁻¹`` of the shared
   parameters (:func:`bundle.bread_factor`, mrcal's ``σ² (JᵀJ)⁻¹``) is propagated to the
   projection of a 24 x 18 grid of pixels: ``σ_u = σ ‖G_u L_c‖`` with ``G_u`` the projection
   Jacobian in that camera's lens parameters after the best camera rotation is taken out
   (:func:`_realigned`, as mrcal does). Median and 95th percentile over the grid. The
   residual scale ``σ`` comes from about ``G`` independent frames for a camera that sees ``G``
   frames, so the median is widened by ``t_{0.975, G-1} / z_{0.975}`` (unbounded at one frame).
2. **Lens parameters at an optimizer bound** (``param_bounds``): the estimate there is not a
   free minimum, and its covariance does not describe it.
3. **Split cross-validation** (mrcal's ``cross-validation`` check): the rig is re-solved on two
   halves of the frames, each made of alternate contiguous blocks in capture order, warm-started
   from the full solution, and each camera's two lenses are compared after the best-fitting
   rotation between them (mrcal ``projection_diff`` at infinity), over the grid cells both
   halves observed. Blocks, not even and odd frames, because neighbouring video frames share
   their errors. Each half has half the data, so its error
   is ``√2`` times the full-data error and the two independent errors subtract: the difference is
   about twice the full-data lens error. Half the median difference is therefore the split's
   estimate of the full-data lens error. Unlike the covariance it assumes no model and no
   linearity, so it also exposes model misfit, per-view bias and flat, nonlinear valleys.

A camera fails a check when the widened median sigma, or the split's error estimate, exceeds
``limit_px`` (the caller's ``max_lens_sigma_px``, by default the configured ``noise_bound``,
else 1 px). That limit is the user's own pixel tolerance; nothing here is tuned per dataset.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np

from . import bundle
from .types import ObjectObs, RigState

GRID = (24, 18)


def frames_per_camera(rig: RigState, object_obs: List[ObjectObs]) -> Dict[int, int]:
    """Distinct frames each camera observes with a solved object pose."""
    seen: Dict[int, set] = {}
    for o in object_obs:
        if (o.object_id, o.frame_id) in rig.object_poses:
            seen.setdefault(o.cam_id, set()).add(o.frame_id)
    return {c: len(seen.get(c, ())) for c in rig.cameras}


def t_widening(frames: int) -> Optional[float]:
    """``t_{0.975, G-1} / z_{0.975}`` for ``G`` frames; ``None`` (unbounded) below two."""
    if frames < 2:
        return None
    from scipy import stats
    return float(stats.t.ppf(0.975, frames - 1) / stats.norm.ppf(0.975))


def params_at_bound(model, rtol: float = 1e-6) -> List[str]:
    """Lens parameters within ``rtol`` of the span of a finite optimizer bound."""
    lo, hi = type(model).param_bounds()
    names = list(type(model).param_names)
    x = np.array([float(getattr(model, p)) for p in names])
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    span = np.where(np.isfinite(hi - lo), hi - lo, np.maximum(1.0, np.abs(x)))
    out = []
    for i, p in enumerate(names):
        if (np.isfinite(lo[i]) and x[i] - lo[i] <= rtol * span[i]) or \
           (np.isfinite(hi[i]) and hi[i] - x[i] <= rtol * span[i]):
            out.append(p)
    return out


def _grid(w: float, h: float, n: int = GRID[0], m: int = GRID[1]) -> np.ndarray:
    u, v = np.meshgrid((np.arange(n) + 0.5) * w / n, (np.arange(m) + 0.5) * h / m)
    return np.c_[u.ravel(), v.ravel()]


def _rays(model, uv: np.ndarray):
    X, ok = model.unproject(uv)
    X = np.asarray(X, float).reshape(-1, 3)
    ok = np.asarray(ok, bool).ravel() & np.isfinite(X).all(axis=1)
    return X, ok


def _observed_cells(rig: RigState, object_obs: List[ObjectObs], cam: int,
                    w: float, h: float, n: int = GRID[0], m: int = GRID[1]) -> np.ndarray:
    """Mask over the grid of cells containing at least one observed corner of ``cam``."""
    mask = np.zeros(n * m, bool)
    for o in object_obs:
        if o.cam_id != cam:
            continue
        uv = np.asarray(o.pts_2d, float)
        i = np.clip((uv[:, 0] / w * n).astype(int), 0, n - 1)
        j = np.clip((uv[:, 1] / h * m).astype(int), 0, m - 1)
        mask[j * n + i] = True
    return mask


def _realigned(G: np.ndarray, Jx: np.ndarray, X: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Lens Jacobians ``G (n, 2, P)`` with the part a camera rotation absorbs removed.

    A lens change that a small rotation of the camera undoes (a principal-point shift against a
    pan, for one) moves no projected ray relative to the others, and the rig's extrinsics take
    it up. mrcal therefore reports projection uncertainty after the best rigid realignment
    (``projection_uncertainty``; at infinity only the rotation matters). Linearised: a parameter
    change ``δθ`` moves pixel ``u`` by ``G_u δθ``, the best rotation over the ``fit`` pixels is
    ``δω = −(Σ R_vᵀR_v)⁻¹ Σ R_vᵀ G_v δθ`` with ``R_u = ∂q/∂X · (−[X_u]×)``, and the realigned
    Jacobian is ``G_u − R_u (Σ R_vᵀR_v)⁻¹ Σ R_vᵀ G_v``. Measured on synthetic 3-camera rigs
    (0.3 px noise, 25 frames): the median over the image went from 3.6-4.5 px without it to
    1.1-1.6 px with it, against a true lens error of 0.7-2.3 px."""
    if fit.sum() < 3:
        fit = np.ones(len(X), bool)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    skew = np.zeros((len(X), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 2] = -z, y, -x
    skew[:, 1, 0], skew[:, 2, 0], skew[:, 2, 1] = z, -y, x
    R = -np.einsum("nij,njk->nik", Jx, skew)                    # ∂q/∂ω, (n, 2, 3)
    A = np.einsum("nij,nik->jk", R[fit], R[fit])
    B = np.einsum("nij,nik->jk", R[fit], G[fit])
    try:
        W = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return G
    return G - np.einsum("nij,jk->nik", R, W)


#: Largest share of a camera's (realigned, equilibrated) lens Jacobian that may lie along the
#: bread's null directions before its lens is called unbounded. Eigenvectors are accurate to
#: about eps / gap, so an unrelated camera's share is ~1e-10 or less; a lens the null direction
#: moves has a share of order one.
_NULL_RTOL = 1e-6


def _null_share(G: np.ndarray, N_c: np.ndarray, BD_c: np.ndarray) -> float:
    """Largest over the grid of ``‖G_u N_c‖ / ‖G_u BD_c‖``: the part of this camera's lens
    projection that lies along unidentified directions."""
    if N_c.shape[1] == 0 or len(G) == 0:
        return 0.0
    GN = np.einsum("nip,pk->nik", G, N_c)
    GB = np.einsum("nip,pk->nik", G, BD_c)
    num = np.sqrt(np.einsum("nik,nik->n", GN, GN))
    den = np.sqrt(np.einsum("nik,nik->n", GB, GB))
    return float(np.max(num / np.maximum(den, 1e-300)))


def _json(x):
    return None if x is None or not np.isfinite(x) else float(x)


def lens_uncertainty(rig: RigState, object_obs: List[ObjectObs], unc: Dict, *,
                     limit_px: float) -> Dict:
    """Checks 1 and 2 from a :func:`bundle.shared_covariance` result ``unc`` (computed with
    free intrinsics). JSON-safe; ``cameras`` is keyed by ``str(cam_id)``."""
    fac = bundle.bread_factor(unc)
    sigma = float(unc.get("sigma_px", float("nan")))
    frames = frames_per_camera(rig, object_obs)
    cams = {}
    for c in sorted(rig.cameras):
        m = rig.cameras[c]
        ic = unc["intr_col"].get(c)
        if ic is None:
            continue
        P = len(type(m).param_names)
        G_c = frames.get(c, 0)
        tf = t_widening(G_c)
        bound = params_at_bound(m)
        med = p95 = med_obs = float("inf")
        null_share = None
        if fac is not None and np.isfinite(sigma):
            w, h = rig.img_size[c]
            X, ok = _rays(m, _grid(w, h))
            observed = _observed_cells(rig, object_obs, c, w, h)[ok]
            if ok.any():
                _q, Jx, Gm, val = m.project_jacobian(X[ok])
                val = np.asarray(val, bool).ravel()
                Gm = _realigned(np.asarray(Gm, float)[val], np.asarray(Jx, float)[val],
                                X[ok][val], observed[val])
                null_share = _null_share(Gm, fac["N"][ic:ic + P], fac["BD"][ic:ic + P])
                GL = np.einsum("nip,pk->nik", Gm, fac["L"][ic:ic + P])
                s_u = sigma * np.sqrt(np.einsum("nik,nik->n", GL, GL))
                if s_u.size and null_share <= _NULL_RTOL:
                    med, p95 = float(np.median(s_u)), float(np.percentile(s_u, 95))
                    so = s_u[observed[val]]
                    med_obs = float(np.median(so)) if so.size else med
        wid = med * tf if tf is not None else float("inf")
        reasons = []
        if not np.isfinite(wid) or wid > limit_px:
            reasons.append("sigma")
        if bound:
            reasons.append("at_bound")
        cams[str(c)] = {"frames": G_c, "sigma_median_px": _json(med), "sigma_p95_px": _json(p95),
                        "sigma_median_observed_px": _json(med_obs),
                        "t_factor": tf, "sigma_median_widened_px": _json(wid),
                        "null_share": _json(null_share) if null_share is not None else None,
                        "params_at_bound": bound, "ok": not reasons, "reasons": reasons}
    n_null = 0 if fac is None else int(fac["N"].shape[1])
    return {"method": "sigma^2 S^-1 (bread) by square-root propagation, 24x18 pixel grid, best "
                      "camera rotation removed; median widened by t(0.975, frames-1)/z(0.975)",
            "sigma_px": _json(sigma), "factor_ok": fac is not None, "n_null_directions": n_null,
            "limit_px": float(limit_px),
            "cameras": cams, "failed": [c for c, v in cams.items() if not v["ok"]]}


def _fit_rotation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rotation ``R`` minimising ``Σ ‖R a_i − b_i‖²`` (Kabsch)."""
    U, _s, Vt = np.linalg.svd(B.T @ A)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def projection_diff(ma, mb, uv: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Pixel difference between lens ``ma`` and ``mb`` at ``uv`` after the rotation that best
    maps ``ma``'s rays onto ``mb``'s over the pixels in ``fit`` (mrcal ``projection_diff`` at
    infinity). Pixels either model cannot unproject/project give NaN."""
    Xa, oka = _rays(ma, uv)
    Xb, okb = _rays(mb, uv)
    both = oka & okb
    use = both & fit
    if use.sum() < 3:
        use = both
    out = np.full(len(uv), np.nan)
    if use.sum() < 3:
        return out
    R = _fit_rotation(Xa[use], Xb[use])
    q, v = mb.project((R @ Xa[oka].T).T)
    v = np.asarray(v, bool).ravel()
    d = np.linalg.norm(np.asarray(q, float) - uv[oka], axis=1)
    d[~v] = np.nan
    out[oka] = d
    return out


SPLIT_BLOCKS = 8
_SPLIT_METHOD = (f"{SPLIT_BLOCKS} contiguous blocks of frames in capture order, alternate blocks "
                 "to each half; warm-started joint BA per half; lens difference after the best "
                 "rotation (mrcal projection_diff at infinity) over the grid cells both halves "
                 "observed")


def split_halves(frames: List[int], blocks: int = SPLIT_BLOCKS) -> List[set]:
    """Cut ``frames`` (sorted, capture order) into ``blocks`` contiguous blocks and give the
    even-numbered blocks to one half and the odd-numbered blocks to the other."""
    n = len(frames)
    k = max(1, min(blocks, n))
    idx = [i * k // n for i in range(n)]
    return [{f for f, b in zip(frames, idx) if b % 2 == h} for h in (0, 1)]


def crossval_lens(rig: RigState, object_obs: List[ObjectObs], *, structure_mode: str,
                  limit_px: float, lens: Optional[Dict] = None, min_frames: int = 3) -> Dict:
    """Check 3: re-solve on two halves of the frames and compare each camera's two lenses over
    the grid cells both halves observed.

    The halves are alternate contiguous blocks (:func:`split_halves`), not even and odd frames:
    frames taken from video are near-duplicates of their neighbours, so even/odd halves share
    their errors and agree too well. On the 8 TUM-VI calibration recordings, even/odd gave 0.001
    to 0.014 px where the lenses of separate recordings differ by 0.27 to 0.35 px; 8 blocks gave
    0.055 to 0.34 px (4 blocks 0.030 to 0.44, 16 blocks 0.024 to 0.30). One block per half
    (first and second half) would leave a camera that saw the target only early in the capture
    out of one half, as on rigs calibrated by walking the target past the cameras.
    ``frame_id`` order is taken as capture order.

    Cameras that see fewer than ``min_frames`` frames in either half are reported as not
    applicable (check 1 already covers them). ``lens`` (a :func:`lens_uncertainty` result) adds
    the expected difference, ``2 x`` the full-data median sigma, for reference."""
    frames = sorted({o.frame_id for o in object_obs
                     if (o.object_id, o.frame_id) in rig.object_poses})
    halves = split_halves(frames)
    multi_board = any(len(o.board_ids) > 1 for o in rig.objects.values())
    refine_boards = multi_board and structure_mode in ("rigid", "bend")
    solved, statuses, counts = [], [], []
    for half in halves:
        obs_h = [o for o in object_obs if o.frame_id in half]
        counts.append(frames_per_camera(rig, obs_h))
        if counts[-1].get(rig.ref_cam_id, 0) == 0:
            return {"method": _SPLIT_METHOD, "applicable": False,
                    "reason": "the reference camera has no frames in one half",
                    "limit_px": float(limit_px), "cameras": {}, "failed": []}
        rig_h = copy.copy(rig)
        rig_h.object_poses = {k: v for k, v in rig.object_poses.items() if k[1] in half}
        rig_h.cameras = {c: copy.deepcopy(m) for c, m in rig.cameras.items()}
        rig_h.T_c_g = dict(rig.T_c_g)
        log: List[Dict] = []
        rig_h = bundle.refine(rig_h, obs_h, fix_intrinsics=False, robust_kernel="cauchy",
                              robust_scale="auto", refine_boards=refine_boards,
                              bend_boards=refine_boards and structure_mode == "bend", log=log)
        solved.append(rig_h)
        statuses.append(log[-1]["status"] if log else None)
    cams = {}
    for c in sorted(rig.cameras):
        na, nb = counts[0].get(c, 0), counts[1].get(c, 0)
        row: Dict[str, object] = {"frames_halves": [na, nb]}
        if min(na, nb) < min_frames:
            row.update(applicable=False, ok=True)
            cams[str(c)] = row
            continue
        w, h = rig.img_size[c]
        uv = _grid(w, h)
        fit = (_observed_cells(rig, [o for o in object_obs if o.frame_id in halves[0]], c, w, h)
               & _observed_cells(rig, [o for o in object_obs if o.frame_id in halves[1]], c, w, h))
        if fit.sum() < 3:                       # halves saw disjoint parts of the image
            fit = _observed_cells(rig, object_obs, c, w, h)
        d = projection_diff(solved[0].cameras[c], solved[1].cameras[c], uv, fit)
        obs_d = d[fit & np.isfinite(d)]
        all_d = d[np.isfinite(d)]
        med = float(np.median(obs_d)) if obs_d.size else float("inf")
        exp = None
        if lens is not None:
            s = lens["cameras"].get(str(c), {}).get("sigma_median_observed_px")
            exp = None if s is None else 2.0 * s
        row.update(applicable=True,
                   error_estimate_px=_json(med / 2.0),
                   diff_median_observed_px=_json(med),
                   diff_p95_observed_px=_json(float(np.percentile(obs_d, 95))) if obs_d.size else None,
                   diff_median_image_px=_json(float(np.median(all_d))) if all_d.size else None,
                   expected_from_noise_px=_json(exp) if exp is not None else None,
                   ratio_to_expected=_json(med / exp) if exp else None,
                   ok=bool(np.isfinite(med) and med / 2.0 <= limit_px))
        cams[str(c)] = row
    return {"method": _SPLIT_METHOD, "applicable": True, "limit_px": float(limit_px), "half_status": statuses,
            "cameras": cams,
            "failed": [c for c, v in cams.items() if v.get("applicable") and not v["ok"]]}
