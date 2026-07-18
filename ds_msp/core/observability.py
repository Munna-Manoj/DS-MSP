"""Observability / conditioning diagnostics for a converged least-squares fit.

Generic linear-algebra + coverage math only (pure NumPy, no rig semantics): the rig layer
(:mod:`ds_msp.rig.audit`) labels columns and turns the raw findings into named messages.

The central quantity is the **equilibrated** normal matrix. The columns of a calibration
Jacobian mix physical units (px/rad for rotations, px/length for translations, px/px for
focals, px/1 for shape parameters), so the raw eigenvalues of ``H = JᵀWJ`` — and any
condition number built from them — are dominated by the arbitrary unit choice (meters vs
millimeters rescales translation eigenvalues by 1e6), not by geometric degeneracy. Van der
Sluis diagonal equilibration (Numer. Math. 14, 1969) fixes this: with
``D = diag(1/sqrt(diag H))``, the scaled matrix ``Ĥ = D H D`` has unit diagonal, its entries
are the weighted-Jacobian column correlations ``Ĥ_pq = cos∠(√W J_p, √W J_q) ∈ [-1, 1]``, its
eigenvalues are dimensionless and ≈1-referenced, and equilibrating to unit diagonal brings
the condition number within a factor of K of the best achievable by any diagonal scaling.
This is the same column scaling Ceres applies by default (``Solver::Options::jacobi_scaling``)
before its linear solves, and the conditioning discipline of Triggs et al., "Bundle
Adjustment — A Modern Synthesis" (2000). A near-null eigenvalue of ``Ĥ`` is a genuine
near-linear-dependence among parameter effects — a degeneracy — independent of units.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def equilibrate(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Van der Sluis-scale a symmetric PSD ``H`` to unit diagonal.

    Returns ``(H_hat, s)`` with ``H_hat = diag(s) H diag(s)`` symmetric with ``diag ≈ 1``
    and ``s = 1/sqrt(diag H + eps)``. A (near-)zero diagonal entry means the parameter has
    (near-)no residual response at all; the epsilon floor keeps ``s`` finite and such a
    column surfaces as a near-null eigenvalue rather than a division blow-up.
    """
    H = np.asarray(H, float)
    d = np.diag(H).copy()
    eps = max(float(np.max(d)), 1.0) * 1e-15
    # floor only the (near-)zero diagonals — leaving positive entries untouched keeps the
    # scaling exactly invariant to per-column unit changes (adding eps everywhere would
    # perturb rescaled columns differently and break that invariance).
    s = 1.0 / np.sqrt(np.where(d > eps, d, eps))
    Hh = H * s[:, None] * s[None, :]
    return 0.5 * (Hh + Hh.T), s


def eigen_weakness(H: np.ndarray, *, tau_rel: float = 1e-3, corr_thresh: float = 0.95,
                   participation: float = 0.9) -> Dict:
    """Equilibrate ``H``, eigen-decompose, and return weak directions + parameter couplings.

    A direction is **weak** when its eigenvalue of the unit-diagonal ``Ĥ`` falls below
    ``tau_rel * median(eigenvalues)``. Because ``Ĥ``'s eigenvalues are dimensionless and
    ≈1-referenced (well-observed directions sit near 1, genuine near-null directions collapse
    to ~1e-8..1e-12), the threshold sits inside a wide empty gap and needs no per-problem
    tuning; ``gap`` reports the measured separation so a caller can verify that.

    Returns a dict with:

    - ``scale`` — the (K,) equilibration scales ``1/sqrt(diag H)``.
    - ``cond_raw`` / ``cond`` — condition numbers of raw ``H`` and of ``Ĥ`` (thresholds and
      naming must use ``Ĥ`` only; ``cond_raw`` is reported to expose the unit-mix artefact).
    - ``eigvals`` — ascending eigenvalues of ``Ĥ``.
    - ``n_weak`` and ``weak`` — a list of ``{index, eigval, ratio, energy (K,),
      participating [col, ...]}`` where ``energy = v**2`` is the per-column participation of
      the weak eigenvector (dimensionless because ``Ĥ`` is equilibrated) and
      ``participating`` is the smallest column set whose cumulative energy reaches
      ``participation``.
    - ``gap`` — ``eigvals[n_weak] / eigvals[n_weak-1]``, the measured separation between the
      flagged set and the rest (``inf`` when nothing is flagged).
    - ``pairs`` — ``[(i, j, corr), ...]`` for every off-diagonal ``|Ĥ_ij| > corr_thresh``:
      (anti)parallel scaled Jacobian columns, i.e. parameters whose effects on the residual
      are near-indistinguishable. Signed: ``+`` move together, ``-`` trade off.
    """
    Hh, s = equilibrate(H)
    eigvals, eigvecs = np.linalg.eigh(Hh)
    tiny = 1e-300
    med = float(np.median(eigvals))
    thresh = tau_rel * med
    weak_idx = np.where(eigvals < thresh)[0]

    weak: List[Dict] = []
    for i in weak_idx:
        v = eigvecs[:, i]
        energy = v * v
        order = np.argsort(energy)[::-1]
        csum = np.cumsum(energy[order])
        n_part = int(np.searchsorted(csum, participation) + 1)
        weak.append({
            "index": int(i),
            "eigval": float(eigvals[i]),
            "ratio": float(eigvals[i] / max(med, tiny)),
            "energy": energy,
            "participating": [int(j) for j in order[:n_part]],
        })

    n_weak = len(weak)
    if n_weak and n_weak < len(eigvals):
        gap = float(eigvals[n_weak] / max(eigvals[n_weak - 1], tiny))
    else:
        gap = float("inf")

    iu, ju = np.triu_indices_from(Hh, k=1)
    strong = np.abs(Hh[iu, ju]) > corr_thresh
    pairs = [(int(i), int(j), float(Hh[i, j]))
             for i, j in zip(iu[strong], ju[strong])]

    eig_raw = np.linalg.eigvalsh(0.5 * (np.asarray(H, float) + np.asarray(H, float).T))
    cond_raw = float(eig_raw[-1] / eig_raw[0]) if eig_raw[0] > tiny else float("inf")
    cond = float(eigvals[-1] / eigvals[0]) if eigvals[0] > tiny else float("inf")
    return {"scale": s, "cond_raw": cond_raw, "cond": cond, "eigvals": eigvals,
           "n_weak": n_weak, "weak": weak, "gap": gap, "pairs": pairs}


def radial_occupancy(uv: np.ndarray, center: np.ndarray, R: Optional[float] = None,
                     n_bins: int = 5) -> Tuple[np.ndarray, float]:
    """Radial coverage of image points: equal-area annulus occupancy + periphery fraction.

    ``R`` is the normalising radius (the caller should pass the max distance from ``center``
    to the image corners; defaults to ``max(r)`` of the data itself). Annulus edges
    ``ρ_k = R·sqrt(k/B)`` have equal area, so occupancy is flat under uniform coverage and an
    empty outer annulus stands out. ``periphery_frac`` is the fraction of points beyond
    ``0.8·R`` — the outer-FOV band that constrains a fisheye model's shape parameters.
    """
    uv = np.asarray(uv, float)
    r = np.linalg.norm(uv - np.asarray(center, float)[None, :], axis=1)
    if R is None:
        R = float(np.max(r)) if r.size else 1.0
    R = max(float(R), 1e-12)
    edges = R * np.sqrt(np.arange(n_bins + 1) / n_bins)
    occ = np.histogram(np.minimum(r, R), bins=edges)[0] / max(r.size, 1)
    periphery_frac = float(np.mean(r > 0.8 * R)) if r.size else 0.0
    return occ, periphery_frac


def orientation_spread(normals: np.ndarray) -> Tuple[np.ndarray, float]:
    """Spread of a set of unit plane normals via the orientation tensor.

    ``T = mean(n nᵀ)`` (a normal and its negative are the same plane, so a vector mean
    would be wrong). Returns ``(eigvals descending (3,), tilt_diversity = 1 - λ1)``:
    ``0`` when every view is frontoparallel-identical, → ``2/3`` for isotropic orientations.
    """
    n = np.asarray(normals, float)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    T = (n[:, :, None] * n[:, None, :]).mean(axis=0)
    eigvals = np.linalg.eigvalsh(T)[::-1]
    return eigvals, float(1.0 - eigvals[0])
