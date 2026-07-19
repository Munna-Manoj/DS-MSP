r"""In-house manifold Levenberg–Marquardt solver (pure NumPy).

Why this exists
---------------
``scipy.optimize.least_squares`` is a *black box*: it owns the iterate loop, so it
cannot **re-base** a manifold parameterization. Optimizing a rotation as a local
perturbation ``δω`` only pays off if, after each accepted step, you fold ``δω`` back
into the base pose and reset ``δω → 0`` — keeping it small and the right-Jacobian
``J_r(δω) ≈ I``. SciPy keeps the base fixed for the whole solve, so ``δω`` drifts
large (back toward the ``‖δω‖ = π`` singularity) and you pay the manifold's extra
Jacobian cost for none of its benefit. That is exactly why the SciPy manifold
calibration came out *slower and less robust* than flat axis-angle.

This solver owns the loop and re-bases every iteration by construction: the caller's
``retract`` produces a new *base* state and we always re-linearize there, so the
increment ``δ`` is, definitionally, the small step away from the current base.

What it does (the parts that make Lie fast *and* robust)
--------------------------------------------------------
* **Gauss–Newton / Levenberg–Marquardt** on the normal equations, solved directly by
  Cholesky (no generic dense TRF) — the system is ``K×K`` in the tangent dimension.
* **Manifold re-basing** via the caller-supplied ``retract`` — the increment never
  drifts toward the chart singularity.
* **IRLS robust kernels** (Huber / pseudo-Huber / Cauchy / Geman-McClure) folded into
  the normal equations, with optional **MAD auto-scale** (re-estimated each iteration)
  and **graduated non-convexity** to escape spurious basins. See :mod:`.robust`.
* **Marquardt damping** ``λ·diag(H)`` (unit-aware) with a floor so flat columns are
  still damped, plus a **Nielsen gain-ratio** schedule.
* **Cholesky with escalating, scale-aware jitter** for rank-deficient / ill-conditioned
  Hessians — never throws, degrades gracefully to a damped identity.
* **Accept/reject** on the robust cost so the iterate can never diverge.

The caller plugs in the geometry through three callables; the solver is geometry-blind,
so the same loop drives two-view BA and multi-image calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .robust import (
    auto_kernel_scale, gnc_scale, gnc_tls_mu_init, gnc_tls_weight,
    robust_cost, robust_weight, studentized_scale_factors,
    GNC_TLS_CONTINUATION, VALID_KERNELS,
)


@dataclass
class OptResult:
    """Outcome of :func:`lm_solve`. ``state`` is the refined state object the caller
    passed in / built via ``retract``; the rest are diagnostics."""
    state: object
    cost: float
    rms: float
    iterations: int
    success: bool
    converged: bool
    final_lambda: float
    #: Per-block inlier weights from a graduated solve (:func:`gnc_tls_solve`); ``None`` for
    #: the plain :func:`lm_solve` path. ≈binary after GNC-TLS converges (1 inlier / 0 outlier).
    weights: Optional[np.ndarray] = None


def _update_lambda(accepted: bool, lam: float, nu: float, *, schedule: str,
                   pred: float, actual: float, lam_min: float,
                   lam_max: float) -> tuple[float, float]:
    """One Levenberg–Marquardt damping update, shared by the dense and Schur loops.

    ``classic``: ÷3 on accept, ×3 on reject. ``nielsen``: gain ratio
    ``ρ = actual / pred`` → ``λ·max(⅓, 1−(2ρ−1)³)`` on accept (``ν←2``),
    ``λ·ν`` on reject (``ν←2ν``). Returns ``(lam, nu)``.
    """
    if accepted:
        if schedule == "nielsen":
            rho = actual / pred if pred > 1e-30 else 1.0
            lam = max(lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), lam_min)
            nu = 2.0
        else:
            lam = max(lam / 3.0, lam_min)
    else:
        if schedule == "nielsen":
            lam = min(lam * nu, lam_max)
            nu *= 2.0
        else:
            lam = min(lam * 3.0, lam_max)
    return lam, nu


def _safe_inv(M: np.ndarray) -> np.ndarray:
    """Inverse of a small SPD block with scale-aware jitter if near-singular.

    NOTE (matrix-calculus study, 2026-07): instrumentation showed a single rig
    calibration LU-inverts thousands of near-singular blocks here (Cholesky-
    diagonal rcond estimates down to 1e-26). Replacing this with a regularized
    (jittered-Cholesky) or pseudo-inverse changes end-to-end calibration
    results enough to shift tests/rig/test_param_pose.py::
    test_model_of_choice_clean[kb-kb] from 'worst pose' <1.0% to ~1.27%: the
    huge-norm LU steps are rejected by lm_solve/schur_lm's cost gate (after
    which a larger lambda re-conditions the block), whereas plausible
    regularized steps get *accepted* into a slightly different basin. The
    historical behavior is therefore load-bearing; changing it needs a
    maintainer decision with recalibrated end-to-end thresholds, not a
    drive-by swap.
    """
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        scale = max(float(np.trace(M)) / M.shape[0], 1.0)
        return np.linalg.inv(M + 1e-8 * scale * np.eye(M.shape[0]))


def _cho_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``(L L^T) x = b`` from a lower Cholesky factor, NumPy-only.

    The math-foundation layer must stay free of scipy (import-linter contract /
    ``tests/contract/test_independence.py``), so the two triangular substitutions go
    through ``np.linalg.solve`` on the factor instead of ``scipy.linalg.solve_triangular``.
    Solving an already-triangular system via LAPACK's generic driver is numerically the
    same backward-stable substitution; the extra factorization bookkeeping is negligible
    at this solver's dense-path sizes."""
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


def _solve_damped(H: np.ndarray, g: np.ndarray, lam: float,
                  D: np.ndarray) -> np.ndarray:
    """Solve ``(H + λ·diag(D)) δ = -g`` by Cholesky, with escalating scale-aware
    jitter if the damped Hessian is not positive-definite (rank-deficient pose,
    near-degenerate geometry). Falls back to a scaled identity as a last resort."""
    K = H.shape[0]
    A = H + lam * np.diag(D)
    try:
        L = np.linalg.cholesky(A)
        return _cho_solve(L, -g)
    except np.linalg.LinAlgError:
        pass
    # A fixed +εI is negligible against a ~1e6-trace pixel² Hessian, so tie the
    # jitter to tr(H)/K and grow it ×100 per retry.
    scale = max(float(np.trace(H)) / K, 1.0)
    fb = 1.0
    for _ in range(4):
        try:
            jit = (fb + 1e-8 * fb * scale)
            L = np.linalg.cholesky(A + jit * np.eye(K))
            return _cho_solve(L, -g)
        except np.linalg.LinAlgError:
            fb *= 100.0
    # Last resort: pure scaled-identity step (a tiny gradient-descent move).
    return -g / (scale)


def lm_solve(
    state0: object,
    residual: Callable[[object], np.ndarray],
    jacobian: Callable[[object], np.ndarray],
    retract: Callable[[object, np.ndarray], object],
    *,
    block: int = 2,
    max_iter: int = 50,
    tol: float = 1e-9,
    lam_init: float = 1e-3,
    lam_min: float = 1e-12,
    lam_max: float = 1e12,
    damping: str = "marquardt",
    schedule: str = "nielsen",
    robust_kernel: str = "none",
    robust_scale: float | str = 1.0,
    robust_scale_floor: float = 0.3,
    gnc_start: float = 0.0,
    gnc_iters: int = 0,
    weights: Optional[np.ndarray] = None,
    studentize: bool = False,
    linear_solve: Optional[Callable[[np.ndarray, np.ndarray, float, np.ndarray], np.ndarray]] = None,
    on_iter: Optional[Callable[[int, int, float, float, Any], None]] = None,
) -> OptResult:
    r"""Minimize ``Σ_i w_i ρ(‖r_i‖²)`` on a manifold by re-basing Levenberg–Marquardt.

    Parameters
    ----------
    state0
        Opaque state object (e.g. ``(R, t, X)``). Only the three callables touch it.
    residual(state) -> (M,)
        Stacked residual vector. Consecutive groups of ``block`` entries form one
        robustness unit ``r_i`` (so ``M`` must be a multiple of ``block``).
    jacobian(state) -> (M, K)
        Jacobian of ``residual`` **with respect to the tangent increment ``δ`` at
        ``δ = 0``** (i.e. the local-perturbation / right Jacobian). ``K`` is the
        tangent dimension. Because we re-base every step, this is evaluated at the
        current base where the perturbation is zero — the cheap form (no ``J_r``).
    retract(state, δ) -> state
        Apply a tangent step: produce a new *base* state ``state ⊞ δ``. This is where
        the re-basing lives (``R ← R·exp(δω)`` etc.).
    block
        Residual components per robustness unit (2 for image points).
    robust_kernel, robust_scale
        See :mod:`.robust`. ``robust_scale='auto'`` re-estimates the inlier scale by
        MAD each iteration (Huber Proposal-2).
    gnc_start, gnc_iters
        Graduated non-convexity: anneal the kernel scale from ``gnc_start`` down to
        the calibrated scale over ``gnc_iters`` iterations (0 disables).
    weights
        Optional per-block confidence weights ``(M//block,)``.
    studentize
        **Opt-in** bounded-influence IRLS (default ``False`` = behavior unchanged).
        When ``True``, the robust kernel is applied to the *studentized* squared
        residual ``s̃_i = r_iᵀ (I − H_ii + εI)⁻¹ r_i`` instead of the raw ``‖r_i‖²``
        (:func:`ds_msp.core.robust.studentized_scale_factors`), so a high-leverage
        outlier whose raw residual self-masks (it pulled the fit toward itself) is
        still inflated and down-weighted. The deflation factors are rebuilt from the
        Jacobian at each re-based iterate; the MAD auto-scale and the accept/reject
        cost use the same studentized metric so the comparison stays consistent.
    linear_solve(H, g, lam, D) -> δ
        Optional custom linear solver (e.g. a Schur-complement BA solve). Defaults to
        dense damped Cholesky :func:`_solve_damped`.
    on_iter(it, max_iter, rms, cost, state)
        Optional callback fired once per outer iteration with the *current best* (plain,
        non-robust) RMS and cost — i.e. after this iteration's step is applied if accepted,
        unchanged if rejected (a flat reading is a real plateau, not a stall in reporting) —
        plus the current opaque ``state`` object itself, so a caller in a layer that
        understands ``state``'s shape (e.g. ``rig/bundle.py``) can render the actual live
        geometry, not just the scalar trace. This solver never inspects `state`'s contents,
        so passing it adds no coupling here. Side-effect only; does not affect the solve.

    Returns
    -------
    OptResult
        ``state`` plus cost / RMS / iteration diagnostics. ``rms`` is the plain
        (non-robust) RMS of ``‖r_i‖`` so it stays comparable across kernels.
    """
    if robust_kernel not in VALID_KERNELS:
        raise ValueError(f"robust_kernel must be one of {VALID_KERNELS!r}")
    if damping not in ("marquardt", "isotropic"):
        raise ValueError("damping must be 'marquardt' or 'isotropic'")
    if schedule not in ("nielsen", "classic"):
        raise ValueError("schedule must be 'nielsen' or 'classic'")
    solve = linear_solve or _solve_damped
    auto = robust_scale == "auto"

    def block_norms(r: np.ndarray) -> np.ndarray:
        return np.linalg.norm(r.reshape(-1, block), axis=1)

    def block_sq(r: np.ndarray, F: Optional[np.ndarray]) -> np.ndarray:
        """Squared block residual ``s_i`` — studentized ``r_iᵀ F_i r_i`` when the
        deflation factors ``F`` are given, plain ``‖r_i‖²`` otherwise."""
        if F is None:
            return block_norms(r) ** 2
        rp = r.reshape(-1, block)
        return np.einsum("nk,nkl,nl->n", rp, F, rp)

    def scale_at(it: int, r: np.ndarray, F: Optional[np.ndarray] = None) -> float:
        if robust_kernel == "none":
            return 1.0
        bn = (block_norms(r) if F is None
              else np.sqrt(np.maximum(block_sq(r, F), 0.0)))
        c = (auto_kernel_scale(bn, robust_kernel, robust_scale_floor)
             if auto else float(robust_scale))
        if gnc_iters > 0 and gnc_start > 0:
            # With auto scale, GNC multiplies the current σ̂; otherwise it is the
            # absolute schedule wide→c.
            if auto:
                c = gnc_scale(it, gnc_iters, gnc_start, 1.0) * c
            else:
                c = gnc_scale(it, gnc_iters, gnc_start, c)
        return c

    def cost(r: np.ndarray, c: float, F: Optional[np.ndarray] = None) -> float:
        s = block_sq(r, F)
        rho = robust_cost(s, robust_kernel, c)
        if weights is not None:
            rho = rho * weights
        return float(rho.sum())

    def row_weights(r: np.ndarray, c: float,
                    F: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if robust_kernel == "none" and weights is None:
            return None
        s = block_sq(r, F)
        w = robust_weight(s, robust_kernel, c)
        if weights is not None:
            w = w * weights
        return np.repeat(w, block)                         # per-block → per-row

    state = state0
    r = np.asarray(residual(state), float)
    if r.size % block:
        raise ValueError(f"residual length {r.size} not a multiple of block {block}")
    c = scale_at(0, r)
    f = cost(r, c)
    lam, nu = lam_init, 2.0
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        J = F = None
        if studentize:
            # Deflation factors are Jacobian-only, so build them once per iterate
            # and reuse for the scale, the cost recompare, and the accept/reject.
            J = np.asarray(jacobian(state), float)
            F = studentized_scale_factors(J, weights, block=block)
        c = scale_at(it - 1, r, F)
        if auto or (gnc_iters > 0 and gnc_start > 0) or studentize:
            f = cost(r, c, F)                               # kernel moved: recompare
        if J is None:
            J = np.asarray(jacobian(state), float)
        wr = row_weights(r, c, F)
        if wr is None:
            H = J.T @ J
            g = J.T @ r
        else:
            Jw = J * wr[:, None]
            H = J.T @ Jw
            g = Jw.T @ r
        D = (np.maximum(np.diag(H), 1e-12) if damping == "marquardt"
             else np.ones(H.shape[0]))

        delta = solve(H, g, lam, D)
        state_try = retract(state, delta)
        r_try = np.asarray(residual(state_try), float)
        f_try = cost(r_try, c, F)                    # same F: consistent comparison

        accepted = f_try < f
        pred = 0.5 * float(delta @ (lam * D * delta - g))   # model's predicted reduction
        lam, nu = _update_lambda(accepted, lam, nu, schedule=schedule, pred=pred,
                                 actual=f - f_try, lam_min=lam_min, lam_max=lam_max)
        if accepted:
            state, r, f = state_try, r_try, f_try
        if on_iter is not None:
            bn_now = block_norms(r)
            on_iter(it, max_iter,
                   float(np.sqrt((bn_now ** 2).mean())) if bn_now.size else float("nan"), f,
                   state)
        if accepted:
            if float(np.max(np.abs(delta))) < tol:
                converged = True
                break
        elif lam >= lam_max:
            break

    bn = block_norms(r)
    rms = float(np.sqrt((bn ** 2).mean())) if bn.size else float("nan")
    return OptResult(state=state, cost=f, rms=rms, iterations=it,
                     success=converged or it < max_iter, converged=converged,
                     final_lambda=lam)


def gnc_tls_solve(
    state0: object,
    residual: Callable[[object], np.ndarray],
    jacobian: Callable[[object], np.ndarray],
    retract: Callable[[object, np.ndarray], object],
    *,
    noise_bound: float,
    block: int = 2,
    max_outer: int = 100,
    continuation: float = GNC_TLS_CONTINUATION,
    inner_max_iter: int = 50,
    weights: Optional[np.ndarray] = None,
    **lm_kwargs: Any,
) -> OptResult:
    r"""High-breakdown, **median-free**, no-initial-guess robust optimization via Graduated
    Non-Convexity with a Truncated-Least-Squares surrogate (Yang et al., RA-L 2020).

    Why this exists (vs ``lm_solve(robust_kernel=..., robust_scale='auto')``)
    -----------------------------------------------------------------------
    The redescending-kernel + MAD-auto-scale path keys the inlier band off the **median**, whose
    breakdown point is exactly **50%**. Past half gross outliers the auto-scale is dragged up,
    the kernel never tightens to the true band, and the solve fails. ``gnc_tls_solve`` instead
    graduates a *truncated* surrogate against an **explicit noise bound** ``c̄ = noise_bound``
    (in calibration this is known — the expected reprojection σ), so it recovers **well past 50%**
    contamination and returns a **hard inlier set**. It is deterministic and needs no init.

    It reuses :func:`lm_solve` as the inner weighted least-squares solver: each graduation level
    fixes per-block weights and runs an ordinary (non-robust) weighted solve, so the same Lie
    re-basing, damping, and Cholesky-jitter robustness apply.

    Parameters
    ----------
    noise_bound
        The inlier scale ``c̄`` in residual-norm units (e.g. pixels for a 2-D reprojection block);
        the inlier band is ``barc2 = c̄²``. Pick ``c̄`` from a χ² bound on the expected noise σ
        (2-DoF 99% ⇒ ``c̄ ≈ 3.03·σ``), **not** from a MAD estimate.
    block
        Residual components per robustness unit (2 for image points, 3 for 3-D registration).
    max_outer, continuation
        Graduation budget and geometric ``μ`` growth (``μ ← μ·continuation`` each level).
    inner_max_iter, weights, **lm_kwargs
        Forwarded to the inner :func:`lm_solve` (``weights`` are optional per-block confidence
        priors, multiplied into the TLS weights; e.g. ``damping``, ``schedule``, ``tol``).

    Returns
    -------
    OptResult
        With ``weights`` set to the final per-block inlier weights (≈1 inlier / ≈0 outlier).
    """
    def inner_solve(state: object, block_weights: Optional[np.ndarray]) -> OptResult:
        return lm_solve(state, residual, jacobian, retract, block=block,
                        robust_kernel="none", weights=block_weights,
                        max_iter=inner_max_iter, **lm_kwargs)
    return _gnc_tls_graduate(state0, residual, inner_solve, noise_bound=noise_bound,
                             block=block, max_outer=max_outer, continuation=continuation,
                             weights=weights)


def gnc_tls_schur_solve(
    state0: object,
    residual: Callable[[object], np.ndarray],
    linearize: Callable[[object], tuple],
    retract: Callable[[object, np.ndarray, np.ndarray], object],
    *,
    noise_bound: float,
    n_groups: int,
    shared_dim: int,
    local_dim: int,
    block: int = 2,
    max_outer: int = 100,
    continuation: float = GNC_TLS_CONTINUATION,
    inner_max_iter: int = 50,
    weights: Optional[np.ndarray] = None,
    **schur_kwargs: Any,
) -> OptResult:
    r"""GNC-TLS robust optimization for **separable** problems — the :func:`schur_lm` analogue
    of :func:`gnc_tls_solve`.

    Same median-free, explicit-``noise_bound``, >50%-breakdown graduation as
    :func:`gnc_tls_solve`, but the inner weighted solve is the sparse Schur-complement
    :func:`schur_lm` — so it drives high-breakdown robustness through bundle adjustment over
    shared intrinsics + per-image poses without paying the dense cost. The per-block TLS weights
    are stacked in the same order ``linearize`` concatenates its groups. See :func:`gnc_tls_solve`
    for the ``noise_bound`` semantics (``c̄ ≈ 3.03·σ`` for a 2-DoF 99% band).
    """
    def inner_solve(state: object, block_weights: Optional[np.ndarray]) -> OptResult:
        return schur_lm(state, residual, linearize, retract, n_groups=n_groups,
                        shared_dim=shared_dim, local_dim=local_dim, block=block,
                        robust_kernel="none", weights=block_weights,
                        max_iter=inner_max_iter, **schur_kwargs)
    return _gnc_tls_graduate(state0, residual, inner_solve, noise_bound=noise_bound,
                             block=block, max_outer=max_outer, continuation=continuation,
                             weights=weights)


def _gnc_tls_graduate(
    state0: object,
    residual: Callable[[object], np.ndarray],
    inner_solve: Callable[[object, Optional[np.ndarray]], OptResult],
    *,
    noise_bound: float,
    block: int,
    max_outer: int,
    continuation: float,
    weights: Optional[np.ndarray],
) -> OptResult:
    """Solver-agnostic GNC-TLS outer loop shared by :func:`gnc_tls_solve` (dense) and
    :func:`gnc_tls_schur_solve` (sparse). ``inner_solve(state, block_weights) -> OptResult``
    runs one ordinary (non-robust) weighted least-squares solve; this loop graduates the
    truncated surrogate ``μ`` against ``barc2 = noise_bound²`` until the per-block weights
    binarize into a hard inlier set."""
    barc2 = float(noise_bound) ** 2
    if barc2 <= 0.0:
        raise ValueError("noise_bound must be positive")

    def block_sq(state: object) -> np.ndarray:
        r = np.asarray(residual(state), float)
        if r.size % block:
            raise ValueError(f"residual length {r.size} not a multiple of block {block}")
        return np.sum(r.reshape(-1, block) ** 2, axis=1)

    # Initialize from an ordinary (unweighted) solve — no robustness yet, no init guess needed.
    res = inner_solve(state0, weights)
    state = res.state
    s = block_sq(state)
    mu = gnc_tls_mu_init(s, barc2)

    w = np.ones_like(s)
    prev = float("inf")
    outer = 0
    for outer in range(1, max_outer + 1):
        w = gnc_tls_weight(s, barc2, mu)
        bw = w if weights is None else w * weights
        res = inner_solve(state, bw)
        state = res.state
        s = block_sq(state)
        cost = float((s * w).sum())
        binary = bool(np.all((w <= 1e-9) | (np.abs(1.0 - w) <= 1e-9)))
        if binary or abs(cost - prev) < 1e-12:
            break
        mu *= continuation
        prev = cost

    bn = np.sqrt(s)
    rms = float(np.sqrt((bn ** 2).mean())) if bn.size else float("nan")
    return OptResult(state=state, cost=float((s * w).sum()), rms=rms, iterations=outer,
                     success=True, converged=True, final_lambda=res.final_lambda, weights=w)


def schur_lm(
    state0: object,
    residual: Callable[[object], np.ndarray],
    linearize: Callable[[object], tuple],
    retract: Callable[[object, np.ndarray, np.ndarray], object],
    *,
    n_groups: int,
    shared_dim: int,
    local_dim: int,
    block: int = 2,
    max_iter: int = 50,
    tol: float = 1e-9,
    lam_init: float = 1e-3,
    lam_min: float = 1e-12,
    lam_max: float = 1e12,
    schedule: str = "nielsen",
    robust_kernel: str = "none",
    robust_scale: float | str = 1.0,
    robust_scale_floor: float = 0.3,
    gnc_start: float = 0.0,
    gnc_iters: int = 0,
    weights: Optional[np.ndarray] = None,
    on_iter: Optional[Callable[[int, int, float, float, Any], None]] = None,
) -> OptResult:
    r"""Levenberg–Marquardt for **separable** problems: a small block of *shared*
    parameters (``shared_dim``) coupled to ``n_groups`` *independent* per-group local
    blocks (``local_dim`` each, e.g. one 6-DOF pose per image).

    This is sparse bundle adjustment. The Hessian is an *arrow*::

        [ U    W₀  W₁ … ]      U   = shared–shared      (shared_dim²)
        [ W₀ᵀ  V₀     0 ]      Vᵢ  = group i local      (local_dim²)
        [ W₁ᵀ  0   V₁   ]      Wᵢ  = shared–group i      (shared_dim × local_dim)
        [ ⋮              ]

    Marginalizing the (block-diagonal, trivially invertible) ``Vᵢ`` gives a reduced
    ``shared_dim × shared_dim`` system — the **Schur complement** ``S = U − Σ Wᵢ Vᵢ⁻¹ Wᵢᵀ``
    — then each ``δ_localᵢ`` back-substitutes independently. Cost is *linear* in the
    number of groups instead of cubic in the full dimension, so calibration with
    hundreds of images stays fast. Same robust IRLS / auto-scale / GNC / re-basing
    machinery as :func:`lm_solve`.

    Callbacks
    ---------
    residual(state) -> (M,)
        Full stacked residual (for the robust cost / accept-reject).
    linearize(state) -> (r_list, A_list, B_list)
        Per group ``i``: residual ``r_list[i] (mᵢ,)``, shared Jacobian
        ``A_list[i] (mᵢ, shared_dim)``, local Jacobian ``B_list[i] (mᵢ, local_dim)``
        — both w.r.t. the tangent increment at the current (re-based) state.
        Invalid rows should already be zeroed.
    retract(state, δ_shared, δ_local) -> state
        ``δ_shared (shared_dim,)``; ``δ_local (n_groups, local_dim)``.
    weights
        Optional per-block confidence weights ``(M//block,)`` stacked in the same order the
        ``linearize`` groups concatenate (group 0's blocks first, then group 1's, …). Folded
        into the IRLS row weights — this is the hook :func:`gnc_tls_solve` uses to drive a
        graduated solve through the sparse separable solver.
    on_iter(it, max_iter, rms, cost, state)
        Optional per-iteration progress callback — see :func:`lm_solve`.
    """
    if robust_kernel not in VALID_KERNELS:
        raise ValueError(f"robust_kernel must be one of {VALID_KERNELS!r}")
    auto = robust_scale == "auto"

    def cost(r: np.ndarray, c: float) -> float:
        s = np.linalg.norm(r.reshape(-1, block), axis=1) ** 2
        rho = robust_cost(s, robust_kernel, c)
        if weights is not None:
            rho = rho * weights
        return float(rho.sum())

    def scale_at(it: int, r: np.ndarray) -> float:
        if robust_kernel == "none":
            return 1.0
        bn = np.linalg.norm(r.reshape(-1, block), axis=1)
        c = auto_kernel_scale(bn, robust_kernel, robust_scale_floor) if auto else float(robust_scale)
        if gnc_iters > 0 and gnc_start > 0:
            c = gnc_scale(it, gnc_iters, gnc_start, 1.0) * c if auto else gnc_scale(it, gnc_iters, gnc_start, c)
        return c

    def row_weights(r_i: np.ndarray, ext_w: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if robust_kernel == "none" and ext_w is None:
            return None
        n_blk = r_i.size // block
        if robust_kernel == "none":
            wk = np.ones(n_blk)
        else:
            s = np.linalg.norm(r_i.reshape(-1, block), axis=1) ** 2
            wk = robust_weight(s, robust_kernel, c)
        if ext_w is not None:
            wk = wk * ext_w
        return np.repeat(wk, block)

    state = state0
    r_full = np.asarray(residual(state), float)
    c = scale_at(0, r_full)
    f = cost(r_full, c)
    lam, nu = lam_init, 2.0
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        c = scale_at(it - 1, r_full)
        if auto or (gnc_iters > 0 and gnc_start > 0):
            f = cost(r_full, c)
        r_list, A_list, B_list = linearize(state)

        # Assemble the arrow blocks (with IRLS row weights folded in).
        U = np.zeros((shared_dim, shared_dim))
        ga = np.zeros(shared_dim)
        Vs, Ws, gbs = [], [], []
        boff = 0                                            # running per-block cursor for `weights`
        for r_i, A_i, B_i in zip(r_list, A_list, B_list):
            n_blk = r_i.size // block
            ext = None if weights is None else weights[boff:boff + n_blk]
            boff += n_blk
            w = row_weights(r_i, ext)
            Aw = A_i if w is None else A_i * w[:, None]
            Bw = B_i if w is None else B_i * w[:, None]
            U += A_i.T @ Aw
            ga += Aw.T @ r_i
            Vs.append(B_i.T @ Bw)
            Ws.append(A_i.T @ Bw)
            gbs.append(Bw.T @ r_i)

        Da = np.maximum(np.diag(U).copy(), 1e-12)
        Dbs = [np.maximum(np.diag(V).copy(), 1e-12) for V in Vs]

        # Schur complement onto the shared block.
        S = U + lam * np.diag(Da)
        rhs = -ga
        Vinvs = []
        for V, W, gb, Db in zip(Vs, Ws, gbs, Dbs):
            Vinv = _safe_inv(V + lam * np.diag(Db))
            Vinvs.append(Vinv)
            Y = W @ Vinv                              # (shared, local)
            S -= Y @ W.T
            rhs += Y @ gb
        # Re-symmetrize: ``S -= Y@W.T`` accumulates float asymmetry, and
        # np.linalg.cholesky silently truncates to the lower triangle otherwise.
        S = 0.5 * (S + S.T)
        try:
            L = np.linalg.cholesky(S)
            d_shared = _cho_solve(L, rhs)
        except np.linalg.LinAlgError:
            # Route through the same escalating scale-aware jitter ladder as the
            # dense path (λ=0, D=0 makes _solve_damped solve S·δ = rhs directly).
            d_shared = _solve_damped(S, -rhs, 0.0, np.zeros(shared_dim))

        d_local = np.zeros((n_groups, local_dim))
        for i, (Vinv, W, gb) in enumerate(zip(Vinvs, Ws, gbs)):
            d_local[i] = -Vinv @ (gb + W.T @ d_shared)

        state_try = retract(state, d_shared, d_local)
        r_try = np.asarray(residual(state_try), float)
        f_try = cost(r_try, c)

        # Predicted reduction for the gain ratio: ½ δᵀ(λ D δ − g) over the full vector.
        dmax = max(float(np.max(np.abs(d_shared))),
                   float(np.max(np.abs(d_local))) if n_groups else 0.0)
        pred = 0.5 * (lam * (float(Da @ (d_shared ** 2))
                             + sum(float(Db @ (dl ** 2)) for Db, dl in zip(Dbs, d_local)))
                      - float(ga @ d_shared)
                      - sum(float(gb @ dl) for gb, dl in zip(gbs, d_local)))
        accepted = f_try < f
        lam, nu = _update_lambda(accepted, lam, nu, schedule=schedule, pred=pred,
                                 actual=f - f_try, lam_min=lam_min, lam_max=lam_max)
        if accepted:
            state, r_full, f = state_try, r_try, f_try
        if on_iter is not None:
            bn_now = np.linalg.norm(r_full.reshape(-1, block), axis=1)
            on_iter(it, max_iter,
                   float(np.sqrt((bn_now ** 2).mean())) if bn_now.size else float("nan"), f,
                   state)
        if accepted:
            if dmax < tol:
                converged = True
                break
        elif lam >= lam_max:
            break

    bn = np.linalg.norm(r_full.reshape(-1, block), axis=1)
    rms = float(np.sqrt((bn ** 2).mean())) if bn.size else float("nan")
    return OptResult(state=state, cost=f, rms=rms, iterations=it,
                     success=converged or it < max_iter, converged=converged,
                     final_lambda=lam)
