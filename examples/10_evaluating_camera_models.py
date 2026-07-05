#!/usr/bin/env python3
"""
Is this camera model right for *my* lens and task?  A measurable framework.

Picking a camera model is usually done by reputation ("everyone uses Kannala-Brandt")
or by parameter count ("more params = more accurate"). Both are wrong often enough to
ruin a calibration. This script builds the model *out of measurements* instead: six
diagnostics, each a number you can verify, that together tell you whether a model has
the right capacity, whether its parameters are identifiable, what it costs, and where
it will fail.

We run the whole battery on two reproducible Kannala-Brandt "ground-truth" lenses so the
answers are exact and need no download:

  * BENIGN   — a mild ~170° lens (smooth radial profile; 2 shape DOF are plenty)
  * DEMANDING — a ~158° industrial lens with a strong theta^5 bend (needs >=3 radial DOF)

and compare two candidate models against each — EUCM (6p) and DS+ (9p) — with KB itself as
the reference. The lesson the contrast teaches: the *same* model is a great choice on one
lens and a poor one on another, and only the diagnostics tell you which.

The six diagnostics (each maps to a question you actually have):
  A. CAPACITY        Can the model even represent my lens?      -> conversion residual
  B. IDENTIFIABILITY Are the parameters pinned by the data?     -> Jacobian Gram condition #
  C. REDUNDANCY      Does an added parameter add a new DOF?     -> column collinearity |cos|
  D. CONSTRAINTS     Is a bound hiding the real story?          -> refit bound vs unbounded
  E. COST            What does it cost to project/unproject?    -> timing + operation class
  F. TASK WEIGHT     Where does my sampling put the error?      -> implicit angular weight

Companion: docs/learn/choosing_a_camera_model.md
           docs/explain/case_study_eucmplus_dsplus_kb.md (historical — see that page's note on
           EUCM+'s removal from the shipped library)

Prereq: pip install -e .     (core only; no dataset needed)
Run:    python examples/10_evaluating_camera_models.py
"""
from __future__ import annotations

import time

import numpy as np

from ds_msp.adapt import convert
from ds_msp.models.registry import model_class


def section(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


# --- two reproducible KB ground-truth lenses -------------------------------------
def kb_lens(name: str):
    KB = model_class("kb")
    if name == "benign":   # mild ~170° lens, 512x512
        p = [192.0, 192.0, 256.0, 256.0, 0.005, -0.003, -0.003, 0.002]
        return KB.from_params(np.array(p, float)), 512, 512, 170.0
    if name == "demanding":  # strong theta^5 bend, ~158°, 2592x1800
        p = [995.3, 995.3, 1276.4, 876.0, 0.2388, -0.0256, -0.0921, 0.0299]
        return KB.from_params(np.array(p, float)), 2592, 1800, 158.0
    raise ValueError(name)


def radial(model, th):
    """Image radius (px from principal point) of rays at angles th (phi=0)."""
    rays = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=1)
    px, _ = model.project(rays)
    return np.hypot(px[:, 0] - model.cx, px[:, 1] - model.cy)


def shape_columns(model, idx, th):
    """Finite-difference d r(theta) / d p_i for each shape param index in `idx`."""
    p0 = model.params.copy()
    C = type(model)
    base = radial(model, th)
    cols = []
    for i in idx:
        h = 1e-4 * max(abs(p0[i]), 1e-3)
        dp = p0.copy()
        dp[i] += h
        cols.append((radial(C.from_params(dp), th) - base) / h)
    return np.stack(cols, axis=1)


# fit a candidate model to a KB ground-truth lens over its FOV (representational capacity)
def fit_to(kb, Tcls, W, H, fov):
    m, info = convert(kb, Tcls, width=W, height=H, max_fov_deg=fov, n_samples=1500,
                      n_restarts=6)
    return m, info["rms_px"]


CANDS = ["eucm", "dsplus"]
SHAPE_IDX = {"eucm": [4, 5], "dsplus": [4, 5, 6]}
SHAPE_LBL = {"eucm": "alpha,beta", "dsplus": "alpha,lam1,lam2"}


def run(lens_name: str) -> None:
    kb, W, H, fov = kb_lens(lens_name)
    section(f"LENS = {lens_name.upper()}  ({W}x{H}, ~{fov:.0f}° FOV, KB ground truth)")
    th = np.deg2rad(np.linspace(1.0, fov / 2 - 1.0, 500))

    # A. CAPACITY ----------------------------------------------------------------
    print("A. CAPACITY  — convert KB ground truth -> model, in-FOV reprojection RMS (px)")
    fitted = {}
    for n in CANDS:
        m, rms = fit_to(kb, model_class(n), W, H, fov)
        fitted[n] = m
        print(f"     {n:9s} ({len(m.params)}p)  RMS = {rms:8.4f}")
    print("     read: RMS >> 0.1 px means the family literally cannot bend like this lens.")

    # B. IDENTIFIABILITY ---------------------------------------------------------
    print("\nB. IDENTIFIABILITY — shape-Jacobian Gram condition # (high => params not pinned)")
    for n in CANDS:
        J = shape_columns(fitted[n], SHAPE_IDX[n], th)
        s = np.linalg.svd(J.T @ J, compute_uv=False)
        print(f"     {n:9s} cond = {s[0] / s[-1]:.2e}   singular values = "
              f"{np.array2string(s, precision=2)}   [{SHAPE_LBL[n]}]")
    print("     read: cond ~1e3 healthy; ~1e6+ a near-null direction; ~1e15+ rank-deficient.")

    # C. REDUNDANCY --------------------------------------------------------------
    print("\nC. REDUNDANCY — does the *last* shape param add a DOF the others don't span?")
    for n in ["dsplus"]:
        J = shape_columns(fitted[n], SHAPE_IDX[n], th)
        last = J[:, -1]
        rest = J[:, :-1]
        coef, *_ = np.linalg.lstsq(rest, last, rcond=None)
        resid = np.linalg.norm(last - rest @ coef) / np.linalg.norm(last)
        cos_prev = abs(rest[:, -1] @ last) / (np.linalg.norm(rest[:, -1]) * np.linalg.norm(last))
        tag = SHAPE_LBL[n].split(",")[-1]
        print(f"     {n:9s} {tag:5s}: |cos with prev|={cos_prev:.4f}  "
              f"residual outside span(others)={resid:.3%}")
    print("     read: residual <1% means the new parameter is (nearly) redundant.")

    # D. CONSTRAINTS -------------------------------------------------------------
    print("\nD. CONSTRAINTS — is EUCM's failure just its alpha<=1 bound, or real lack of capacity?")
    EUCM = model_class("eucm")
    lb, ub = EUCM.param_bounds()
    orig = EUCM.param_bounds
    for amax in [1.0, 3.0, 8.0]:
        ub2 = ub.copy()
        ub2[4] = amax
        EUCM.param_bounds = staticmethod(lambda lb=lb, ub2=ub2: (lb, ub2))
        m, rms = fit_to(kb, EUCM, W, H, fov)
        print(f"     EUCM alpha<= {amax:4.1f}   RMS = {rms:8.4f}   (fitted alpha={m.params[4]:.3f})")
    EUCM.param_bounds = orig
    dsplus_rms = fit_to(kb, model_class("dsplus"), W, H, fov)[1]
    print(f"     DS+ (3rd shape DOF)   RMS = {dsplus_rms:8.4f}")
    print("     read: if unbounding alpha still can't match DS+, the extra param is real capacity.")


def cost_and_weight() -> None:
    section("E + F  (lens-independent: operation cost and sampling weight)")
    kb, W, H, fov = kb_lens("demanding")
    rng = np.random.default_rng(0)
    P3 = rng.normal(size=(300000, 3))
    P3[:, 2] = np.abs(P3[:, 2]) + 0.3
    P3 /= np.linalg.norm(P3, axis=1, keepdims=True)
    gx, gy = np.meshgrid(np.linspace(2, W - 2, 400), np.linspace(2, H - 2, 400))
    PX = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)
    fitted = {"kb": kb}
    for n in CANDS:
        fitted[n], _ = fit_to(kb, model_class(n), W, H, fov)

    print("E. COST — project 300k rays / unproject 160k px (closed-form vs Newton vs quartic)")
    print(f"     {'model':9s} {'project ms':>11} {'unproject ms':>13} {'roundtrip px':>13} {'valid%':>7}")
    for n in ["kb", "eucm", "dsplus"]:
        m = fitted[n]
        t = time.perf_counter()
        for _ in range(3):
            m.project(P3)
        tp = (time.perf_counter() - t) / 3 * 1e3
        t = time.perf_counter()
        for _ in range(3):
            r, v = m.unproject(PX)
        tu = (time.perf_counter() - t) / 3 * 1e3
        uv, ok = m.project(r)
        mk = v & ok
        rt = np.linalg.norm(uv[mk] - PX[mk], axis=1)
        print(f"     {n:9s} {tp:11.1f} {tu:13.1f} {np.median(rt):13.1e} {mk.mean() * 100:7.1f}")
    print("     read: KB unproject = Newton (exact, iterative); DS+ = Ferrari quartic when")
    print("           lam2!=0 (slow); EUCM = one sqrt (fast). Project is ~tied.")

    print("\nF. TASK WEIGHT — implicit angular weight of uniform-pixel sampling, w(θ)=r·r'")
    th = np.deg2rad(np.linspace(0.5, fov / 2, 2000))
    r = radial(kb, th)
    rp = np.gradient(r, th)
    w = np.clip(r * rp, 0, None)
    cum = np.cumsum(w) / w.sum()
    half = np.searchsorted(th, np.deg2rad(fov / 4))
    print(f"     inner half-FOV (θ< {fov / 4:.0f}°) carries {cum[half] * 100:4.1f}% of the fit weight;"
          f"  outer half {100 - cum[half] * 100:4.1f}%")
    print("     read: pixel-uniform sampling over-weights the periphery — match the sampling")
    print("           FOV to where your task actually uses the lens (see convert max_fov_deg).")


def main() -> None:
    print(__doc__.split("\n\n")[0])
    run("benign")
    run("demanding")
    cost_and_weight()
    section("HOW TO READ THE WHOLE TABLE")
    print("  Same two candidates, verdicts that depend on lens AND on how much FOV you use:")
    print("   - BENIGN: over the FULL FOV, DS+ tracks KB almost exactly (A: 0.02px) while EUCM")
    print("     drifts at the rim (0.24px). But if your task only uses the inner field (most")
    print("     calibration corners do), EUCM is within tolerance and far cheaper — pick it.")
    print("   - DEMANDING: EUCM cannot bend enough (A), even unbounded (D) -> real lack of")
    print("     capacity. DS+ fits best (A) with its 3rd DOF genuinely independent (C: 4-13%),")
    print("     at the cost of a slower quartic unproject when lam2!=0 (E).")
    print("  No model is 'best' in the abstract. The diagnostics decide, per lens and per task.")
    print("  (These are full-FOV capacity numbers; on real corners that only reach the inner")
    print("   field, the models tie — which is exactly why diagnostic A must use YOUR task FOV.)")


if __name__ == "__main__":
    main()
