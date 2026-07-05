#!/usr/bin/env python3
"""
Learning by doing — hard rejection vs a robust loss, and why naive RMS lies.

Companion to the capstone (`examples/03`). Same real TUM-VI corners, three ways to
deal with the handful of mis-localized peripheral corners, side by side:

  (a) plain L2          — every corner pulls with weight proportional to its error^2;
                          a few 3-6 px outliers dominate and bias the focal length.
  (b) hard rejection    — bundle-adjust, DROP every corner that reprojects > 1 px,
                          refit. Throws data away; brittle at the threshold.
  (c) robust (Cauchy)   — keep EVERY corner, but down-weight large residuals
                          continuously (IRLS, weight = rho'(r)/r). No data lost,
                          no cliff-edge.

It also makes the evaluation trap concrete: under a robust loss, the naive RMS over
*all* corners is inflated by the very outliers the loss correctly ignored — so it
looks "worse" while the fit is actually better. The honest read is median / inlier RMS.

See `docs/learn/robust_losses_and_evaluation.md` for the math.

Prerequisites:
  - `pip install -e .[calib]`
  - `bash scripts/download_datasets.sh tumvi`

Run:
  python examples/04_robust_vs_rejection.py --stride 4
  python examples/04_robust_vs_rejection.py --scales 1   # single-scale detection: the
                                                          # multi-scale default (1,2,3) now
                                                          # detects few enough severe
                                                          # mis-decodes that L2/hard-reject/
                                                          # Cauchy barely differ -- single-
                                                          # scale reproduces a real, larger
                                                          # outlier population to show the
                                                          # effect clearly.
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

from ds_msp.calib import AprilGridTarget, calibrate, detect_aprilgrid
from ds_msp.io.kalibr import load_kalibr_with_resolution
from ds_msp.models import KannalaBrandtModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAM_DIR = os.path.join(HERE, "datasets", "tumvi",
                       "dataset-calib-cam1_512_16", "mav0", "cam0", "data")
CAMCHAIN = os.path.join(HERE, "datasets", "tumvi",
                        "dataset-room1_512_16", "dso", "camchain.yaml")


def stats(model, poses, Xs, UVs):
    """Per-corner reprojection error: the naive all-corner RMS vs robust reads."""
    errs = []
    for X, uv, (rvec, tvec) in zip(Xs, UVs, poses):
        R, _ = cv2.Rodrigues(rvec)
        uvp, valid = model.project((R @ X.T).T + tvec)
        errs.append(np.linalg.norm(uvp - uv, axis=1)[valid])
    e = np.concatenate(errs)
    inl = e[e < 1.0]
    return {
        "naive_rms": float(np.sqrt((e ** 2).mean())),       # over ALL corners
        "median": float(np.median(e)),
        "inlier_rms": float(np.sqrt((inl ** 2).mean())),    # over corners < 1 px
        "inlier_pct": 100.0 * len(inl) / len(e),
    }


def hard_reject(seed, Xs, UVs, VIs, reject_px=1.0):
    """Two-pass: fit, drop corners reprojecting worse than reject_px, refit."""
    first = calibrate(seed, Xs, UVs, VIs, max_nfev=120)
    m = first["model"]
    Xk, Uk, Vk, kept, total = [], [], [], 0, 0
    for X, uv, (rvec, tvec) in zip(Xs, UVs, first["poses"]):
        R, _ = cv2.Rodrigues(rvec)
        uvp, valid = m.project((R @ X.T).T + tvec)
        good = valid & (np.linalg.norm(uvp - uv, axis=1) < reject_px)
        total += len(X)
        kept += int(good.sum())
        if good.sum() >= 8:
            Xk.append(X[good])
            Uk.append(uv[good])
            Vk.append(np.ones(int(good.sum()), bool))
    out = calibrate(m, Xk, Uk, Vk, max_nfev=120)
    out["used"] = f"{kept}/{total}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--scales", type=str, default="1,2,3",
                    help="comma-separated detection scales (default: the library's own "
                         "multi-scale default). Pass '1' for single-scale detection, which "
                         "reproduces a real, sizeable outlier population -- useful for "
                         "seeing the L2/hard-reject/Cauchy difference clearly (see "
                         "docs/learn/robust_losses_and_evaluation.md).")
    args = ap.parse_args()
    if not os.path.isdir(CAM_DIR):
        raise SystemExit("Fetch TUM-VI first: bash scripts/download_datasets.sh tumvi")

    scales = tuple(float(s) for s in args.scales.split(","))
    ref, (W, H) = load_kalibr_with_resolution(CAMCHAIN, cam="cam0")
    paths = sorted(glob.glob(os.path.join(CAM_DIR, "*.png")))[::args.stride]
    target = AprilGridTarget(6, 6, 0.088, 0.3)
    Xs, UVs, VIs = target.build_correspondences(
        detect_aprilgrid(paths, family="t36h11", min_tags=6, refine=True, scales=scales),
        min_corners=8)
    n = sum(len(x) for x in Xs)
    print(f"{len(Xs)} frames, {n} detected corners. Published fx = {ref.fx:.3f}\n")

    def seed():
        return KannalaBrandtModel(180.0, 180.0, W / 2, H / 2)
    runs = [
        ("L2 (no robustness)", calibrate(seed(), Xs, UVs, VIs, max_nfev=120), f"{n}/{n}"),
        ("hard reject >1px",   None, None),  # filled below (needs its own fit)
        ("Cauchy f_scale=0.5", calibrate(seed(), Xs, UVs, VIs, max_nfev=120,
                                         loss="cauchy", f_scale=0.5), f"{n}/{n}"),
    ]
    hr = hard_reject(seed(), Xs, UVs, VIs)
    runs[1] = ("hard reject >1px", hr, hr["used"])

    hdr = f"{'method':22s} {'corners':>9s} {'Δfx':>7s} {'median':>8s} {'inlierRMS':>10s} {'naiveRMS':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for name, out, used in runs:
        m = out["model"]
        s = stats(m, out["poses"], Xs, UVs)
        print(f"{name:22s} {used:>9s} {abs(m.fx-ref.fx):7.2f} {s['median']:8.3f} "
              f"{s['inlier_rms']:10.3f} {s['naive_rms']:9.3f}")

    print("\nRead it like this:")
    print("  * L2's Δfx is worst — a few big outliers, weighted by error^2, drag the focal.")
    print("  * hard-reject can win on Δfx (tightest fit of the corners it kept), but its")
    print("    naiveRMS explodes: once it drops a corner, nothing constrains where the")
    print("    model thinks that corner should be, so a badly mis-decoded one (real cases")
    print("    here are 20-100+ px off) can blow up when scored against its true position.")
    print("  * Cauchy never fully lets go of any corner, so its naiveRMS stays close to L2's")
    print("    even though its fit is tighter. THE TRAP either way: naive RMS over ALL")
    print("    corners reflects whichever method stopped constraining its worst residuals,")
    print("    not fit quality. Judge by median / inlier RMS, never by all-corner RMS.")


if __name__ == "__main__":
    main()
