#!/usr/bin/env python3
"""
Sanity-check calibration end to end using only the bundled test assets (no external
dataset download): build 3D<->2D correspondences from test_config.json's checkerboard
geometry and pre-detected keypoints (both bundled test images -- a single planar view
alone leaves Double Sphere's focal/xi/alpha gauge underdetermined, see
docs/explain/are_two_models_the_same_camera.md), then bundle-adjust a Double Sphere
model with ds_msp.calib.calibrate and check it converges close to the reference
intrinsics with a tight reprojection fit.

This replaces the deleted calibrate.py/validate.py scripts' role in verify_all.sh.

Run:
    python tests/verify_calibration.py
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from ds_msp.calib import calibrate
from ds_msp.models import DoubleSphereModel
from ds_msp.ops import solve_pnp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(HERE, "test_config.json")


def main() -> None:
    cfg = json.load(open(CFG))
    intr = cfg["intrinsics"]
    board = cfg["checkerboard"]
    rows, cols, sq = board["rows"], board["cols"], board["square_size"]

    X = np.array([[c * sq, r * sq, 0.0] for r in range(rows) for c in range(cols)])
    visibility = np.ones(len(X), dtype=bool)
    Xs, kps, viss = [], [], []
    for t in cfg["test_images"]:
        kp = np.array(t["keypoints_2d"])
        assert len(kp) == rows * cols, f"expected {rows * cols} corners, got {len(kp)}"
        Xs.append(X)
        kps.append(kp)
        viss.append(visibility)

    print("--- Calibration (both bundled views, DS model) ---")
    seed = DoubleSphereModel(fx=intr["fx"] * 0.9, fy=intr["fy"] * 0.9,
                             cx=intr["width"] / 2, cy=intr["height"] / 2,
                             xi=0.0, alpha=0.5)
    result = calibrate(seed, Xs, kps, viss, max_nfev=200)

    print(f"success:   {result['success']}")
    print(f"rms_px:    {result['rms_px']:.4f}")
    print(f"median_px: {result['median_px']:.4f}")
    print(f"model:     {result['model']}")

    assert result["success"], "calibration did not converge"
    assert result["rms_px"] < 1.0, f"reprojection RMS too high: {result['rms_px']:.3f} px"
    fx_err = abs(result["model"].fx - intr["fx"]) / intr["fx"]
    assert fx_err < 0.1, f"focal length too far from reference: {100*fx_err:.1f}% off"
    print("[PASS] calibration converges close to the reference intrinsics with sub-pixel RMS")

    print("\n--- PnP + reprojection RMS (known reference intrinsics, per test image) ---")
    cam = DoubleSphereModel(**{k: intr[k] for k in ("fx", "fy", "cx", "cy", "xi", "alpha")})
    for t in cfg["test_images"]:
        uv = np.array(t["keypoints_2d"])
        ok, rvec, tvec = solve_pnp(cam, X, uv)
        assert ok, f"PnP failed on {t['file']}"
        R, _ = cv2.Rodrigues(rvec)
        uvp, valid = cam.project((R @ X.T).T + tvec)
        err = np.linalg.norm(uvp - uv, axis=1)
        rms = float(np.sqrt((err**2).mean()))
        print(f"{t['file']}: mean={err.mean():.3f} px, rms={rms:.3f} px, max={err.max():.3f} px")
        assert rms < 1.0, f"PnP reprojection RMS too high on {t['file']}: {rms:.3f} px"
    print("[PASS] PnP recovers pose with sub-pixel reprojection RMS on both bundled views")


if __name__ == "__main__":
    main()
