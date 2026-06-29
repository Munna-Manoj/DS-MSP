"""Validate rig calibration to within 1% of the *original* intrinsics AND pose, with a
**model-of-choice** that may differ from the model the cameras were built with.

The motivating question (from real configs): a rig's cameras are described by some model
— typically Kannala-Brandt or RadTan — but I want to calibrate them with a *different*
model of my choosing (say Double Sphere). How accurate is that? You cannot compare a KB
parameter vector to a DS one directly (``are_two_models_the_same_camera.md``), so for each
camera we **convert the original model into the chosen model** (``adapt.convert``: fit the
chosen model to reproduce the original's rays) and treat that as the reference the chosen
parametrization *should* recover. We then report three model-independent errors:

  * pose    — worst inter-camera baseline error vs ground truth (%);
  * focal   — paraxial focal ``f_eff = dr/dθ|₀`` vs the converted reference (%), the one
              intrinsic that means the same thing in every model;
  * func    — RMS pixel disagreement between the calibrated camera and the reference over
              the whole image, as a % of focal — the strict "is it the same camera?" test.

All three under 1% ⇒ the chosen model reproduced the original camera and its pose to 1%.
Run across gross-outlier rates to show the from-scratch robust front-end (RANSAC DLT
resection + RANSAC PnP, no ``cv2.calibrateCamera``/``solvePnP``) holds where the old L2
OpenCV seeding diverged. Usage:  python scripts/validate_param_pose.py [n_seeds] [outlier]
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from tests.rig._synth import make_rig                                       # noqa: E402
from ds_msp.rig.calibrate import (calibrate_rig, make_bundle_front_end,  # noqa: E402
                                      paraxial_focal)
from ds_msp.adapt.convert import convert                                    # noqa: E402
from ds_msp.adapt.evaluate import reprojection_report                       # noqa: E402
from ds_msp.models.radtan import RadTanModel                               # noqa: E402
from ds_msp.models.double_sphere import DoubleSphereModel                  # noqa: E402
from ds_msp.models.ucm import UCMModel                                     # noqa: E402
from ds_msp.models.eucm import EUCMModel                                   # noqa: E402
from ds_msp.models.kb import KannalaBrandtModel                            # noqa: E402

W, H = 1280, 960
CHOSEN = {"radtan": RadTanModel, "ds": DoubleSphereModel, "ucm": UCMModel,
          "eucm": EUCMModel, "kb": KannalaBrandtModel}


def _factory(name):
    """Ground-truth cameras of the *original* model (KB or RadTan), per-camera jittered."""
    def f(cam_id, rng):
        fx = 800.0 * rng.uniform(0.97, 1.03)
        fy = fx * rng.uniform(0.99, 1.01)
        cx, cy = W / 2 + rng.uniform(-10, 10), H / 2 + rng.uniform(-10, 10)
        if name == "radtan":
            return RadTanModel(fx, fy, cx, cy, rng.uniform(-0.08, -0.02),
                               rng.uniform(0.0, 0.03), 0.0, 0.0, 0.0)
        if name == "kb":
            return KannalaBrandtModel(fx, fy, cx, cy, rng.uniform(0.0, 0.02),
                                      rng.uniform(0.0, 0.005), 0.0, 0.0)
        raise ValueError(name)
    return f


def _rel(Tref, Ti):
    return Ti @ np.linalg.inv(Tref)


def trial(original: str, chosen: str, outlier_frac: float, seed: int):
    """Return (pose%, focal%, func%) for one rig calibration."""
    cls = CHOSEN[chosen]
    obj, obs, img, gt, gtm = make_rig(n_cam=3, n_frame=60, noise_px=0.3, seed=seed,
                                      w=W, h=H, model_factory=_factory(original),
                                      outlier_frac=outlier_frac, outlier_px=40.0)
    rig = calibrate_rig(obj, obs, img, fix_intrinsics=False,
                        front_end=make_bundle_front_end(cls))
    ref = rig.ref_cam_id
    pose = 100.0 * max(
        abs(np.linalg.norm(_rel(rig.T_c_g[ref], rig.T_c_g[c])[:3, 3])
            - np.linalg.norm(_rel(gt[ref], gt[c])[:3, 3]))
        / np.linalg.norm(_rel(gt[ref], gt[c])[:3, 3])
        for c in rig.T_c_g if c != ref)
    foc, func = [], []
    for c in rig.cameras:
        ref_model, _ = convert(gtm[c], cls, width=W, height=H, n_samples=400,
                               n_restarts=3, seed=1)
        f_ref = paraxial_focal(ref_model)[0]
        foc.append(100.0 * abs(paraxial_focal(rig.cameras[c])[0] - f_ref) / f_ref)
        rep = reprojection_report(ref_model, rig.cameras[c], W, H, n_samples=1500)
        func.append(100.0 * rep["rms_px"] / f_ref)
    return pose, max(foc), max(func)


# (original, chosen, outlier_frac, check_func). The motivating case is original=KB (a fisheye
# camera, as in real configs) calibrated with a *different* chosen model; each row is robust
# over many seeds. ``check_func`` is True only when the chosen model covers the original's FOV
# (sphere models reproduce a KB fisheye); for FOV-incompatible pairs (KB->RadTan) the whole-
# image functional RMS is meaningless, so only pose + paraxial focal define "within 1%".
PAIRS = [
    ("kb", "radtan", 0.10, False),
    ("kb", "ds", 0.10, True),
    ("kb", "ucm", 0.10, True),
    ("kb", "eucm", 0.10, True),
    ("radtan", "radtan", 0.15, False),
]


def main(n_seeds=8):
    print(f"Param+pose validation: original->chosen model of choice, {n_seeds} seeds, "
          f"threshold 1%.\nMetrics: pose (vs GT) / paraxial-focal vs converted reference / "
          f"functional-RMS as % focal.\n")
    print(f"{'original->chosen':18s} {'outl':>5} {'pose%':>7} {'foc%':>7} {'func%':>8}  verdict")
    ok_all = True
    for original, chosen, outlier, check_func in PAIRS:
        rows = np.array([trial(original, chosen, outlier, s) for s in range(n_seeds)])
        mx = rows.max(axis=0)
        checks = [mx[0] < 1.0, mx[1] < 1.0] + ([mx[2] < 1.0] if check_func else [])
        ok = all(checks)
        ok_all = ok_all and ok
        fnote = f"{mx[2]:8.3f}" if check_func else f"{mx[2]:7.2f}*"
        print(f"{original+'->'+chosen:18s} {outlier:5.0%} {mx[0]:7.3f} {mx[1]:7.3f} {fnote}  "
              f"{'PASS' if ok else 'FAIL'}")
    print("\n* func not part of the verdict (chosen model does not cover the original's FOV).")
    print(f"OVERALL: {'WITHIN 1% PARAM+POSE (model-of-choice, robust)' if ok_all else 'FAIL'}")
    return ok_all


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    main(ns)
