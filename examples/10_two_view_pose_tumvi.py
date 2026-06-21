"""
Real-data two-view relative pose on a TUM-VI fisheye pair (companion to Chapter 8).

Picks two TUM-VI room1 cam0 frames a few steps apart, KLT-matches features between
them, unprojects both pixel sets through the loaded calibrated model to unit bearing
rays, and runs ds_msp.mvg.ransac_relative_pose. It reports the angular Sampson inlier
residual (radians, FOV-independent), the inlier count / ratio, and sanity-checks the
recovered translation direction against the mocap ground-truth baseline direction.

This reuses the KLT tracking + load_kalibr + mocap machinery already shipped and
validated in examples/09_monocular_vo_tumvi.py (imported, not duplicated).

Honest status: this DEMONSTRATES the pipeline end-to-end on a real fisheye stream. It is
NOT a sub-1e-3 deg pose guarantee — real correspondences carry noise. The deterministic
correctness claim lives in the synthetic round-trip (tests/mvg/test_two_view.py).

Run:
    python examples/10_two_view_pose_tumvi.py --start 400 --gap 4
"""

from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np

# Reuse the validated TUM-VI helpers from example 09 (filename starts with a digit, so
# it cannot be `import`ed by name — load it as a module via importlib).
_EX09 = os.path.join(os.path.dirname(__file__), "09_monocular_vo_tumvi.py")
_spec = importlib.util.spec_from_file_location("ex09", _EX09)
ex09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex09)

from ds_msp.io import load_kalibr  # noqa: E402
from ds_msp.mvg import (  # noqa: E402
    essential_from_rays,
    ransac_relative_pose,
    sampson_residual,
)

ROOM = ex09.ROOM


def match_two_frames(frames, i, j):
    """Pixel correspondences between frame i and frame j by shared KLT track id.

    `frames[k]` is a {track_id: (u, v)} dict (from ex09.track_features). Returns two
    (N, 2) pixel arrays in matching order.
    """
    a, b = frames[i], frames[j]
    common = sorted(set(a) & set(b))
    uv1 = np.array([a[t] for t in common], dtype=np.float64)  # (N, 2) pixels, frame i
    uv2 = np.array([b[t] for t in common], dtype=np.float64)  # (N, 2) pixels, frame j
    return uv1, uv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=400, help="first frame index in the CSV")
    ap.add_argument("--gap", type=int, default=4, help="frames between the two views")
    ap.add_argument("--threshold", type=float, default=0.005,
                    help="RANSAC Sampson inlier cutoff (radians)")
    args = ap.parse_args()

    model = load_kalibr(f"{ROOM}/dso/camchain.yaml", "cam0")
    print(f"camera: {type(model).__name__}  fx={model.fx:.2f} cx={model.cx:.2f}")

    # Track over a short window covering both frames, then match the two endpoints.
    rows = ex09.load_image_list(stride=1, num=args.gap + 1, start=args.start)
    frames = ex09.track_features(rows)
    uv1, uv2 = match_two_frames(frames, 0, args.gap)
    print(f"frames {args.start} -> {args.start + args.gap}: {len(uv1)} KLT matches")

    # Pixels -> unit bearing rays through the fisheye model (the whole point: rays, not pixels).
    f1, ok1 = model.unproject(uv1)   # f1: (N, 3) unit rays in camera 1
    f2, ok2 = model.unproject(uv2)   # f2: (N, 3) unit rays in camera 2
    ok = ok1 & ok2
    f1, f2 = f1[ok], f2[ok]
    print(f"valid bearing pairs: {int(ok.sum())} / {len(uv1)}")

    R, t, inliers = ransac_relative_pose(f1, f2, threshold=args.threshold, seed=0)

    # Re-fit E on the reported inliers so the printed Sampson residual matches the pose.
    E = essential_from_rays(f1[inliers], f2[inliers], normalize=True)
    res = sampson_residual(E, f1[inliers], f2[inliers])
    n_in = int(inliers.sum())
    print(f"\nRANSAC relative pose ({n_in} inliers / {len(f1)} = {100 * n_in / len(f1):.1f}%):")
    print(f"  inlier Sampson residual: median {np.median(res):.2e} rad  "
          f"max {res.max():.2e} rad  (~{np.degrees(np.median(res)):.3f} deg)")

    # Sanity-check translation DIRECTION against the mocap baseline (scale is unobservable).
    ts = [int(r[0]) for r in rows]
    gt = ex09.load_gt_centers([ts[0], ts[args.gap]])
    baseline = gt[1] - gt[0]
    if np.linalg.norm(baseline) > 1e-6:
        baseline /= np.linalg.norm(baseline)
        # t maps cam1->cam2; the camera-2 centre in cam-1 frame is -R.T @ t (a direction in
        # the camera frame, not world — so this is a coarse, not rigorous, sanity check).
        ang = np.degrees(np.arccos(np.clip(abs(t @ (R.T @ baseline)), -1, 1)))
        print(f"  recovered |t| direction vs mocap baseline: ~{ang:.1f} deg "
              f"(coarse; camera!=world frame)")


if __name__ == "__main__":
    main()
