#!/usr/bin/env python3
"""
Chapter 8, section 6 demo -- two-view pose from a real TUM-VI fisheye pair.

Synthetic rays are noise-free; real ones are not. This script runs the exact same
pipeline as the synthetic exercises in docs/learn/08_two_view_geometry_on_rays.md,
but on two real TUM-VI room1 fisheye frames a few frames apart:

  1. KLT-track features between the two raw fisheye frames (same tracker as
     examples/09_monocular_vo_tumvi.py -- duplicated here so this script stays
     single-file runnable).
  2. Unproject both matched pixel sets through the *loaded, calibrated* camera model
     to bearing vectors (rays) -- ds_msp.mvg's essential-matrix / RANSAC machinery
     is ray-based, not pixel-based.
  3. Run ransac_relative_pose on the ray correspondences to recover (R, t) and the
     inlier mask.

Prerequisites:
  - `pip install -e .`
  - TUM-VI room1 sequence: `bash scripts/download_datasets.sh tumvi`

Run:
  python examples/11_two_view_pose_tumvi.py --start 400 --gap 4
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from ds_msp.io import load_kalibr
from ds_msp.mvg import ransac_relative_pose, sampson_residual

ROOM = "datasets/tumvi/dataset-room1_512_16"


def load_frame_list():
    rows = []
    with open(f"{ROOM}/mav0/cam0/data.csv") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ts, fname = line.split(",")
                rows.append((int(ts), fname))
    return rows


def mocap_position_at(ts):
    """Nearest mocap0 position sample to a cam0 timestamp (same lookup as example 09)."""
    data = np.loadtxt(f"{ROOM}/mav0/mocap0/data.csv", delimiter=",", comments="#")
    j = int(np.argmin(np.abs(data[:, 0].astype(np.int64) - ts)))
    return data[j, 1:4]


def klt_match(img1, img2, max_corners=600):
    """Two-frame KLT match with a forward-backward consistency check."""
    gf = dict(maxCorners=max_corners, qualityLevel=0.01, minDistance=10, blockSize=7)
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    pts1 = cv2.goodFeaturesToTrack(img1, **gf)
    pts2, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, pts1, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(img2, img1, pts2, None, **lk)
    fb = np.linalg.norm((pts1 - back).reshape(-1, 2), axis=1)
    ok = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.0)
    return pts1[ok].reshape(-1, 2), pts2[ok].reshape(-1, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=400)
    ap.add_argument("--gap", type=int, default=4)
    args = ap.parse_args()
    if not os.path.isdir(ROOM):
        raise SystemExit("Fetch TUM-VI first: bash scripts/download_datasets.sh tumvi")

    model = load_kalibr(f"{ROOM}/dso/camchain.yaml", "cam0")
    print(f"camera: {type(model).__name__}  fx={model.fx:.2f} cx={model.cx:.2f}")

    rows = load_frame_list()
    ts1, f1_name = rows[args.start]
    ts2, f2_name = rows[args.start + args.gap]
    img1 = cv2.imread(os.path.join(ROOM, "mav0/cam0/data", f1_name), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(ROOM, "mav0/cam0/data", f2_name), cv2.IMREAD_GRAYSCALE)

    uv1, uv2 = klt_match(img1, img2)
    print(f"frames {args.start} -> {args.start + args.gap}: {len(uv1)} KLT matches")

    ray1, valid1 = model.unproject(uv1)
    ray2, valid2 = model.unproject(uv2)
    valid = valid1 & valid2
    ray1, ray2 = ray1[valid], ray2[valid]
    print(f"valid bearing pairs: {len(ray1)} / {len(uv1)}")

    R, t, inliers = ransac_relative_pose(ray1, ray2, threshold=0.005)
    n_in = int(inliers.sum())
    print(f"\nRANSAC relative pose ({n_in} inliers / {len(ray1)} = "
          f"{100 * n_in / len(ray1):.1f}%):")

    # Sampson residual on the inliers only -- how well they satisfy f2^T E f1 = 0.
    t_hat = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = t_hat @ R
    resid = sampson_residual(E, ray1[inliers], ray2[inliers])
    print(f"  inlier Sampson residual: median {np.median(resid):.2e} rad  "
          f"max {resid.max():.2e} rad  (~{np.degrees(np.median(resid)):.3f} deg)")

    # Coarse sanity check: does the recovered (unscaled) t point roughly where the mocap
    # baseline says the camera actually moved? Camera and mocap/body frames are NOT aligned
    # here, so this is a rough consistency check, not a rigorous frame-correct evaluation
    # (Chapter 9 builds that).
    p1, p2 = mocap_position_at(ts1), mocap_position_at(ts2)
    baseline = p2 - p1
    cos_ang = np.dot(t, baseline) / (np.linalg.norm(t) * np.linalg.norm(baseline) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
    print(f"  recovered |t| direction vs mocap baseline: ~{ang:.1f} deg "
          f"(coarse; camera != world frame)")


if __name__ == "__main__":
    main()
