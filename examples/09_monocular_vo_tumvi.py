"""
Tier-2 real-data validation: monocular VO on TUM-VI room1 vs mocap ground truth.

Prototype harness (NOT yet a library API): KLT-tracks features on the raw fisheye
cam0 stream, feeds the per-frame correspondences to ds_msp.vo.estimate_trajectory,
then aligns the recovered (up-to-scale) trajectory to the mocap0 ground truth with a
Sim(3) fit and reports ATE / rotation-RPE.

Run:
    python examples/09_monocular_vo_tumvi.py --stride 2 --num 200

Caveat: mocap0 measures the body/marker frame, so the camera↔body lever-arm (~7 cm)
puts a small floor under ATE; this validates trajectory *shape/scale* recovery, which
is the Tier-2 question. A rigorous lever-arm-corrected eval comes with Tier-3.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from ds_msp.io import load_kalibr
from ds_msp.vo import ate_rmse, estimate_trajectory

ROOM = "datasets/tumvi/dataset-room1_512_16"


def load_image_list(stride, num, start):
    rows = []
    with open(f"{ROOM}/mav0/cam0/data.csv") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ts, fname = line.split(",")
                rows.append((int(ts), fname))
    rows = rows[start::stride]
    if num:
        rows = rows[:num]
    return rows


def track_features(image_rows, max_corners=600, topup_below=350):
    """KLT-track features; return per-frame {track_id: (u,v)} dicts (forward-backward checked)."""
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    gf = dict(maxCorners=max_corners, qualityLevel=0.01, minDistance=10, blockSize=7)

    frames = []
    prev_gray = None
    prev_pts = None          # (M,1,2) float32
    prev_ids = None          # (M,) int
    next_id = 0

    for k, (_, fname) in enumerate(image_rows):
        img = cv2.imread(os.path.join(ROOM, "mav0/cam0/data", fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(fname)

        if prev_gray is None:
            pts = cv2.goodFeaturesToTrack(img, **gf)
            ids = np.arange(len(pts))
            next_id = len(pts)
        else:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, img, prev_pts, None, **lk)
            back, _, _ = cv2.calcOpticalFlowPyrLK(img, prev_gray, nxt, None, **lk)
            fb = np.linalg.norm((prev_pts - back).reshape(-1, 2), axis=1)
            ok = (st.reshape(-1) == 1) & (fb < 1.0)
            pts = nxt[ok]
            ids = prev_ids[ok]
            # top up with fresh corners when the track count gets low
            if len(pts) < topup_below:
                mask = np.full(img.shape, 255, np.uint8)
                for p in pts.reshape(-1, 2):
                    cv2.circle(mask, (int(p[0]), int(p[1])), 10, 0, -1)
                fresh = cv2.goodFeaturesToTrack(img, mask=mask, **gf)
                if fresh is not None:
                    new_ids = np.arange(next_id, next_id + len(fresh))
                    next_id += len(fresh)
                    pts = np.vstack([pts, fresh])
                    ids = np.concatenate([ids, new_ids])

        frames.append({int(i): (float(p[0]), float(p[1]))
                       for i, p in zip(ids, pts.reshape(-1, 2))})
        prev_gray, prev_pts, prev_ids = img, pts.reshape(-1, 1, 2).astype(np.float32), ids

    return frames


def load_gt_centers(cam_timestamps):
    data = np.loadtxt(f"{ROOM}/mav0/mocap0/data.csv", delimiter=",", comments="#")
    gt_ts = data[:, 0].astype(np.int64)
    gt_pos = data[:, 1:4]
    centers = []
    for ts in cam_timestamps:
        j = int(np.argmin(np.abs(gt_ts - ts)))
        centers.append(gt_pos[j])
    return np.array(centers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--start", type=int, default=400)
    args = ap.parse_args()

    model = load_kalibr(f"{ROOM}/dso/camchain.yaml", "cam0")
    print(f"camera: {type(model).__name__}  fx={model.fx:.2f} cx={model.cx:.2f}")

    rows = load_image_list(args.stride, args.num, args.start)
    print(f"frames: {len(rows)} (stride {args.stride}, start {args.start})")
    frames = track_features(rows)
    tracks = [len(f) for f in frames]
    print(f"tracks/frame: min {min(tracks)} median {int(np.median(tracks))} max {max(tracks)}")

    res = estimate_trajectory(model, frames, min_common=8, threshold=0.01)
    est_centers = res.centers

    gt_centers = load_gt_centers([ts for ts, _ in rows])
    path_len = float(np.linalg.norm(np.diff(gt_centers, axis=0), axis=1).sum())

    ate = ate_rmse(est_centers, gt_centers, align=True)
    print(f"\nGT path length:   {path_len:.3f} m")
    print(f"ATE (Sim3) RMSE:  {ate:.4f} m   ({100 * ate / max(path_len, 1e-9):.1f}% of path)")
    print(f"landmarks:        {len(res.landmarks)}")


if __name__ == "__main__":
    main()
