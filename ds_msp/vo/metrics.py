"""
Trajectory-evaluation metrics for visual odometry (Tier 2).

The standard way to score a VO/SLAM trajectory against ground truth:

- **Sim(3) Umeyama alignment** (`align_sim3`) — a monocular trajectory is only
  recovered *up to a global similarity* (scale + rigid transform), so before any
  error metric we align the estimate to ground truth by the closed-form Umeyama
  least-squares similarity (Umeyama 1991).
- **ATE** (`ate_rmse`) — absolute trajectory error: RMSE of the aligned camera
  centres. The headline "how far off is the whole path" number.
- **RPE** (`rpe_rmse`) — relative pose error over a fixed step: drift per segment,
  scale-free, the local-consistency complement to ATE.

All pure NumPy and model-agnostic — they take camera centres / 4×4 poses, nothing else.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def align_sim3(src: np.ndarray, dst: np.ndarray, *, with_scale: bool = True
               ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Closed-form similarity that best maps ``src`` onto ``dst`` (Umeyama 1991).

    Finds ``s, R, t`` minimising ``Σ ‖dst_i − (s·R·src_i + t)‖²`` for point sets
    ``src, dst`` of shape ``(N, 3)``.

    Returns ``(s, R, t)`` with ``s`` scalar scale, ``R`` (3, 3) rotation, ``t`` (3,).
    With ``with_scale=False`` the scale is fixed to 1 (rigid SE(3) alignment).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must both be (N, 3) and equal-shaped")
    n = src.shape[0]

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d

    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_s = (sc ** 2).sum() / n
        s = float((D * np.diag(S)).sum() / var_s) if var_s > 0 else 1.0
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3(s: float, R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a similarity ``s·R·pts + t`` to an ``(N, 3)`` point set."""
    return (s * (np.asarray(pts, dtype=np.float64) @ np.asarray(R).T)) + np.asarray(t)


def ate_rmse(est: np.ndarray, gt: np.ndarray, *, align: bool = True,
             with_scale: bool = True) -> float:
    """Absolute Trajectory Error (RMSE of camera centres), Sim(3)-aligned by default.

    ``est, gt`` are ``(N, 3)`` camera-centre trajectories in correspondence.
    """
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if align:
        s, R, t = align_sim3(est, gt, with_scale=with_scale)
        est = apply_sim3(s, R, t, est)
    d = est - gt
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def rpe_rmse(est_poses: np.ndarray, gt_poses: np.ndarray, *, delta: int = 1
             ) -> Tuple[float, float]:
    """Relative Pose Error over a fixed step ``delta``.

    ``est_poses, gt_poses`` are ``(N, 4, 4)`` camera-to-world poses in correspondence.
    For each ``i`` the relative motion ``inv(P_i)·P_{i+delta}`` is compared between
    estimate and ground truth; the error ``E_i = inv(ΔGT)·ΔEST``.

    Returns ``(trans_rmse, rot_rmse_deg)`` — translational RMSE (same units as the
    poses) and rotational RMSE in degrees.
    """
    est_poses = np.asarray(est_poses, dtype=np.float64)
    gt_poses = np.asarray(gt_poses, dtype=np.float64)
    n = len(est_poses)
    if n != len(gt_poses):
        raise ValueError("est_poses and gt_poses must have equal length")
    if delta < 1 or delta >= n:
        raise ValueError(f"delta must be in [1, {n - 1}], got {delta}")

    trans_sq, rot_sq = [], []
    for i in range(n - delta):
        d_est = np.linalg.inv(est_poses[i]) @ est_poses[i + delta]
        d_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[i + delta]
        err = np.linalg.inv(d_gt) @ d_est
        trans_sq.append((err[:3, 3] ** 2).sum())
        cos = (np.trace(err[:3, :3]) - 1.0) / 2.0
        rot_sq.append(np.arccos(np.clip(cos, -1.0, 1.0)) ** 2)
    trans_rmse = float(np.sqrt(np.mean(trans_sq)))
    rot_rmse_deg = float(np.degrees(np.sqrt(np.mean(rot_sq))))
    return trans_rmse, rot_rmse_deg
