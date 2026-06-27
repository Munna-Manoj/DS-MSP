"""
Monocular visual odometry (Tier 2).

A fisheye measures *rays*, so this VO never undistorts to a pinhole — it composes the
Tier-1 bearing-vector stack directly:

  unproject → two-view relative pose → ray triangulation → scale-propagated chaining

Each consecutive frame pair gives a relative pose with a **unit-norm** translation
(monocular scale is unobservable from two views). We fix the global scale once on the first
pair, then **propagate** it: triangulated landmarks shared across an overlapping triple tie
each new pair's unit translation to the established metric, so the chained trajectory is
self-consistent up to a single global similarity (recovered at evaluation by `align_sim3`).

This first increment runs on **given correspondences** (per-frame ``{landmark_id: pixel}``
dicts) — exact on noise-free synthetic data. Wiring a real feature tracker (KLT) and
reporting ATE on TUM-VI/EuRoC is the next increment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np

from ..mvg.bundle import estimate_relative_pose
from ..mvg.two_view import triangulate_rays

Frame = Mapping[int, Sequence[float]]  # {landmark_id: (u, v)}

__all__ = ["VOResult", "estimate_trajectory"]


@dataclass
class VOResult:
    """Output of :func:`estimate_trajectory`."""
    poses: np.ndarray                       # (N, 4, 4) camera-to-world per frame
    landmarks: Dict[int, np.ndarray] = field(default_factory=dict)  # id -> world xyz

    @property
    def centers(self) -> np.ndarray:
        """(N, 3) camera centres in world frame."""
        return self.poses[:, :3, 3].copy()


def _rel_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def estimate_trajectory(model, frames: Sequence[Frame], *, min_common: int = 8,
                        threshold: float = 0.005, seed: int = 0) -> VOResult:
    """Estimate a monocular camera trajectory from per-frame correspondences.

    Parameters
    ----------
    model : CameraModel
        Any central DS-MSP model (its ``unproject`` lifts pixels to bearing rays).
    frames : sequence of mapping
        One ``{landmark_id: (u, v)}`` per frame; ids link the same 3D point across frames.
    min_common : int
        Minimum shared correspondences required between consecutive frames (≥ 8 for the
        eight-point estimator).
    threshold, seed :
        Forwarded to the robust two-view estimator (angular RANSAC threshold / RNG seed).

    Returns
    -------
    VOResult
        ``poses`` (N, 4, 4) camera-to-world (frame 0 is the world origin, global scale 1),
        and the triangulated ``landmarks`` map.
    """
    n = len(frames)
    if n < 2:
        raise ValueError("need at least 2 frames")

    poses = [np.eye(4)]                      # T_wc[0] = identity (world = frame 0)
    landmarks: Dict[int, np.ndarray] = {}

    for k in range(n - 1):
        fa, fb = frames[k], frames[k + 1]
        common = sorted(set(fa) & set(fb))
        if len(common) < min_common:
            raise ValueError(
                f"frames {k}->{k + 1} share only {len(common)} correspondences "
                f"(need ≥ {min_common})"
            )
        px1 = np.array([fa[i] for i in common], dtype=np.float64)
        px2 = np.array([fb[i] for i in common], dtype=np.float64)
        f1, _ = model.unproject(px1)
        f2, _ = model.unproject(px2)

        # Relative pose (R, t): X_cam{k+1} = R · X_cam{k} + t, with ‖t‖ = 1.
        R, t, _, _ = estimate_relative_pose(f1, f2, threshold=threshold, seed=seed)
        # Re-triangulate the full common set (in camera-k frame) at unit translation scale.
        X_unit, d1, d2 = triangulate_rays(f1, f2, R, t)
        front = (d1 > 0) & (d2 > 0)

        # --- resolve the scale of this pair's unit translation ---
        T_wc_k = poses[k]
        T_cw_k = np.linalg.inv(T_wc_k)
        known = [(j, i) for j, i in enumerate(common)
                 if i in landmarks and front[j]]
        if known:
            # Existing landmarks brought into camera-k frame lie on the same rays as the
            # unit-scale triangulation → ratio of distances along the ray = the scale.
            idx = [j for j, _ in known]
            X_known_camk = _transform_points(T_cw_k, np.array([landmarks[i] for _, i in known]))
            num = np.linalg.norm(X_known_camk, axis=1)
            den = np.linalg.norm(X_unit[idx], axis=1)
            scale = float(np.median(num / np.maximum(den, 1e-12)))
        else:
            scale = 1.0                      # first pair fixes the global gauge

        # Chain the (scaled) relative pose: T_wc[k+1] = T_wc[k] · inv(T_{k+1<-k}).
        T_rel = _rel_transform(R, scale * t)
        T_wc_kp1 = T_wc_k @ np.linalg.inv(T_rel)
        poses.append(T_wc_kp1)

        # Add newly-seen landmarks (scaled, in world frame).
        X_world = _transform_points(T_wc_k, scale * X_unit)
        for j, i in enumerate(common):
            if front[j] and i not in landmarks:
                landmarks[i] = X_world[j]

    return VOResult(poses=np.stack(poses), landmarks=landmarks)
