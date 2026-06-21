"""Synthetic end-to-end test for monocular VO (Tier 2).

Generates a known trajectory + 3D cloud, projects through a real DS-MSP camera model,
and asserts the recovered trajectory matches ground truth (up to the global similarity
that monocular VO can never observe) — the "prove a number" gate.
"""

import numpy as np
import pytest

from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.vo.metrics import ate_rmse, rpe_rmse
from ds_msp.vo.odometry import estimate_trajectory


def _Ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _synthetic(model, width=512, height=512, n_pts=250, n_frames=8, noise_px=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X_w = rng.uniform([-2.0, -2.0, 4.0], [2.0, 2.0, 8.0], (n_pts, 3))

    poses, frames = [], []
    for k in range(n_frames):
        R_wc = _Ry(0.03 * k)                 # gentle yaw
        c = np.array([0.25 * k, 0.02 * k, 0.0])
        T = np.eye(4)
        T[:3, :3] = R_wc
        T[:3, 3] = c
        poses.append(T)

        X_cam = (X_w - c) @ R_wc             # R_cw (X_w - c), R_cw = R_wc.T
        px, valid = model.project(X_cam)
        in_front = X_cam[:, 2] > 0
        inb = (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
        keep = valid & in_front & inb
        if noise_px:
            px = px + rng.normal(0, noise_px, px.shape)
        frames.append({i: tuple(px[i]) for i in np.nonzero(keep)[0]})

    return np.stack(poses), frames


def test_noise_free_trajectory_recovered_exactly():
    model = KannalaBrandtModel(190.0, 190.0, 255.0, 255.0, 0.02, 0.01, 0.0, 0.0)
    gt_poses, frames = _synthetic(model, n_frames=8, noise_px=0.0)

    res = estimate_trajectory(model, frames)
    assert res.poses.shape == gt_poses.shape

    ate = ate_rmse(res.centers, gt_poses[:, :3, 3], align=True)
    assert ate < 1e-6, f"noise-free ATE too high: {ate}"

    # rotational RPE is scale-free → should be ~0 even though translation is up-to-scale
    _, rot_rpe = rpe_rmse(res.poses, gt_poses, delta=1)
    assert rot_rpe < 1e-3, f"rotation RPE too high: {rot_rpe} deg"

    # landmarks were triangulated for the persistent cloud
    assert len(res.landmarks) > 100


def test_modest_noise_keeps_trajectory_accurate():
    model = KannalaBrandtModel(190.0, 190.0, 255.0, 255.0, 0.02, 0.01, 0.0, 0.0)
    gt_poses, frames = _synthetic(model, n_frames=8, noise_px=0.3, seed=3)

    res = estimate_trajectory(model, frames)
    # absolute trajectory error stays small relative to the ~1.75 m path length
    ate = ate_rmse(res.centers, gt_poses[:, :3, 3], align=True)
    assert ate < 0.1, f"noisy ATE too high: {ate}"


def test_too_few_correspondences_raises():
    model = KannalaBrandtModel(190.0, 190.0, 255.0, 255.0, 0.0, 0.0, 0.0, 0.0)
    frames = [{0: (1.0, 2.0)}, {0: (1.1, 2.1)}]
    with pytest.raises(ValueError, match="correspondences"):
        estimate_trajectory(model, frames)
