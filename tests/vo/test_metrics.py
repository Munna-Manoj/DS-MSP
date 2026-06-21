"""Tests for VO trajectory metrics (Tier 2)."""

import numpy as np
from scipy.spatial.transform import Rotation

from ds_msp.vo.metrics import align_sim3, apply_sim3, ate_rmse, rpe_rmse


def test_align_sim3_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.uniform(-5, 5, (50, 3))
    s_true = 2.37
    R_true = Rotation.from_rotvec([0.3, -0.7, 0.2]).as_matrix()
    t_true = np.array([1.0, -2.0, 0.5])
    dst = apply_sim3(s_true, R_true, t_true, src)

    s, R, t = align_sim3(src, dst)
    assert abs(s - s_true) < 1e-9
    np.testing.assert_allclose(R, R_true, atol=1e-9)
    np.testing.assert_allclose(t, t_true, atol=1e-9)


def test_ate_zero_for_similarity_transformed_trajectory():
    rng = np.random.default_rng(1)
    gt = np.cumsum(rng.uniform(-1, 1, (30, 3)), axis=0)
    # an estimate that differs only by a global similarity should align to ATE ~ 0
    est = apply_sim3(0.5, Rotation.from_rotvec([0.1, 0.2, -0.3]).as_matrix(),
                     np.array([3.0, 1.0, -2.0]), gt)
    assert ate_rmse(est, gt, align=True) < 1e-9
    # without alignment the error is large
    assert ate_rmse(est, gt, align=False) > 1.0


def test_rpe_zero_for_identical_poses_and_scale_invariant_rotation():
    rng = np.random.default_rng(2)
    poses = []
    for i in range(10):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(rng.uniform(-0.5, 0.5, 3)).as_matrix()
        T[:3, 3] = rng.uniform(-2, 2, 3)
        poses.append(T)
    poses = np.stack(poses)

    tr, rot = rpe_rmse(poses, poses, delta=1)
    assert tr < 1e-12 and rot < 1e-9

    # scaling all camera centres leaves rotational RPE untouched
    scaled = poses.copy()
    scaled[:, :3, 3] *= 3.0
    _, rot_scaled = rpe_rmse(scaled, poses, delta=1)
    assert rot_scaled < 1e-9
