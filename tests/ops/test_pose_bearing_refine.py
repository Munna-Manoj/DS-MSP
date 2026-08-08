"""Public PnP consensus refinement must stay camera-neutral and full-sphere.

These are new regressions for the ``refine`` contract. Before the guarded bearing polish was
added, every adequately-sized call selected the bearing RANSAC engine but ignored ``refine``;
``refine=True`` and ``False`` therefore returned bit-identical linear estimates even on noisy
data. The fixtures cover both an ordinary forward view and an all-peripheral negative-z view.
"""

import cv2
import numpy as np
import pytest

import ds_msp.geometry.resection as resection
from ds_msp.core.lie import so3_exp
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.ops import solve_pnp_ransac


def _pose_error(rvec, tvec, R_gt, t_gt):
    R = cv2.Rodrigues(rvec)[0]
    rotation = np.degrees(
        np.arccos(np.clip((np.trace(R.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0))
    )
    return float(rotation), float(np.linalg.norm(tvec - t_gt))


def _forward_scene(seed=1):
    model = DoubleSphereModel.sample()
    rng = np.random.default_rng(seed)
    R_gt = so3_exp([0.08, -0.12, 0.04])
    t_gt = np.array([0.1, -0.05, 2.5])
    X = rng.uniform([-0.7, -0.5, -0.3], [0.7, 0.5, 0.3], (60, 3))
    uv, valid = model.project(X @ R_gt.T + t_gt)
    X, uv = X[valid], np.asarray(uv)[valid]
    uv = uv + rng.normal(0.0, 1.5, uv.shape)
    outliers = rng.choice(len(X), 10, replace=False)
    uv[outliers] += rng.uniform(-80.0, 80.0, (len(outliers), 2))
    return model, X, uv, R_gt, t_gt


def _peripheral_scene(seed=1):
    model = DoubleSphereModel.sample()
    rng = np.random.default_rng(seed)
    R_gt = so3_exp([0.2, -0.3, 0.15])
    t_gt = np.array([0.1, -0.05, -0.2])
    theta = np.radians(rng.uniform(95.0, 110.0, 100))
    phi = rng.uniform(0.0, 2.0 * np.pi, 100)
    directions = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    camera_points = directions * rng.uniform(0.6, 3.0, 100)[:, None]
    X = (camera_points - t_gt) @ R_gt
    uv, valid = model.project(camera_points)
    X, uv = X[valid], np.asarray(uv)[valid]
    uv = uv + np.random.default_rng(9004).normal(0.0, 1.0, uv.shape)
    return model, X, uv, R_gt, t_gt


def _solve_pair(model, X, uv, *, seed, threshold=4.0):
    common = dict(thresh_px=threshold, max_iters=500, seed=seed)
    raw = solve_pnp_ransac(model, X, uv, refine=False, **common)
    polished = solve_pnp_ransac(model, X, uv, refine=True, **common)
    assert raw[0] and polished[0]
    return raw, polished


def test_refine_flag_polishes_noisy_forward_bearings():
    model, X, uv, R_gt, t_gt = _forward_scene()
    rays, valid = model.unproject(uv)
    assert valid.all() and (rays[:, 2] > 0.0).all()

    raw, polished = _solve_pair(model, X, uv, seed=1)
    raw_rot, raw_trans = _pose_error(raw[1], raw[2], R_gt, t_gt)
    ref_rot, ref_trans = _pose_error(polished[1], polished[2], R_gt, t_gt)

    assert not np.array_equal(polished[1], raw[1]), "refine=True must not be ignored"
    assert polished[3].sum() >= raw[3].sum()
    assert ref_rot < 0.5 * raw_rot
    assert ref_trans < 0.3 * raw_trans


def test_refine_uses_negative_z_consensus_bearings():
    model, X, uv, R_gt, t_gt = _peripheral_scene()
    rays, valid = model.unproject(uv)
    peripheral = valid & (rays[:, 2] <= 0.0)
    assert peripheral.all()

    raw, polished = _solve_pair(model, X, uv, seed=1, threshold=4.0)
    raw_rot, raw_trans = _pose_error(raw[1], raw[2], R_gt, t_gt)
    ref_rot, ref_trans = _pose_error(polished[1], polished[2], R_gt, t_gt)

    assert raw[3].all() and polished[3].all()
    assert peripheral[polished[3]].all()
    assert polished[3].sum() >= raw[3].sum()
    assert ref_rot < 0.7 * raw_rot
    assert ref_trans < 0.8 * raw_trans


@pytest.mark.parametrize("failure", ["lower_support", "worse_score"])
def test_harmful_bearing_polish_preserves_supported_ransac_pose(monkeypatch, failure):
    model = DoubleSphereModel.sample()
    rng = np.random.default_rng(21)
    R_gt = so3_exp([0.05, -0.1, 0.02])
    t_gt = np.array([0.1, -0.05, 2.2])
    X = rng.uniform([-0.4, -0.4, -0.2], [0.4, 0.4, 0.2], (40, 3))
    uv, valid = model.project(X @ R_gt.T + t_gt)
    assert valid.all()

    kwargs = dict(thresh_px=20.0, max_iters=100, seed=0)
    baseline = solve_pnp_ransac(model, X, uv, refine=False, **kwargs)
    assert baseline[0] and baseline[3].all()
    T_bad = np.eye(4)
    T_bad[:3, :3] = cv2.Rodrigues(baseline[1])[0]
    T_bad[:3, 3] = baseline[2]
    T_bad[:3, 3] += (
        np.array([0.0, 0.0, -100.0])
        if failure == "lower_support"
        else np.array([0.002, 0.0, 0.0])
    )

    monkeypatch.setattr(resection, "refine_pose_bearings", lambda *args, **kwargs: T_bad)
    polished = solve_pnp_ransac(model, X, uv, refine=True, **kwargs)

    assert polished[0]
    np.testing.assert_array_equal(polished[1], baseline[1])
    np.testing.assert_array_equal(polished[2], baseline[2])
    np.testing.assert_array_equal(polished[3], baseline[3])


def test_numerically_failed_bearing_polish_preserves_supported_pose(monkeypatch):
    model, X, uv, _, _ = _forward_scene()
    kwargs = dict(thresh_px=4.0, max_iters=100, seed=0)
    baseline = solve_pnp_ransac(model, X, uv, refine=False, **kwargs)
    assert baseline[0]

    def fail(*_args, **_kwargs):
        raise np.linalg.LinAlgError("synthetic LM failure")

    monkeypatch.setattr(resection, "refine_pose_bearings", fail)
    polished = solve_pnp_ransac(model, X, uv, refine=True, **kwargs)

    assert polished[0]
    for actual, expected in zip(polished[1:], baseline[1:]):
        np.testing.assert_array_equal(actual, expected)


pytestmark = pytest.mark.req("FR-OPS-003")
