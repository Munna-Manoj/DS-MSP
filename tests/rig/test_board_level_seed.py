"""Rig seeding must keep target errors out of the lens, and each camera's own lens intact."""
import copy

import numpy as np
import pytest

from ds_msp.detect.charuco import BoardSpec
from ds_msp.core.lie import so3_exp
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig.calibrate import _board_views, _fit_one_camera, make_bundle_front_end
from ds_msp.rig.reconstruct import reconstruct_objects
from ds_msp.rig.types import BoardObs, Object3D, ObjectObs

pytestmark = pytest.mark.req("FR-RIG-001", "FR-RIG-003", "FR-CALIB-010")


def _cube_object():
    """Two perpendicular boards fused into one object, with exact geometry."""
    x, y = np.meshgrid(np.arange(1, 8) * 0.05, np.arange(1, 6) * 0.05)
    plane = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    second = plane @ np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]]).T + (0.5, 0, 0.1)
    pts = np.vstack([plane, second])
    n = len(plane)
    rows = np.array([(0, k) for k in range(n)] + [(1, k) for k in range(n)], int)
    return Object3D(object_id=0, board_ids=[0, 1], ref_board_id=0,
                    T_co_b={0: np.eye(4), 1: np.eye(4)}, pts_3d=pts, pts_obj_2_board=rows,
                    pts_board_2_obj={(int(b), int(k)): i for i, (b, k) in enumerate(rows)})


def _observations(obj, model, rng, cam=0, n=40, w=1600, h=1200):
    obs = []
    centre = obj.pts_3d.mean(0)
    frame = 0
    while len(obs) < n and frame < 5000:
        frame += 1
        R = so3_exp(rng.uniform(-0.9, 0.9, 3))
        t = so3_exp(rng.uniform(-1.3, 1.3, 3)) @ np.array([0, 0, 1.0]) * rng.uniform(0.5, 1.3) - R @ centre
        Xc = obj.pts_3d @ R.T + t
        uv, ok = model.project(Xc)
        uv = np.asarray(uv, float)
        ok = np.asarray(ok, bool).ravel() & (uv[:, 0] > 2) & (uv[:, 0] < w - 3) & (uv[:, 1] > 2) & (uv[:, 1] < h - 3)
        if ok.sum() < 20:
            continue
        rows = np.flatnonzero(ok)
        obs.append(ObjectObs(cam_id=cam, frame_id=frame, object_id=0, point_rows=rows,
                             pts_2d=uv[rows] + rng.normal(0, 0.1, (len(rows), 2))))
    return obs


def test_views_split_along_printed_boards():
    obj = _cube_object()
    n = len(obj.pts_3d)
    obs = ObjectObs(cam_id=0, frame_id=0, object_id=0, point_rows=np.arange(n),
                    pts_2d=np.zeros((n, 2)))
    points, pixels, _ = _board_views({0: obj}, [obs])
    assert len(points) == 2 and [len(p) for p in points] == [n // 2, n // 2]


def test_inter_board_error_does_not_reach_the_lens():
    """Fused geometry is an estimate; a wrong inter-board pose must not change the lens fit."""
    obj = _cube_object()
    true = KannalaBrandtModel(380, 380, 806, 594, 0.02, -0.004, 0.0005, 0.0)
    obs = _observations(obj, true, np.random.default_rng(1))
    exact = _fit_one_camera(KannalaBrandtModel, {0: obj}, obs, 1600, 1200, None, "soft_l1", 1.0, 80)
    wrong = copy.deepcopy(obj)
    n = len(wrong.pts_3d) // 2
    wrong.pts_3d[n:] = wrong.pts_3d[n:] @ so3_exp(np.array([0.06, -0.04, 0.05])).T + (0.05, -0.04, 0.03)
    moved = _fit_one_camera(KannalaBrandtModel, {0: wrong}, obs, 1600, 1200, None, "soft_l1", 1.0, 80)
    assert exact["fit"]["seed"] == "radial_boards" and moved["fit"]["seed"] == "radial_boards"
    np.testing.assert_allclose(moved["model"].params, exact["model"].params, atol=1e-3)


def test_mixed_lens_rig_keeps_each_camera_lens():
    """A rig median across different lenses used to reset a correct camera to a wrong focal."""
    obj = _cube_object()
    wide = KannalaBrandtModel(380, 380, 806, 594, 0.02, -0.004, 0.0005, 0.0)
    narrow = RadTanModel(1100, 1100, 805, 595, -0.12, 0.03, 0.0004, -0.0003, 0.0)
    rng = np.random.default_rng(2)
    by_cam = {0: _observations(obj, wide, rng, 0), 1: _observations(obj, wide, rng, 1),
              2: _observations(obj, narrow, rng, 2)}
    size = {c: (1600, 1200) for c in by_cam}
    fe = make_bundle_front_end({0: "kb", 1: "kb", 2: "radtan"}, n_jobs=1)
    cams = fe({0: obj}, by_cam, size)
    assert fe.report[2]["fallback"] is None
    assert abs(cams[2].params[0] / 1100 - 1) < 0.02


def test_reconstruction_with_the_configured_lens_is_accurate_on_a_fisheye():
    true = DoubleSphereModel(330, 330, 803, 596, -0.2, 0.6)
    obj = _cube_object()
    spec = [BoardSpec(n_x=8, n_y=6, length_square=0.04, length_marker=0.03, square_size=0.05)] * 2
    rng = np.random.default_rng(3)
    board_obs = []
    for o in _observations(obj, true, rng, n=50):
        rows = obj.pts_obj_2_board[o.point_rows]
        for b in (0, 1):
            m = rows[:, 0] == b
            if m.sum() >= 8:
                board_obs.append(BoardObs(cam_id=0, frame_id=o.frame_id, board_id=b,
                                          corner_ids=rows[m, 1], pts_2d=o.pts_2d[m]))
    objects, _ = reconstruct_objects(board_obs, spec, {0: (1600, 1200)}, model_spec={0: "ds"})
    assert len(objects) == 1 and sorted(objects[0].board_ids) == [0, 1]
    fused = objects[0]
    # Same corner in both clouds, then a best rigid fit, as the object frame is arbitrary.
    want = np.array([obj.pts_3d[obj.pts_board_2_obj[(int(b), int(k))]]
                     for b, k in fused.pts_obj_2_board])
    a, b = fused.pts_3d - fused.pts_3d.mean(0), want - want.mean(0)
    U, _, Vt = np.linalg.svd(b.T @ a)
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    assert np.linalg.norm(a @ R.T - b, axis=1).max() < 3e-3   # metres, on a 0.5 m target


def test_gross_outliers_do_not_collapse_the_focal():
    """Blunders can be absorbed by a tiny focal at a low residual; that must not win."""
    obj = _cube_object()
    true = RadTanModel(900, 900, 805, 595, -0.06, 0.01, 0.0, 0.0, 0.0)
    rng = np.random.default_rng(5)
    obs = _observations(obj, true, rng, n=40)
    for o in obs:
        m = rng.random(len(o.pts_2d)) < 0.15
        angle = rng.uniform(0, 2 * np.pi, m.sum())
        o.pts_2d[m] += 40.0 * np.column_stack([np.cos(angle), np.sin(angle)])
    r = _fit_one_camera(RadTanModel, {0: obj}, obs, 1600, 1200, None, "soft_l1", 1.0, 80)
    assert abs(r["model"].params[0] / 900 - 1) < 0.05
