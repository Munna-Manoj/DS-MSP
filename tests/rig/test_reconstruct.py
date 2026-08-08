"""Multi-board fused-object reconstruction from raw detections (MC-Calib's
``calibrate3DObjects``, McCalib.cpp:765-956).

A 3-board rigid object viewed by a 2-camera rig: synthesize per-board ChArUco detections
(no object given), reconstruct the fused object, and check the recovered geometry matches
ground truth and that observations map onto it. This closes the parity gap where
``number_board > 1`` previously required a pre-built ``calibrated_objects_data.yml``.
"""
import numpy as np
import pytest

from ds_msp.calib.charuco import BoardSpec, board_object_points
from ds_msp.core.lie import so3_exp
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig.reconstruct import (object_obs_from_board_obs, reconstruct_object,
                                     reconstruct_objects)
from ds_msp.rig.types import BoardObs

W, H, F = 1280, 960, 800.0


def _T(rvec, t):
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.asarray(rvec, float))
    T[:3, 3] = t
    return T


def _make_board_obs(noise_px=0.1, seed=0):
    specs = [BoardSpec(5, 5, 0.04, 0.03, 0.1) for _ in range(3)]
    # ground-truth board->object poses; board 0 = object frame (its min id is the reference)
    T_co_b = {0: np.eye(4),
              1: _T([0.0, 0.6, 0.0], [0.45, 0.0, 0.25]),
              2: _T([-0.6, 0.0, 0.0], [0.0, 0.45, 0.25])}
    bp = {b: board_object_points(specs[b]) for b in range(3)}
    gt_pts = {}
    for b in range(3):
        Ph = np.c_[bp[b], np.ones(len(bp[b]))]
        gt_pts[b] = (T_co_b[b] @ Ph.T).T[:, :3]

    model = RadTanModel(F, F, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
    cams = {0: np.eye(4), 1: _T([0.0, 0.14, 0.0], [0.15, 0.0, 0.0])}
    rng = np.random.default_rng(seed)
    board_obs = []
    for fr in range(40):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        Tg = _T(axis * rng.uniform(-0.5, 0.5),
                [rng.uniform(-0.25, 0.25), rng.uniform(-0.2, 0.2), rng.uniform(1.9, 2.5)])
        for c, Tcg in cams.items():
            for b in range(3):
                Xg = (Tg[:3, :3] @ gt_pts[b].T).T + Tg[:3, 3]
                Xc = (Tcg[:3, :3] @ Xg.T).T + Tcg[:3, 3]
                uv, val = model.project(Xc)
                inb = val & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                rows = np.where(inb)[0]
                if len(rows) < 6:
                    continue
                pts = uv[rows] + rng.normal(scale=noise_px, size=(len(rows), 2))
                board_obs.append(BoardObs(cam_id=c, frame_id=fr, board_id=b,
                                          corner_ids=rows, pts_2d=pts))
    return specs, board_obs, gt_pts, {0: (W, H), 1: (W, H)}


def test_reconstruct_fuses_all_boards_with_correct_geometry():
    specs, board_obs, gt_pts, img_size = _make_board_obs()
    obj = reconstruct_object(board_obs, specs, img_size)

    assert set(obj.board_ids) == {0, 1, 2}, "all three co-observed boards must fuse"
    # per-point error vs GT: object frame is board 0 in both, so no extra alignment needed
    err = []
    for r, (b, c) in enumerate(obj.pts_obj_2_board):
        err.append(np.linalg.norm(obj.pts_3d[r] - gt_pts[int(b)][int(c)]))
    err = np.array(err)
    assert np.median(err) < 3e-3, f"median geometry error {np.median(err) * 1e3:.2f} mm"
    assert err.max() < 1e-2, f"max geometry error {err.max() * 1e3:.2f} mm"


def test_object_obs_pool_boards_per_image():
    specs, board_obs, _gt, img_size = _make_board_obs()
    obj = reconstruct_object(board_obs, specs, img_size)
    obs = object_obs_from_board_obs(board_obs, obj)
    # one ObjectObs per (camera, frame) — all boards pooled into one pose constraint
    keys = {(o.cam_id, o.frame_id) for o in obs}
    assert len(obs) == len(keys), "boards in one image must pool into a single ObjectObs"
    # rows index the fused cloud, and pooling yields more corners than any single board
    assert all(o.point_rows.max() < len(obj.pts_3d) for o in obs)
    assert max(len(o.point_rows) for o in obs) > specs[0].n_corners


def _make_disjoint_board_obs(noise_px=0.1, seed=0):
    """Two boards that are NEVER co-observed: board ``b`` is only ever seen by camera ``b``
    (board 0 by cam 0, board 1 by cam 1), so no single image contains both -> two separate
    covisibility components -> two objects."""
    specs = [BoardSpec(5, 5, 0.04, 0.03, 0.1) for _ in range(2)]
    bp = {b: board_object_points(specs[b]) for b in range(2)}
    model = RadTanModel(F, F, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    board_obs = []
    for b, c in [(0, 0), (1, 1)]:                     # board_id == cam_id, never together
        for fr in range(20):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            Tcb = _T(axis * rng.uniform(-0.4, 0.4),
                     [rng.uniform(-0.2, 0.2), rng.uniform(-0.15, 0.15), rng.uniform(0.8, 1.2)])
            Xc = (Tcb[:3, :3] @ bp[b].T).T + Tcb[:3, 3]
            uv, val = model.project(Xc)
            inb = val & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            rows = np.where(inb)[0]
            if len(rows) < 6:
                continue
            pts = uv[rows] + rng.normal(scale=noise_px, size=(len(rows), 2))
            board_obs.append(BoardObs(cam_id=c, frame_id=fr, board_id=b,
                                      corner_ids=rows, pts_2d=pts))
    return specs, board_obs, {0: (W, H), 1: (W, H)}


@pytest.mark.req("FR-RIG-017")
def test_reconstruct_objects_fuses_covisible_into_single_object():
    """Three mutually co-observed boards fuse into exactly ONE object via the multi-object
    entry point — parity with the singular ``reconstruct_object`` behavior."""
    specs, board_obs, _gt, img_size = _make_board_obs()
    objects, obs = reconstruct_objects(board_obs, specs, img_size)
    assert len(objects) == 1, "co-observed boards must fuse into a single object"
    assert set(objects[0].board_ids) == {0, 1, 2}
    assert objects[0].object_id == 0
    assert obs and all(o.object_id == 0 for o in obs)
    assert all(o.point_rows.max() < len(objects[0].pts_3d) for o in obs)


@pytest.mark.req("FR-RIG-017")
def test_reconstruct_objects_keeps_disjoint_objects():
    """Two boards never co-observed -> exactly two objects, each with one board, and each
    ObjectObs carries the object_id of the object that contains the board its camera saw."""
    specs, board_obs, img_size = _make_disjoint_board_obs()
    objects, obs = reconstruct_objects(board_obs, specs, img_size)
    assert len(objects) == 2, "boards never seen together must stay as two objects"
    assert sorted(sorted(o.board_ids) for o in objects) == [[0], [1]]
    # board_id == cam_id in this fixture, so cam c's obs must reference board c's object
    board_to_obj = {b: o.object_id for o in objects for b in o.board_ids}
    assert obs
    for ob in obs:
        assert ob.object_id == board_to_obj[ob.cam_id]
        obj = next(o for o in objects if o.object_id == ob.object_id)
        assert ob.point_rows.min() >= 0
        assert ob.point_rows.max() < len(obj.pts_3d), "rows must index this object's pts_3d"


def test_reconstruct_with_model_aware_init_models_narrow_fov():
    """``init_models`` resects each board with the camera's native model (MC-Calib's
    cam_params_path init) instead of the Brown bootstrap. Backward-compatibility check on the
    narrow-FOV (RadTan) rig — the genuinely wide-FOV case (the actual motivation for
    ``init_models``, and for the coplanar bearing-vector homography fix, ADR-0019) is
    :func:`test_reconstruct_with_model_aware_init_models_wide_fov` below."""
    specs, board_obs, gt_pts, img_size = _make_board_obs()
    # the true generating model for every camera (a known prior, like cam_params_path)
    init_models = {0: RadTanModel(F, F, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0),
                   1: RadTanModel(F, F, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0)}
    obj = reconstruct_object(board_obs, specs, img_size, init_models=init_models)
    assert set(obj.board_ids) == {0, 1, 2}
    err = np.array([np.linalg.norm(obj.pts_3d[r] - gt_pts[int(b)][int(c)])
                    for r, (b, c) in enumerate(obj.pts_obj_2_board)])
    assert np.median(err) < 3e-3, f"median geometry error {np.median(err) * 1e3:.2f} mm"
    assert err.max() < 1e-2


def _make_wide_fov_board_obs(seed=0):
    """Two ChArUco boards viewed by a single genuinely wide-FOV Double Sphere camera (xi=0.3,
    alpha=0.6 -- a ~180 deg-class fisheye, not ``DoubleSphereModel.sample()``'s narrower
    default), tilted ~65-77 deg from fronto-parallel and close (0.35-0.55 m) so a large
    fraction of each board's corners are genuinely past 90 deg off-axis -- the exact regime
    the coplanar homography DLT cannot represent (``z > 0`` only) and the bearing-vector
    homography (ADR-0019) fixes. Board resection here is a single-board (coplanar) PnP per
    frame, unlike the non-coplanar fused-object DLT case ADR-0018 already covered.
    """
    W, H = 1280, 960
    model = DoubleSphereModel(fx=280.0, fy=280.0, cx=W / 2, cy=H / 2, xi=0.3, alpha=0.6)
    specs = [BoardSpec(5, 5, 0.04, 0.03, 0.35), BoardSpec(5, 5, 0.04, 0.03, 0.35)]
    bp = {0: board_object_points(specs[0]), 1: board_object_points(specs[1])}
    T_co_b = {0: np.eye(4), 1: _T([0.0, 0.9, 0.0], [1.0, 0.0, 0.3])}
    gt_pts = {}
    for b in range(2):
        Ph = np.c_[bp[b], np.ones(len(bp[b]))]
        gt_pts[b] = (T_co_b[b] @ Ph.T).T[:, :3]           # object frame == board 0's own frame
    # the combined centroid, in the object frame -- used only to steer each frame's camera pose
    # so the tilt pivots near mid-board (better peripheral coverage); gt_pts itself is untouched,
    # so it stays exactly in reconstruct_object's own (uncentered) object-frame convention.
    centroid = np.concatenate([gt_pts[0], gt_pts[1]]).mean(0)

    rng = np.random.default_rng(seed)
    board_obs = []
    for fr in range(40):
        ang_y = rng.uniform(1.05, 1.35)                    # ~60-77 deg tilt about Y
        jitter = rng.normal(scale=0.08, size=3)
        R = so3_exp([jitter[0], ang_y, jitter[2]])
        target = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05),
                           rng.uniform(0.35, 0.55)])
        t = target - R @ centroid                           # centroid -> ~target in camera frame
        for b in range(2):
            Xg = gt_pts[b] @ R.T + t
            uv, val = model.project(Xg)
            inb = val & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            rows = np.where(inb)[0]
            if len(rows) < 6:
                continue
            pts = uv[rows] + rng.normal(scale=0.1, size=(len(rows), 2))
            board_obs.append(BoardObs(cam_id=0, frame_id=fr, board_id=b,
                                      corner_ids=rows, pts_2d=pts))
    return specs, board_obs, gt_pts, {0: (W, H)}, model


def test_reconstruct_with_model_aware_init_models_wide_fov():
    """The genuinely wide-FOV case: a single Double Sphere camera resects two co-observed
    ChArUco boards whose corners are routinely past 90 deg off-axis. Before ADR-0019 the
    coplanar board-resection path (``robust_pose_irls`` -> ``estimate_pose_ransac`` ->
    ``ransac_pnp_normalized``) silently dropped every peripheral corner via its ``z > 0``
    normalized-plane filter; with a board tilted this steeply that can leave too few forward
    corners to resect at all. After the bearing-vector homography fix, every valid corner
    (front or peripheral) is used and the fused geometry is recovered correctly."""
    specs, board_obs, gt_pts, img_size, model = _make_wide_fov_board_obs()
    n_peripheral = 0
    for o in board_obs:
        rays, _ = model.unproject(o.pts_2d)
        n_peripheral += int((rays[:, 2] <= 0).sum())
    assert n_peripheral > 80, "guard: the scene must genuinely exercise peripheral corners"

    obj = reconstruct_object(board_obs, specs, img_size, init_models={0: model})
    assert set(obj.board_ids) == {0, 1}
    err = np.array([np.linalg.norm(obj.pts_3d[r] - gt_pts[int(b)][int(c)])
                    for r, (b, c) in enumerate(obj.pts_obj_2_board)])
    assert np.median(err) < 3e-3, f"median geometry error {np.median(err) * 1e3:.2f} mm"
    assert err.max() < 1e-2, f"max geometry error {err.max() * 1e3:.2f} mm"
