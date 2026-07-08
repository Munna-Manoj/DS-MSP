"""Object-level fusion (``merge3DObjects``): two rigid objects that never share a board
image but are seen concurrently by different cameras are merged into one Object3D.

The scenario is deterministic and noise-free, so it validates the *algebra* — the pose
composition directions, in particular the inverse-per-edge convention shared with
``object3d.build_objects`` (McCalib.cpp:929). A single flipped inverse blows up the fused
point error, so the tolerances here are tight (< 1e-9).
"""
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.rig.merge import (
    inter_object_transforms,
    merge_objects,
    remap_object_obs,
)
from ds_msp.rig.types import Object3D, ObjectObs

pytestmark = pytest.mark.req("FR-RIG-017")


def _T(rvec, t):
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.asarray(rvec, float))
    T[:3, 3] = t
    return T


def _grid(nx, ny, board_id, spacing=0.05):
    """A planar (z=0) corner grid as an (nx*ny, 3) point cloud."""
    pts, rows = [], []
    k = 0
    for j in range(ny):
        for i in range(nx):
            pts.append([i * spacing, j * spacing, 0.0])
            rows.append((board_id, k))
            k += 1
    return np.array(pts, float), rows


def _single_board_object(object_id, board_id, nx, ny):
    pts, rows = _grid(nx, ny, board_id)
    return Object3D(
        object_id=object_id, board_ids=[board_id], ref_board_id=board_id,
        T_co_b={board_id: np.eye(4)}, pts_3d=pts,
        pts_obj_2_board=np.array(rows, int),
        pts_board_2_obj={(b, c): r for r, (b, c) in enumerate(rows)},
    )


def _object_traj(n, seed=0):
    """A diverse rigid-body motion G(f) = object0's pose in the group frame."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        out.append(_T(axis * rng.uniform(0.2, 1.0),
                      [rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3),
                       rng.uniform(1.5, 2.5)]))
    return out


def _build_scene(n_frames=12):
    obj0 = _single_board_object(object_id=0, board_id=0, nx=3, ny=2)
    obj1 = _single_board_object(object_id=1, board_id=1, nx=2, ny=2)

    # Known rigid link: obj1's frame expressed in obj0's frame (obj1 -> obj0).
    T_o0_o1 = _T([0.2, -0.3, 0.5], [0.4, 0.1, -0.2])

    # Known per-camera extrinsics (group-ref -> camera). Non-trivial so the inv(extr)
    # lift is actually exercised. cam A sees only obj0, cam B sees only obj1.
    extr = {0: _T([0.1, 0.2, -0.05], [0.3, -0.1, 0.5]),
            1: _T([0.0, 0.9, 0.1], [0.6, 0.2, -0.3])}

    traj = _object_traj(n_frames, seed=3)
    n0 = len(obj0.pts_3d)
    n1 = len(obj1.pts_3d)
    obs = []
    for f, G in enumerate(traj):
        T_g_o0 = G
        T_g_o1 = G @ T_o0_o1
        # object->camera = (group->camera) @ (object->group)
        obs.append(ObjectObs(cam_id=0, frame_id=f, object_id=0,
                             point_rows=np.arange(n0), pts_2d=np.zeros((n0, 2)),
                             T_c_o=extr[0] @ T_g_o0))
        obs.append(ObjectObs(cam_id=1, frame_id=f, object_id=1,
                             point_rows=np.arange(n1), pts_2d=np.zeros((n1, 2)),
                             T_c_o=extr[1] @ T_g_o1))
    return [obj0, obj1], obs, extr, T_o0_o1, n0, n1, n_frames


def test_inter_object_transforms_recovers_link():
    from ds_msp.rig.averaging import average_transform

    _, obs, extr, T_o0_o1, _, _, n_frames = _build_scene()
    samples, counts = inter_object_transforms(obs, extr)

    # Ordered pairs both present each frame => samples for (0,1) and (1,0).
    assert set(samples) == {(0, 1), (1, 0)}
    assert counts[(0, 1)] == n_frames               # only the canonical oi<oj is counted
    assert len(counts) == 1

    # T_pair(0,1) = inv(T_g_o1) @ T_g_o0 = obj0 in obj1's frame = inv(T_o0_o1).
    est = average_transform(samples[(0, 1)])
    assert np.allclose(est, np.linalg.inv(T_o0_o1), atol=1e-9)
    # All frames give the identical rigid link (noise-free): zero spread.
    for s in samples[(0, 1)]:
        assert np.allclose(s, np.linalg.inv(T_o0_o1), atol=1e-9)


def test_merge_objects_fuses_into_one_rigid_cloud():
    objs, obs, extr, T_o0_o1, n0, n1, _ = _build_scene()
    obj0, obj1 = objs
    merged_objects, remap = merge_objects(objs, obs, extr)

    # Exactly one merged object (both are one connected component).
    assert len(merged_objects) == 1
    m = merged_objects[0]
    assert m.object_id == 0
    assert m.board_ids == [0, 1]                      # union of board ids
    assert m.ref_board_id == 0                        # ref = min object_id's ref board
    assert len(m.pts_3d) == n0 + n1

    # obj0 is the reference => its points are unchanged (T_mo_o[0] = I).
    assert np.allclose(m.pts_3d[:n0], obj0.pts_3d, atol=1e-9)

    # obj1's points land at T_o0_o1 @ p (obj1 frame -> merged/obj0 frame).
    P1_h = np.c_[obj1.pts_3d, np.ones(n1)]
    expected = (T_o0_o1 @ P1_h.T).T[:, :3]
    assert np.max(np.abs(m.pts_3d[n0:] - expected)) < 1e-9

    # Merged board poses: board 1 -> merged object equals T_o0_o1 (obj1's board was I).
    assert np.allclose(m.T_co_b[0], np.eye(4), atol=1e-9)
    assert np.allclose(m.T_co_b[1], T_o0_o1, atol=1e-9)

    # remap: old rows relocate by the recorded offset.
    assert remap[0] == (0, 0)
    assert remap[1] == (0, n0)
    # A concrete corner: obj1 row 2 -> merged row n0+2, at the expected 3D location.
    assert m.pts_board_2_obj[(1, 2)] == n0 + 2
    assert np.allclose(m.pts_3d[n0 + 2], expected[2], atol=1e-9)


def test_remap_object_obs_relabels_and_shifts():
    objs, obs, extr, _, n0, n1, _ = _build_scene()
    merged_objects, remap = merge_objects(objs, obs, extr)
    m = merged_objects[0]

    remapped = remap_object_obs(obs, remap)
    assert len(remapped) == len(obs)
    for o in remapped:
        assert o.object_id == 0
        assert o.point_rows.min() >= 0
        assert o.point_rows.max() < len(m.pts_3d)

    # obj1 observations get their rows shifted by n0; obj0 observations unchanged.
    obj1_obs = [ro for ro, o in zip(remapped, obs) if o.object_id == 1]
    assert all(np.array_equal(ro.point_rows, np.arange(n1) + n0) for ro in obj1_obs)
    obj0_obs = [ro for ro, o in zip(remapped, obs) if o.object_id == 0]
    assert all(np.array_equal(ro.point_rows, np.arange(n0)) for ro in obj0_obs)
