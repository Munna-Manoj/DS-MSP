"""Front-end intrinsic pre-calibration is embarrassingly parallel across cameras — each
camera's fit and pose seeding is independent of every other camera. Verifies NFR-PERF-003:
the ``n_jobs`` parallel path is pure parallelism (same seeds, same solver, same deterministic
result), not an approximation, and speeds up every camera model equally rather than only the
basin-free ones NFR-PERF-001/002 target.
"""
from unittest.mock import patch

import numpy as np
import pytest

from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig.calibrate import make_bundle_front_end

from ._synth import make_rig

pytestmark = pytest.mark.req("NFR-PERF-003")

W, H = 640, 480


def _rig(model_name, n_cam=4, n_frame=8, seed=0):
    facs = {
        "radtan": lambda cam_id, rng: RadTanModel(
            700.0 * rng.uniform(0.98, 1.02), 700.0, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0),
        "ds": lambda cam_id, rng: DoubleSphereModel(300, 300, W / 2, H / 2, 0.1, 0.6),
    }
    obj, obs, img, gt_ext, gtm = make_rig(n_cam=n_cam, n_frame=n_frame, noise_px=0.2,
                                          seed=seed, w=W, h=H, model_factory=facs[model_name])
    obs_by_cam = {}
    for o in obs:
        obs_by_cam.setdefault(o.cam_id, []).append(o)
    return obj, obs_by_cam, img


@pytest.mark.parametrize("model_name", ["radtan", "ds"])
def test_parallel_matches_serial_intrinsics_exactly(model_name):
    """Parallel execution is pure concurrency, not an approximation: the calibrated
    intrinsics must match the serial path to numerical precision, for both a basin-free
    model (RadTan) and a genuine wide-FOV sphere model (DS)."""
    obj, obs_by_cam, img = _rig(model_name)
    cams_serial = make_bundle_front_end(model_name, n_jobs=1)(obj, obs_by_cam, img)
    cams_parallel = make_bundle_front_end(model_name, n_jobs=-1)(obj, obs_by_cam, img)
    for c in cams_serial:
        np.testing.assert_allclose(cams_serial[c].params, cams_parallel[c].params,
                                   rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("model_name", ["radtan", "ds"])
def test_parallel_pose_seeding_matches_serial_and_is_fully_populated(model_name):
    """The gated-PnP pose stage runs in the same pool; every observation's ``T_c_o`` must be
    set (parallel results correctly propagated back onto the *original* ObjectObs instances,
    not lost across the process boundary) and match the serial pose exactly."""
    obj, obs_by_cam, img = _rig(model_name)
    make_bundle_front_end(model_name, n_jobs=1)(obj, obs_by_cam, img)
    poses_serial = {(o.cam_id, o.frame_id): o.T_c_o for cam in obs_by_cam.values() for o in cam}

    obj2, obs_by_cam2, img2 = _rig(model_name)
    make_bundle_front_end(model_name, n_jobs=-1)(obj2, obs_by_cam2, img2)
    for cam in obs_by_cam2.values():
        for o in cam:
            assert o.T_c_o is not None
            key = (o.cam_id, o.frame_id)
            np.testing.assert_allclose(o.T_c_o, poses_serial[key], rtol=1e-8, atol=1e-8)


def test_n_jobs_one_never_spawns_a_process_pool():
    """``n_jobs=1`` is the explicit serial escape hatch (debugging, tiny rigs where pool
    start-up would only add overhead) — it must not touch ProcessPoolExecutor at all."""
    obj, obs_by_cam, img = _rig("radtan", n_cam=2)
    with patch("ds_msp.rig.calibrate.ProcessPoolExecutor") as mock_pool:
        make_bundle_front_end("radtan", n_jobs=1)(obj, obs_by_cam, img)
    mock_pool.assert_not_called()


def test_more_cameras_than_workers_still_calibrates_every_camera():
    """Camera count exceeding the worker cap must still queue and process every camera
    (the realistic 100s-of-cameras case: a handful of workers, many more tasks)."""
    obj, obs_by_cam, img = _rig("radtan", n_cam=6)
    cams = make_bundle_front_end("radtan", n_jobs=2)(obj, obs_by_cam, img)
    assert set(cams) == set(obs_by_cam)
    assert all(cams[c] is not None for c in cams)
