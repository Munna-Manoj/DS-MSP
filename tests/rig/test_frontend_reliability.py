"""Regression checks for uncertainty, structure constraints and pre-export gates."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from ds_msp.core.covariance import pixel_whiteners
from ds_msp.core.lie import se3_exp
from ds_msp.data.observations import BoardObs, ObjectObs, RigState
from ds_msp.io.mccalib import CameraGT, Scenario
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig import bundle, pipeline
from ds_msp.rig.accuracy import accuracy_profile_from_dict
from ds_msp.rig.reconstruct import object_obs_from_board_obs, object_obs_from_board_obs_multi
from ds_msp.rig.merge import remap_object_obs
from ._synth import make_object

pytestmark = pytest.mark.req("FR-CALIB-009", "FR-RIG-025", "FR-RIG-020", "FR-RIG-021", "FR-RIG-024")


def fixture():
    obj = make_object({0: np.eye(4)}, nx=5, ny=5, pitch=0.04)
    cam = RadTanModel(500, 500, 320, 240, 0, 0, 0, 0, 0)
    ext = {0: np.eye(4), 1: se3_exp(np.array([0., 0.02, 0., .2, 0., 0.]))}
    pose = np.eye(4)
    pose[:3, 3] = [-.1, -.1, 1.2]
    obs = []
    for c in ext:
        T = ext[c] @ pose
        uv, _ = cam.project(obj.pts_3d @ T[:3, :3].T + T[:3, 3])
        obs.append(ObjectObs(c, 0, 0, np.arange(len(uv)), uv + [.2, -.1]))
    rig = RigState({0: cam, 1: cam}, ext, 0, {(0, 0): pose}, {0: obj}, {0: (640, 480), 1: (640, 480)})
    return rig, obs


def test_dense_whitened_jacobian_matches_finite_difference():
    rig, obs = fixture()
    obs = [replace(o, pixel_covariances=np.tile([[4., 1.], [1., 2.]], (len(o.pts_2d), 1, 1))) for o in obs]
    state, residual, jac, retract, k = bundle.build_problem(rig, obs, fix_intrinsics=False)
    J = jac(state)
    numeric = np.column_stack([(residual(retract(state, np.eye(k)[i]*1e-6)) -
                                residual(retract(state, -np.eye(k)[i]*1e-6)))/2e-6 for i in range(k)])
    np.testing.assert_allclose(J, numeric, atol=1e-6, rtol=1e-5)
    W = pixel_whiteners(obs[0].pixel_covariances, len(obs[0].pts_2d))
    np.testing.assert_allclose(residual(state)[:2], W[0] @ [-.2, .1], atol=1e-10)


def test_schur_whitening_matches_its_residual_and_numeric_jacobians():
    rig, obs = fixture()
    obs = [replace(o, pixel_covariances=np.tile([[4., 1.], [1., 2.]], (len(o.pts_2d), 1, 1))) for o in obs]
    state, residual, linearize, retract, k, ng = bundle.build_schur_problem(rig, obs, fix_intrinsics=False)
    rs, As, Bs = linearize(state)
    np.testing.assert_allclose(np.concatenate(rs), residual(state), atol=1e-12)
    numeric = np.column_stack([(residual(retract(state, np.eye(k)[i]*1e-6, np.zeros((ng, 6)))) -
                                residual(retract(state, -np.eye(k)[i]*1e-6, np.zeros((ng, 6)))))/2e-6 for i in range(k)])
    np.testing.assert_allclose(np.vstack(As), numeric, atol=1e-6, rtol=1e-5)
    numeric = np.column_stack([(residual(retract(state, np.zeros(k), np.eye(6)[i][None]*1e-6)) -
                                residual(retract(state, np.zeros(k), -np.eye(6)[i][None]*1e-6)))/2e-6 for i in range(6)])
    np.testing.assert_allclose(Bs[0], numeric, atol=1e-6, rtol=1e-5)


def test_covariances_survive_board_fusion_and_object_remap():
    rig, obs = fixture()
    o = obs[0]
    cov = np.tile([[4., 1.], [1., 2.]], (len(o.pts_2d), 1, 1))
    b = BoardObs(0, 0, 0, o.point_rows, o.pts_2d, pixel_covariances=cov)
    for mapped in [object_obs_from_board_obs([b], rig.objects[0]),
                   object_obs_from_board_obs_multi([b], [rig.objects[0]])]:
        np.testing.assert_array_equal(mapped[0].pixel_covariances, cov)
        np.testing.assert_array_equal(remap_object_obs(mapped, {0: (1, 2)})[0].pixel_covariances, cov)


def test_covariance_keypoint_cache_roundtrip(tmp_path):
    from ds_msp.io.mccalib import _load_detections, save_mccalib_detected_keypoints
    from ds_msp.rig.reconstruct import detect_board_obs_keypoints
    rig, obs = fixture()
    cov = np.tile([[4., 1.], [1., 2.]], (len(obs[0].pts_2d), 1, 1))
    obs[0] = replace(obs[0], pixel_covariances=cov)
    path = str(tmp_path / "keypoints.yml")
    save_mccalib_detected_keypoints(obs, rig.objects[0], rig.img_size, path)
    loaded, _ = _load_detections(path, rig.objects[0], include_covariances=True)
    np.testing.assert_array_equal(loaded[0][0][2], cov)
    assert loaded[1][0][2] is None
    boards, _ = detect_board_obs_keypoints(path)
    np.testing.assert_array_equal(boards[0].pixel_covariances, cov)
    assert boards[1].pixel_covariances is None


def test_reference_anchors_are_noncollinear_and_prior_limits_deformation():
    rig, obs = fixture()
    xyz = rig.objects[0].pts_3d
    anchors = sorted(bundle._structure_anchors(xyz, range(len(xyz))))
    assert np.linalg.matrix_rank(xyz[anchors] - xyz[anchors[0]]) == 2
    fitted = bundle.refine_object_structure(rig, obs, prior_sigma=1e-6)
    assert np.max(np.linalg.norm(fitted.objects[0].pts_3d - xyz, axis=1)) < 1e-6
    np.testing.assert_array_equal(fitted.objects[0].pts_3d[anchors], xyz[anchors])
    with pytest.raises(ValueError, match="collinear"):
        bundle._structure_anchors(np.c_[np.arange(5), np.zeros((5, 2))], range(5))


def scenario(rig, obs, truth=False):
    gt = {c: CameraGT(K=rig.cameras[c].K, dist=None, pose=T) for c, T in rig.T_c_g.items()} if truth else {}
    return Scenario(name="gate", object=rig.objects[0], object_obs=obs, cam_ids=[0, 1],
                    img_size=rig.img_size, gt=gt, mccalib={}, mccalib_rms={})


def profile():
    return accuracy_profile_from_dict({"schema":"ds-msp-rig-accuracy-profile", "schema_version":1,
        "name":"test", "length_unit":"m", "roi":{"points":[[0, 0, 1]]},
        "required_pairs":[[0,1]], "limits":{"rotation_error_deg_max":.1, "relative_translation_error_max":.001}})


def test_observability_refusal_does_not_create_or_overwrite_output(tmp_path, monkeypatch):
    rig, obs = fixture()
    monkeypatch.setattr(pipeline, "calibrate_rig", lambda *a, **kw: rig)
    import ds_msp.rig.audit as audit
    monkeypatch.setattr(audit, "audit_rig", lambda *a, **kw: {"n_weak":1, "gauge_ok":True, "findings":[]})
    sentinel = tmp_path / "calibrated_cameras_data.yml"
    sentinel.write_text("previous accepted parameters")
    with pytest.raises(RuntimeError, match="observability audit refused"):
        pipeline.calibrate_scenario(scenario(rig, obs), "radtan", audit_gate="refuse", save_dir=str(tmp_path))
    assert sentinel.read_text() == "previous accepted parameters"
    assert len(list(tmp_path.iterdir())) == 1


def test_accuracy_gate_requires_independent_truth_and_passing_limits(tmp_path, monkeypatch):
    rig, obs = fixture()
    monkeypatch.setattr(pipeline, "calibrate_rig", lambda *a, **kw: rig)
    with pytest.raises(RuntimeError, match="accuracy gate refused"):
        pipeline.calibrate_scenario(scenario(rig, obs), "radtan", audit_gate="off", accuracy_gate="refuse",
                                    accuracy_profile=profile(), save_dir=str(tmp_path))
    assert not list(tmp_path.iterdir())
    scn = scenario(rig, obs, truth=True)
    result = pipeline.calibrate_scenario(scn, "radtan", audit_gate="off", accuracy_gate="refuse",
                                         accuracy_profile=profile(), save_dir=str(tmp_path))
    assert result["acceptance"]["accuracy_status"] == "PASS"
    assert (tmp_path / "calibrated_cameras_data.yml").exists()
    bad = copy.deepcopy(rig)
    bad.T_c_g[1][0, 3] += .01
    monkeypatch.setattr(pipeline, "calibrate_rig", lambda *a, **kw: bad)
    before = (tmp_path / "calibrated_cameras_data.yml").read_bytes()
    with pytest.raises(RuntimeError, match="accuracy gate refused"):
        pipeline.calibrate_scenario(scn, "radtan", audit_gate="off", accuracy_gate="refuse",
                                    accuracy_profile=profile(), save_dir=str(tmp_path))
    assert (tmp_path / "calibrated_cameras_data.yml").read_bytes() == before
