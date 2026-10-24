"""Schur-structured frame-clustered sandwich covariance (FR-RIG-027): it must equal the dense
estimator it replaces at scale, and its error bars must be honest against truth."""
import copy

import numpy as np
import pytest

from ds_msp.rig import bundle
from ds_msp.rig.calibrate import calibrate_rig
from ds_msp.rig.pipeline import make_fixed_intrinsic_front_end

from ._synth import make_rig

pytestmark = pytest.mark.req("FR-RIG-027")


def _calibrate(obs, obj, img_size, models, fix_intrinsics=True):
    return calibrate_rig(obj, copy.deepcopy(obs), img_size, fix_intrinsics=fix_intrinsics,
                         front_end=make_fixed_intrinsic_front_end(models), structure_mode="off")


def _dense_reference(rig, obs, fix_intrinsics):
    """The dense estimator, but with a column-scaled, jitter-free inverse: the stock one's trace-relative jitter perturbs ill-conditioned intrinsic
    directions by ~0.5%, which would hide a real formula error below tolerance."""
    import ds_msp.core.covariance as cov

    orig = cov._sym_inv

    def scaled_inv(A, _jitter):
        A = 0.5 * (A + A.T)
        d = 1.0 / np.sqrt(np.diag(A))
        return d[:, None] * np.linalg.inv(d[:, None] * A * d[None, :]) * d[None, :]
    cov._sym_inv = scaled_inv
    try:
        return bundle._dense_parameter_covariance(
            rig, obs, fix_intrinsics=fix_intrinsics, fix_extrinsics=False, kernel="cauchy",
            scale=None, small_cluster_correction=True)
    finally:
        cov._sym_inv = orig


@pytest.mark.parametrize("fix_intrinsics", [True, False])
def test_schur_covariance_equals_dense_clustered_sandwich(fix_intrinsics):
    obj, obs, img_size, _ext, models = make_rig(n_cam=3, n_frame=10, noise_px=0.4, seed=3,
                                                multi_board=True)
    rig = _calibrate(obs, obj, img_size, models, fix_intrinsics)
    dense = _dense_reference(rig, obs, fix_intrinsics)
    unc = bundle.shared_covariance(rig, obs, fix_intrinsics=fix_intrinsics,
                                   scale=dense["scale"])
    std = np.sqrt(np.diag(unc["cov"]))
    for c, entry in dense["clustered"].items():
        if entry["extrinsic_std"] is not None:
            col = unc["cam_col"][c]
            np.testing.assert_allclose(std[col:col + 6], entry["extrinsic_std"], rtol=1e-6)
        if entry["intrinsic_std"] is not None:
            col = unc["intr_col"][c]
            n = len(entry["intrinsic_std"])
            np.testing.assert_allclose(std[col:col + n], entry["intrinsic_std"], rtol=1e-6)


def test_pair_uncertainty_is_honest_against_truth():
    """Across noisy synthetic rigs, each relative-pose error component divided by its reported
    sigma must be ~unit (z-scores): not >>1 (overconfident), not <<1 (useless bars). Per-axis
    z, not the 6-D Mahalanobis distance, because inverting a covariance estimated from few
    frame clusters inflates the latter by df/(df-p-1) even when every sigma is right."""
    z = []
    for seed in range(12):
        obj, obs, img_size, gt_ext, models = make_rig(n_cam=2, n_frame=24, noise_px=0.5,
                                                      seed=100 + seed)
        rig = _calibrate(obs, obj, img_size, models)
        pair = bundle.pair_uncertainty(rig, bundle.shared_covariance(rig, obs))[(0, 1)]
        T = rig.T_c_g[1] @ np.linalg.inv(rig.T_c_g[0])
        G = gt_ext[1] @ np.linalg.inv(gt_ext[0])
        w = G[:3, :3].T @ T[:3, :3]
        dw = 0.5 * np.array([w[2, 1] - w[1, 2], w[0, 2] - w[2, 0], w[1, 0] - w[0, 1]])
        dC = (-T[:3, :3].T @ T[:3, 3]) - (-G[:3, :3].T @ G[:3, 3])
        e = np.r_[dw, dC]
        z.extend(e / np.sqrt(np.diag(np.asarray(pair["covariance"]))))
    z = np.asarray(z)
    assert 0.6 < np.sqrt(np.mean(z ** 2)) < 1.5, np.sqrt(np.mean(z ** 2))
    assert np.mean(np.abs(z) < 3.0) > 0.95


def test_rigid_board_poses_are_marginalised():
    obj, obs, img_size, _ext, models = make_rig(n_cam=2, n_frame=12, noise_px=0.3, seed=5)
    rig = _calibrate(obs, obj, img_size, models)
    fixed = bundle.shared_covariance(rig, obs)
    boards = bundle.shared_covariance(rig, obs, refine_boards=True)
    col = fixed["cam_col"][1]
    s_fixed = np.sqrt(np.diag(fixed["cov"]))[col:col + 6]
    s_boards = np.sqrt(np.diag(boards["cov"]))[col:col + 6]
    assert np.all(s_boards >= s_fixed * 0.999)          # nuisance parameters never shrink sigma


def test_bread_curvature_clamp_is_psd_and_exact_for_inliers():
    """Each block's curvature along its residual, ``ω + 2ω′s``, must stay >= 0; residuals
    inside the kernel scale keep the exact Triggs term."""
    from ds_msp.core.covariance import psd_curvature_derivative, robust_weight_derivative
    from ds_msp.core.robust import robust_weight
    s = np.logspace(-4, 3, 200)
    for kernel in ("cauchy", "geman_mcclure", "huber", "pseudo_huber"):
        om = robust_weight(s, kernel, 1.0)
        exact = robust_weight_derivative(s, kernel, 1.0)
        clamped = psd_curvature_derivative(s, om, exact)
        assert np.all(om + 2.0 * clamped * s >= -1e-15)
        keep = om + 2.0 * exact * s >= 0.0
        np.testing.assert_allclose(clamped[keep], exact[keep], rtol=1e-12, atol=0.0)


def _outlier_rig(seed=7, n_bad=6):
    """A calibrated rig whose observations then gain gross outliers on whole boards -- the
    frames whose robust bread is indefinite without the curvature clamp."""
    obj, obs, img_size, _ext, models = make_rig(n_cam=2, n_frame=16, noise_px=0.3, seed=seed)
    rig = _calibrate(obs, obj, img_size, models)
    rng = np.random.default_rng(seed)
    bad = copy.deepcopy(obs)
    for o in bad[:n_bad]:
        o.pts_2d = o.pts_2d + rng.normal(0.0, 40.0, o.pts_2d.shape)
    return rig, bad


def test_gross_outliers_keep_the_covariance_finite(monkeypatch):
    rig, bad = _outlier_rig()
    unc = bundle.shared_covariance(rig, bad)
    assert unc["identifiable"] and np.isfinite(unc["cov"]).all()
    pairs = bundle.pair_uncertainty(rig, unc)
    assert all(p["identifiable"] and np.isfinite(p["rotation_sigma_deg"]) for p in pairs.values())
    # the trigger: without the clamp the same frames give indefinite pose blocks
    import ds_msp.rig.bundle as rb
    monkeypatch.setattr(rb, "psd_curvature_derivative", lambda s, om, omp: omp)
    seen = []
    real_inv = np.linalg.inv

    def spy(M):
        if M.shape == (6, 6):
            seen.append(np.linalg.eigvalsh(M)[0])
        return real_inv(M)
    monkeypatch.setattr(rb.np.linalg, "inv", spy)
    rb.shared_covariance(rig, bad)
    assert min(seen) < 0.0


def test_singular_schur_complement_fails_closed():
    S = np.diag([4.0, 1.0, 0.0])
    cov, ok, _ = bundle._sandwich_from_schur(S, np.eye(3))
    assert not ok and np.isinf(cov).all()
    S = np.array([[1.0, 2.0], [2.0, 1.0]])                 # indefinite
    cov, ok, _ = bundle._sandwich_from_schur(S, np.eye(2))
    assert not ok
    cov, ok, _ = bundle._sandwich_from_schur(np.diag([1.0, np.nan]), np.eye(2))
    assert not ok


def _absorbed_case(seed=0):
    """Shared block of 3 camera + 4 structure columns where the frame poses absorbed two
    structure directions -- one single column, one combination of two columns (two boards that
    only ever move together relative to the reference board) -- leaving rounding-level noise."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(40, 7)) * np.r_[1.0, 30.0, 0.2, 5.0, 1.0, 2.0, 2.0]
    G = A.T @ A
    Q = np.eye(7)
    Q[:, 3] = 0.0
    n = np.zeros(7)
    n[5], n[6] = 1.0, -1.0
    Q -= np.outer(n, n) / 2.0
    S = Q @ G @ Q
    S[3, 3] = -1e-11 * G[3, 3]                             # rounding left after elimination
    R = rng.normal(size=(7, 7))
    M = Q @ R @ R.T @ Q
    return S, M, G


def test_absorbed_structure_is_unbounded_and_spares_the_cameras():
    S, M, G = _absorbed_case()
    cov, ok, absorbed = bundle._sandwich_from_schur(S, M, G, [([3], 1), ([5, 6], 1)])
    assert ok and absorbed == [3, 5, 6]
    assert np.isinf(cov[3]).all() and np.isinf(cov[:, 5]).all()
    # the camera block equals the sandwich on the identified reparameterisation
    keep = [0, 1, 2, 4]
    B = np.zeros((7, 5))
    B[keep, range(4)] = 1.0
    B[[5, 6], 4] = 1.0 / np.sqrt(2.0)
    Si = np.linalg.inv(B.T @ S @ B)
    ref = B @ Si @ (B.T @ M @ B) @ Si @ B.T
    np.testing.assert_allclose(cov[:3, :3], ref[:3, :3], rtol=1e-8)
    # without the gauges the same matrix is singular and fails closed
    cov, ok, _ = bundle._sandwich_from_schur(S, M, G)
    assert not ok and np.isinf(cov).all()
    # a listed gauge that is not null contradicts the board graph: fail closed
    cov, ok, _ = bundle._sandwich_from_schur(S, M, G, [([0], 1)])
    assert not ok and np.isinf(cov).all()


def test_ill_conditioned_but_identified_stays_bounded():
    """Near-redundant intrinsics (EUCM alpha/beta against focal) leave scaled eigenvalues near
    1e-10 that are still information, not rounding; they must keep a finite sigma."""
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    S = Q @ np.diag([1.0, 0.5, 0.2, 1e-3, 1e-7, 2e-10]) @ Q.T
    cov, ok, absorbed = bundle._sandwich_from_schur(S, np.eye(6), S)
    assert ok and absorbed == [] and np.isfinite(cov).all()
    np.testing.assert_allclose(cov, np.linalg.inv(S) @ np.linalg.inv(S), rtol=1e-4)


def test_board_never_seen_with_another_is_absorbed_not_fatal():
    """Board 2 shares no frame with boards 0/1, so the frame poses absorb its in-object pose:
    its columns are unbounded, board 1 and the cameras keep a finite sigma close to the one
    with the boards held fixed (the sandwich's meat changes too, so it need not be larger)."""
    obj, obs, img_size, _ext, models = make_rig(n_cam=2, n_frame=24, noise_px=0.3, seed=11)
    board = np.asarray(obj.pts_obj_2_board, int)[:, 0]
    split = []
    for o in obs:
        on2 = board[o.point_rows] == 2
        sel = on2 if o.frame_id % 2 else ~on2
        if sel.sum() >= 6:
            o = copy.copy(o)
            o.point_rows, o.pts_2d = o.point_rows[sel], o.pts_2d[sel]
            split.append(o)
    rig = _calibrate(split, obj, img_size, models)
    unc = bundle.shared_covariance(rig, split, refine_boards=True)
    bases, board_col, _bend, _end = bundle._board_layout(
        rig, 6, refine_boards=True, bend_boards=False)
    c1, c2 = board_col[(0, 1)], board_col[(0, 2)]
    assert unc["identifiable"] and unc["unidentified_structure"] == list(range(c2, c2 + 6))
    std = np.sqrt(np.diag(unc["cov"]))
    assert np.isfinite(std[c1:c1 + 6]).all()
    col = unc["cam_col"][1]
    s_fixed = np.sqrt(np.diag(bundle.shared_covariance(rig, split)["cov"]))[col:col + 6]
    ratio = std[col:col + 6] / s_fixed
    assert np.all((ratio > 0.8) & (ratio < 1.5)), ratio


def test_unidentifiable_extrinsics_refuse_and_write_strict_json(monkeypatch):
    """An unbounded sigma never passes a limit (NaN/inf compare False), is reported as not
    identifiable with a warning, and the acceptance record stays strict JSON."""
    import json

    from ds_msp.rig import pipeline
    obj, obs, img_size, _ext, models = make_rig(n_cam=2, n_frame=12, noise_px=0.3, seed=5)
    rig = _calibrate(obs, obj, img_size, models)
    real = bundle.shared_covariance

    def unidentified(*a, **k):
        out = real(*a, **k)
        out["cov"] = np.full_like(out["cov"], np.inf)
        out["identifiable"] = False
        return out
    monkeypatch.setattr(pipeline.bundle, "shared_covariance", unidentified)
    unc = pipeline.extrinsic_uncertainty(rig, obs, fix_intrinsics=True,
                                         max_rotation_sigma_deg=1.0, max_centre_sigma=1.0)
    assert unc["identifiable"] is False and unc["within_limits"] is False
    assert unc["worst"]["rotation_sigma_deg"] is None
    json.dumps(unc, allow_nan=False)
