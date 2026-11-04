"""OCamCalib model: round-trip + a DS->OCam conversion (OCam represents DS well)."""

import numpy as np
import pytest

from ds_msp.models.ocam import OCamModel
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.adapt import convert
from ds_msp.testing import sample_forward_points


def test_ocam_roundtrip():
    m = OCamModel.sample()
    P = sample_forward_points()
    uv, v1 = m.project(P)
    rays, v2 = m.unproject(uv)
    ok = v1 & v2
    d = P[ok] / np.linalg.norm(P[ok], axis=1, keepdims=True)
    cos = np.sum(rays[ok] * d, axis=1)
    assert (cos > 1 - 1e-5).all()


def test_ds_to_ocam_conversion():
    ds = DoubleSphereModel.sample()
    ocam, report = convert(ds, OCamModel, width=1920, height=1080, n_samples=600)
    assert report["converged"]
    assert report["rms_px"] < 1.0, report


@pytest.mark.req("FR-MODEL-001")
@pytest.mark.parametrize("roll_deg", [1.0, 10.0, 30.0])
def test_affine_reproduces_a_camera_roll_exactly(roll_deg):
    # Why OCam cannot fix rig roll (ADR-0028): the affine [[c, d], [e, 1]] spans every
    # 2x2 matrix up to scale, so A' = A Rz / k with the polynomial rescaled by k
    # reproduces a camera roll about the optical axis exactly.
    theta = np.deg2rad(roll_deg)
    c, s = np.cos(theta), np.sin(theta)
    m = OCamModel(640.0, 360.0, 1.02, 0.01, -0.015, -220.0, 0.0, 6.0, -1.5, 0.25)
    Rz = np.array([[c, -s], [s, c]])
    A = np.array([[m.c, m.d], [m.e, 1.0]]) @ Rz
    k = A[1, 1]
    A = A / k
    a = (m.a0, m.a1, m.a2, m.a3, m.a4)
    rolled = OCamModel(m.cx, m.cy, A[0, 0], A[0, 1], A[1, 0],
                       *(ai * k ** (1 - i) for i, ai in enumerate(a)))
    P = sample_forward_points()
    R = np.eye(3)
    R[:2, :2] = Rz

    uv_rot, ok_rot = m.project(P @ R.T)
    uv_aff, ok_aff = rolled.project(P)

    ok = ok_rot & ok_aff
    assert ok.all()
    assert np.abs(uv_rot - uv_aff).max() < 1e-9


@pytest.mark.req("NFR-NUM-014")
def test_on_axis_point_behind_the_camera_is_invalid():
    m = OCamModel(640.0, 360.0, 1.0, 0.0, 0.0, -220.0, 0.0, 6.0, -1.5, 0.25)
    _uv, valid = m.project(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]))
    assert valid.tolist() == [False, True]


@pytest.mark.req("NFR-NUM-014")
def test_jacobian_on_the_axis_matches_finite_differences():
    m = OCamModel(640.0, 360.0, 1.02, 0.01, -0.015, -220.0, 0.0, 6.0, -1.5, 0.25)
    P = np.array([[0.0, 0.0, 2.0]])
    _uv, J, _Jp, valid = m.project_jacobian(P)
    assert valid[0]
    h = 1e-6
    fd = np.stack([(m.project(P + h * e)[0] - m.project(P - h * e)[0])[0] / (2 * h)
                   for e in np.eye(3)], axis=-1)
    np.testing.assert_allclose(J[0], fd, rtol=1e-5, atol=1e-6)


@pytest.mark.req("NFR-NUM-014")
@pytest.mark.parametrize("params, deg", [
    # OCam cameras fitted on MC-Calib Seq03 and Seq00, at angles where the old Newton start
    # (-a0 / (a1/R + m)) blew up and left an in-view point unconverged.
    ((648.613, 267.6669, 1.0017, 0.0125, -0.0124, -688.3602, -1.9578, 2.3745, -0.5245, 0.0284),
     88.88),
    ((629.0475, 373.6203, 0.9993, -0.0031, 0.0023, -701.5089, 1.9558, 0.5886, 0.4854, -0.0271),
     65.9),
])
def test_rho_solve_converges_wherever_a_root_exists(params, deg):
    m = OCamModel(*params)
    assert m.project(np.array([[np.sin(np.deg2rad(deg)), 0.0, np.cos(np.deg2rad(deg))]]))[1][0]
    a = np.deg2rad(np.linspace(1.0, 110.0, 400))
    P = np.c_[np.sin(a), np.zeros_like(a), np.cos(a)]
    uv, valid = m.project(P)
    # Valid from the axis out to where w(rho) + m·rho has no root, with no holes.
    assert valid[0] and not np.any(np.diff(valid.astype(int)) > 0)
    rays, ok = m.unproject(uv[valid])
    assert ok.all()
    np.testing.assert_allclose(rays, P[valid], atol=1e-6)


# OCam camera 1 fitted on MC-Calib Seq01. Its projection folds at an image radius of 1487 px.
SEQ01_CAM1 = (636.562, 355.355, 0.999983, 0.000694037, -0.000702516,
              -642.44, 2.78595, -1.23641, 0.205288, -0.0117244)


@pytest.mark.req("NFR-NUM-014")
def test_pixels_past_the_fold_do_not_unproject():
    m = OCamModel(*SEQ01_CAM1)
    cx, cy = SEQ01_CAM1[:2]
    rays, ok = m.unproject(np.array([[cx + 1400.0, cy], [cx + 1600.0, cy]]))
    assert ok.tolist() == [True, False]
    uv, valid = m.project(rays[:1])
    assert valid[0]
    np.testing.assert_allclose(uv[0], [cx + 1400.0, cy], atol=1e-6)


@pytest.mark.req("NFR-NUM-014")
def test_every_valid_pixel_projects_back_to_itself():
    """Unproject then project: past the fold the ray would come back at the nearer pixel."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        a = (-rng.uniform(200.0, 900.0), *rng.uniform(-3.0, 3.0, 2), *rng.uniform(-0.3, 0.3, 2))
        m = OCamModel(640.0, 360.0, 1.0, 0.0, 0.0, *a)
        r = rng.uniform(0.0, 3000.0, 300)
        ph = rng.uniform(0.0, 2.0 * np.pi, 300)
        px = np.c_[640.0 + r * np.cos(ph), 360.0 + r * np.sin(ph)]
        rays, ok = m.unproject(px)
        uv, valid = m.project(rays[ok])
        assert valid.all(), f"params {a}"
        np.testing.assert_allclose(uv, px[ok], atol=1e-6, err_msg=f"params {a}")
