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
