"""Strict analytic-Jacobian contract (the "differentiability" guarantee).

DS-MSP uses hand-derived analytic Jacobians (no autodiff) for speed and stability. This
suite is the standing proof that every model's ``project_jacobian`` — and the SO(3) right
Jacobian behind the manifold re-basing solver — agrees with finite differences to a strict
tolerance. Richardson-extrapolated central differences make the FD truncation error
negligible (empirically ~1e-11 across all models), so a failure here means the *analytic*
derivative is wrong, not the differencing.

This is the 1e-6 gate; the cheap 5e-3 ``allclose`` checks in
``test_camera_model_contract`` remain as the always-on fast smoke tier.
"""

import pytest

from ds_msp.testing import gradcheck_project, gradcheck_retraction

from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.dsplus import DSPlusModel
from ds_msp.models.eucm import EUCMModel
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.ocam import OCamModel
from ds_msp.models.radtan import RadTanModel
from ds_msp.models.ucm import UCMModel
from ds_msp.testing import FakeModel

REL_TOL = 1e-6

_FACTORIES = [
    ("fake_pinhole", FakeModel.sample),
    ("ds", DoubleSphereModel.sample),
    ("ucm", UCMModel.sample),
    ("eucm", EUCMModel.sample),
    ("kb", KannalaBrandtModel.sample),
    ("radtan", RadTanModel.sample),
    ("ocam", OCamModel.sample),
    ("dsplus", DSPlusModel.sample),
]


@pytest.mark.jac
@pytest.mark.parametrize("factory", [f for _, f in _FACTORIES],
                         ids=[n for n, _ in _FACTORIES])
def test_project_jacobian_strict(factory):
    r = gradcheck_project(factory(), rel_tol=REL_TOL)
    assert r["ok"], (f"point_rel_err={r['point_rel_err']:.2e}, "
                     f"param_rel_err={r['param_rel_err']:.2e} (tol {REL_TOL:.0e})")


@pytest.mark.jac
def test_so3_right_jacobian_strict():
    r = gradcheck_retraction(rel_tol=REL_TOL)
    assert r["ok"], f"so3 right-Jacobian rel_err={r['rel_err']:.2e} (tol {REL_TOL:.0e})"

# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-MODEL-003", "NFR-NUM-001", "NFR-REPRO-001")
