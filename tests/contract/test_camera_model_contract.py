"""
Model-agnostic contract suite. EVERY camera model must pass this — it is the
guarantee of signature, data-type, and behavioral compatibility that lets the
converter and services treat models interchangeably.

Adding a model = appending its ``.sample`` factory to ``ds_msp.testing``'s
``REFERENCE_MODELS`` (or importing it here). The suite then runs against it
automatically.
"""

import numpy as np
import pytest

from ds_msp.core.contracts import CameraModel
from ds_msp.testing import (
    REFERENCE_MODELS,
    finite_difference_param_jacobian,
    finite_difference_point_jacobian,
    sample_forward_points,
)

from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.ucm import UCMModel
from ds_msp.models.eucm import EUCMModel
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.radtan import RadTanModel
from ds_msp.models.ocam import OCamModel
from ds_msp.models.dsplus import DSPlusModel
from ds_msp.models.eucmplus import EUCMPlusModel

MODEL_FACTORIES = list(REFERENCE_MODELS)
MODEL_FACTORIES.append(("ds", DoubleSphereModel.sample))
MODEL_FACTORIES.append(("ucm", UCMModel.sample))
MODEL_FACTORIES.append(("eucm", EUCMModel.sample))
MODEL_FACTORIES.append(("kb", KannalaBrandtModel.sample))
MODEL_FACTORIES.append(("radtan", RadTanModel.sample))
MODEL_FACTORIES.append(("ocam", OCamModel.sample))
MODEL_FACTORIES.append(("dsplus", DSPlusModel.sample))
MODEL_FACTORIES.append(("eucmplus", EUCMPlusModel.sample))


@pytest.fixture(params=[f for _, f in MODEL_FACTORIES],
                ids=[n for n, _ in MODEL_FACTORIES])
def model(request):
    return request.param()


def test_satisfies_protocol(model):
    assert isinstance(model, CameraModel)


def test_param_names_match_vector(model):
    assert len(model.param_names) == model.params.size
    assert model.params.dtype == np.float64


def test_K_shape(model):
    K = model.K
    assert K.shape == (3, 3)
    assert K[2, 2] == 1.0


def test_project_shapes_and_dtypes(model):
    P = sample_forward_points()
    uv, valid = model.project(P)
    assert uv.shape == (len(P), 2)
    assert uv.dtype == np.float64
    assert valid.shape == (len(P),)
    assert valid.dtype == bool


def test_unproject_shapes_and_unit_norm(model):
    P = sample_forward_points()
    uv, _ = model.project(P)
    rays, valid = model.unproject(uv)
    assert rays.shape == (len(uv), 3)
    norms = np.linalg.norm(rays[valid], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_roundtrip_project_unproject(model):
    P = sample_forward_points()
    uv, v1 = model.project(P)
    rays, v2 = model.unproject(uv)
    ok = v1 & v2
    assert ok.sum() >= 0.9 * len(P)
    d = P[ok] / np.linalg.norm(P[ok], axis=1, keepdims=True)
    cos = np.sum(rays[ok] * d, axis=1)
    assert (cos > 1 - 1e-6).all()


def test_invalid_rows_are_zeroed_not_nan(model):
    P = sample_forward_points()
    uv, _ = model.project(P)
    assert not np.isnan(uv).any()
    rays, _ = model.unproject(uv)
    assert not np.isnan(rays).any()


def test_point_jacobian_matches_finite_difference(model):
    P = sample_forward_points()
    _, J_point, _, _ = model.project_jacobian(P)
    assert J_point.shape == (len(P), 2, 3)
    # Relative tolerance accommodates FD truncation on steep models; analytic
    # Jacobians for the well-behaved models agree far tighter than this ceiling.
    assert np.allclose(J_point, finite_difference_point_jacobian(model, P),
                       rtol=5e-3, atol=1e-4)


def test_param_jacobian_matches_finite_difference(model):
    P = sample_forward_points()
    _, _, J_param, _ = model.project_jacobian(P)
    assert J_param.shape == (len(P), 2, model.params.size)
    assert np.allclose(J_param, finite_difference_param_jacobian(model, P),
                       rtol=5e-3, atol=1e-4)


def test_jacobian_uv_matches_project(model):
    P = sample_forward_points()
    uv_p, _ = model.project(P)
    uv_j, _, _, _ = model.project_jacobian(P)
    assert np.allclose(uv_p, uv_j)


def test_param_vector_roundtrip(model):
    m2 = type(model).from_params(model.params)
    assert np.allclose(m2.params, model.params)


def test_serialization_roundtrip(model):
    m2 = type(model).from_dict(model.to_dict())
    assert np.allclose(m2.params, model.params)


def test_param_bounds_shape(model):
    lb, ub = type(model).param_bounds()
    assert lb.shape == ub.shape == (model.params.size,)
    assert (lb <= ub).all()


def test_batch_shape_preserved(model):
    P = sample_forward_points(n=12).reshape(3, 4, 3)
    uv, valid = model.project(P)
    assert uv.shape == (3, 4, 2)
    assert valid.shape == (3, 4)
