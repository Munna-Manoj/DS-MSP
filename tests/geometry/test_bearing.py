"""Regression tests for the shared full-sphere bearing residual."""

import numpy as np
import pytest

from ds_msp.geometry.bearing import chordal_bearing_residual_jacobian


def test_antipodal_ray_is_maximal_not_a_perfect_fit():
    observed = np.array([[0.0, 0.0, 1.0]])
    predicted = -observed

    residual, jacobian, valid = chordal_bearing_residual_jacobian(predicted, observed)

    assert valid.tolist() == [True]
    assert np.linalg.norm(residual[0]) == 2.0
    assert np.isfinite(jacobian).all()


def test_chordal_cost_is_monotone_over_full_angular_domain():
    angles = np.linspace(0.0, np.pi, 181)
    predicted = np.column_stack([np.sin(angles), np.zeros_like(angles), np.cos(angles)])
    observed = np.tile(np.array([0.0, 0.0, 1.0]), (len(angles), 1))

    residual, _jacobian, valid = chordal_bearing_residual_jacobian(predicted, observed)
    cost = np.einsum("ij,ij->i", residual, residual)

    assert valid.all()
    assert np.all(np.diff(cost) >= 0.0)
    assert cost[0] == 0.0
    assert cost[-1] == 4.0


def test_degenerate_and_nonfinite_rows_are_invalid_and_zeroed():
    predicted = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [np.nan, 0.0, 1.0],
        [np.inf, 0.0, 1.0],
    ])
    observed = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])

    residual, jacobian, valid = chordal_bearing_residual_jacobian(predicted, observed)

    assert valid.tolist() == [False, False, False, False]
    assert np.array_equal(residual, np.zeros((4, 3)))
    assert np.array_equal(jacobian, np.zeros((4, 3, 3)))


pytestmark = pytest.mark.req("FR-CALIB-002", "FR-MVG-003")
