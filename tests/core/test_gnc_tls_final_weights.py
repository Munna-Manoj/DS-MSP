"""Returned GNC-TLS labels must describe the returned state, not its predecessor."""

import numpy as np
import pytest

from ds_msp.core.optimize import OptResult, _gnc_tls_graduate
from ds_msp.core.robust import gnc_tls_mu_init, gnc_tls_weight


def test_gnc_tls_recomputes_weights_after_final_variable_update():
    states = iter([np.array([0.0, 4.0]), np.array([4.0, 0.0])])

    def residual(state):
        return np.asarray(state, float)

    def inner_solve(_state, _weights):
        state = next(states)
        return OptResult(
            state=state,
            cost=0.0,
            rms=0.0,
            iterations=1,
            success=True,
            converged=True,
            final_lambda=1e-3,
        )

    out = _gnc_tls_graduate(
        np.zeros(2), residual, inner_solve,
        noise_bound=1.0, block=1, max_outer=1, continuation=1.4, weights=None,
    )

    initial_sq = np.array([0.0, 16.0])
    mu = gnc_tls_mu_init(initial_sq, barc2=1.0)
    final_sq = residual(out.state) ** 2
    expected = gnc_tls_weight(final_sq, barc2=1.0, mu=mu)

    np.testing.assert_array_equal(out.weights, expected)
    assert out.weights[0] < out.weights[1]


pytestmark = pytest.mark.req("FR-CORE-001")
