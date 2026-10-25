"""Solver termination status (ds_msp.core.optimize): a solve reports CONVERGED,
NO_CONVERGENCE or NUMERICAL_FAILURE, and a non-finite cost, residual or linearisation is
never reported as success."""
import numpy as np
import pytest

from ds_msp.core.optimize import (
    CONVERGED, NO_CONVERGENCE, NUMERICAL_FAILURE, gnc_tls_schur_solve, gnc_tls_solve, lm_solve,
    schur_lm,
)

pytestmark = pytest.mark.req("NFR-NUM-008")


def _line(nan_at_start=False, nan_jacobian=False, nan_off_start=False):
    """Fit y = a·x + b; state is the 2-vector (a, b)."""
    x = np.linspace(-1.0, 1.0, 20)
    y = 2.0 * x - 1.0 + 0.01 * np.sin(7 * x)
    x0 = np.array([0.0, 0.0])

    def residual(s):
        if nan_at_start or (nan_off_start and not np.array_equal(s, x0)):
            return np.full(x.size, np.nan)
        return s[0] * x + s[1] - y

    def jacobian(s):
        J = np.c_[x, np.ones_like(x)]
        if nan_jacobian:
            J[3, 0] = np.nan
        return J

    return x0, residual, jacobian, (lambda s, d: s + d)


def _solve(**kw):
    x0, r, J, ret = _line(**{k: kw.pop(k) for k in list(kw) if k.startswith("nan")})
    return x0, lm_solve(x0, r, J, ret, block=1, **kw)


def test_clean_solve_converges():
    _, res = _solve()
    assert res.status == CONVERGED and res.success and res.converged
    assert res.stop_reason == "step_tol"
    np.testing.assert_allclose(res.state, [2.0, -1.0], atol=0.02)


def test_iteration_budget_is_not_success():
    _, res = _solve(max_iter=1, lam_init=1e3)       # one heavily damped step: not there yet
    assert res.status == NO_CONVERGENCE and not res.success
    assert res.stop_reason == "max_iter"


@pytest.mark.parametrize("fault", ["nan_at_start", "nan_jacobian"])
def test_non_finite_problem_is_a_numerical_failure(fault):
    x0, res = _solve(**{fault: True})
    assert res.status == NUMERICAL_FAILURE and not res.success
    assert res.stop_reason == "non_finite"
    np.testing.assert_array_equal(res.state, x0)    # no step taken from garbage


def test_non_finite_trial_steps_are_rejected_never_accepted():
    x0, res = _solve(nan_off_start=True)
    np.testing.assert_array_equal(res.state, x0)    # every NaN trial rejected
    assert np.isfinite(res.cost)
    assert res.status != CONVERGED or res.stop_reason == "lambda_max"


def test_damping_stall_at_a_stationary_point_is_convergence():
    """Starting at the exact least-squares optimum, no step can lower the cost: the solve
    stalls at lambda_max with a zero gradient, which is convergence."""
    x0, r, J, ret = _line()
    A = J(x0)
    opt = np.linalg.lstsq(A, -r(np.zeros(2)), rcond=None)[0]
    res = lm_solve(opt, r, J, ret, block=1, tol=0.0)
    assert res.stop_reason == "lambda_max"
    assert res.gradient_ratio < 1e-12
    assert res.status == CONVERGED and res.success


def test_damping_stall_with_a_wrong_jacobian_is_not_convergence():
    """A sign-flipped Jacobian makes every step go uphill: the solve stalls with a large
    gradient left, which is not convergence."""
    x0, r, J, ret = _line()
    res = lm_solve(x0, r, lambda s: -J(s), ret, block=1, max_iter=500)
    assert res.stop_reason == "lambda_max"
    assert res.status == NO_CONVERGENCE and not res.success


def _schur_line(nan=False):
    x0, r, J, _ = _line()

    def linearize(s):
        e = r(s)
        A = J(s)[:, :1]
        B = J(s)[:, 1:]
        if nan:
            B = B.copy()
            B[0, 0] = np.nan
        return [e], [A], [B]

    def retract(s, d_shared, d_local):
        return s + np.r_[d_shared, d_local[0]]

    return x0, r, linearize, retract


@pytest.mark.parametrize("nan, expected", [(False, CONVERGED), (True, NUMERICAL_FAILURE)])
def test_schur_lm_status(nan, expected):
    x0, r, lin, ret = _schur_line(nan)
    res = schur_lm(x0, r, lin, ret, n_groups=1, shared_dim=1, local_dim=1, block=1)
    assert res.status == expected and res.success == (expected == CONVERGED)
    if nan:
        np.testing.assert_array_equal(res.state, x0)


def test_gnc_tls_reports_its_inner_and_outer_status():
    x0, r, J, ret = _line()
    ok = gnc_tls_solve(x0, r, J, ret, noise_bound=0.05, block=1)
    assert ok.status == CONVERGED and ok.success and ok.stop_reason in ("binary", "cost_stable")

    x0, r_bad, J, ret = _line(nan_at_start=True)
    bad = gnc_tls_solve(x0, r_bad, J, ret, noise_bound=0.05, block=1)
    assert bad.status == NUMERICAL_FAILURE and not bad.success

    x0, r, lin, retS = _schur_line(nan=True)
    bad_s = gnc_tls_schur_solve(x0, r, lin, retS, noise_bound=0.05, n_groups=1,
                                shared_dim=1, local_dim=1, block=1)
    assert bad_s.status == NUMERICAL_FAILURE and not bad_s.success


def test_gnc_tls_budget_exhaustion_is_not_success():
    """Residuals spread across the TLS band keep the weights fractional, so one graduation
    level cannot binarise them: the outer budget runs out."""
    x = np.linspace(-1.0, 1.0, 40)
    y = 2.0 * x + 0.15 * np.sin(9.0 * x)
    res = gnc_tls_solve(np.zeros(2), lambda s: s[0] * x + s[1] - y,
                        lambda s: np.c_[x, np.ones_like(x)], lambda s, d: s + d,
                        noise_bound=0.1, block=1, max_outer=1)
    assert res.stop_reason == "max_outer"
    assert res.status == NO_CONVERGENCE and not res.success
