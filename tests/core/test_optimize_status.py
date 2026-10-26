"""Solver termination status (ds_msp.core.optimize): a solve reports CONVERGED,
NO_CONVERGENCE or NUMERICAL_FAILURE, and a non-finite cost, residual or linearisation is
never reported as success."""
import numpy as np
import pytest

from ds_msp.core.optimize import (
    CONVERGED, NO_CONVERGENCE, NUMERICAL_FAILURE, box_active, gnc_tls_schur_solve, gnc_tls_solve,
    lm_solve, schur_lm,
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


def _boxed_line():
    """Fit y = a·x + b with a clipped to [0, 1]; the data want a = 2, so the constrained
    optimum is a = 1 with the gradient still pushing a upward."""
    x = np.linspace(-1.0, 1.0, 20)
    y = 2.0 * x - 1.0
    lo, hi = np.array([0.0, -np.inf]), np.array([1.0, np.inf])
    return (np.array([0.5, 0.0]), lambda s: s[0] * x + s[1] - y,
            lambda s: np.c_[x, np.ones_like(x)], lambda s, d: np.clip(s + d, lo, hi), lo, hi)


def test_box_constrained_optimum_converges_on_the_projected_gradient():
    x0, r, J, ret, lo, hi = _boxed_line()
    raw = lm_solve(x0, r, J, ret, block=1, max_iter=200)
    assert raw.state[0] == 1.0
    assert raw.status == NO_CONVERGENCE                  # raw gradient points out of the box
    kkt = lm_solve(x0, r, J, ret, block=1, max_iter=200,
                   bound_active=lambda s, g: box_active(s, lo, hi, g))
    np.testing.assert_allclose(kkt.state, raw.state, atol=1e-9)  # a, b uncoupled
    assert kkt.status == CONVERGED and kkt.gradient_ratio < 1e-12


def _coupled_box(n_groups=3):
    """y = a·x + b·(x + 0.05) + c_i per group i, with a clipped to [0, 1]. The data want
    a = 2, b = 0. The two slopes are 0.999 correlated, like EUCM alpha against focal, so a
    clipped step keeps a large b compensation for the a move that never happens."""
    x = np.linspace(-1.0, 1.0, 20)
    ys = [2.0 * x + 0.3 * i for i in range(n_groups)]
    lo, hi = np.array([0.0, -np.inf]), np.array([1.0, np.inf])
    feats = np.c_[x, x + 0.05]

    def residual(s):
        ab, c = s[:2], s[2:]
        return np.concatenate([feats @ ab + c[i] - ys[i] for i in range(n_groups)])

    def linearize(s):
        r = residual(s).reshape(n_groups, -1)
        return list(r), [feats] * n_groups, [np.ones((x.size, 1))] * n_groups

    def jacobian(s):
        return np.c_[np.tile(feats, (n_groups, 1)), np.kron(np.eye(n_groups), np.ones((x.size, 1)))]

    def clip(s, d):
        t = s + d
        t[:2] = np.clip(t[:2], lo, hi)
        return t

    # constrained optimum: a = 1, then (b, c) by least squares
    A = jacobian(None)[:, 1:]
    rhs = np.concatenate(ys) - np.tile(x, n_groups)
    opt = np.r_[1.0, np.linalg.lstsq(A, rhs, rcond=None)[0]]
    return np.r_[0.5, 0.0, np.zeros(n_groups)], residual, linearize, jacobian, clip, lo, hi, opt


def _run_coupled(solver, held, x0=None, **kw):
    x0_, r, lin, J, clip, lo, hi, opt = _coupled_box()
    x0 = x0_ if x0 is None else x0
    mask = (lambda s, g: np.r_[box_active(s[:2], lo, hi, g[:2]),
                               np.zeros(len(g) - 2, bool)]) if held else None
    schur_retract = lambda s, ds, dl: clip(s, np.r_[ds, dl[:, 0]])   # noqa: E731
    schur_kw = dict(n_groups=3, shared_dim=2, local_dim=1, block=1)
    if solver == "dense":
        return lm_solve(x0, r, J, clip, block=1, bound_active=mask, **kw), opt
    if solver == "schur":
        return schur_lm(x0, r, lin, schur_retract, bound_active=mask, **schur_kw, **kw), opt
    gkw = {"inner_max_iter": kw.pop("max_iter", 60), **kw}
    if solver == "gnc_dense":
        return gnc_tls_solve(x0, r, J, clip, noise_bound=10.0, block=1, bound_active=mask,
                             **gkw), opt
    return gnc_tls_schur_solve(x0, r, lin, schur_retract, noise_bound=10.0,
                               bound_active=mask, **schur_kw, **gkw), opt


@pytest.mark.parametrize("solver", ["dense", "schur", "gnc_dense", "gnc_schur"])
def test_bound_coordinate_is_held_not_clipped(solver):
    """With ``bound_active`` the step is solved on the free coordinates, so a coupled
    parameter pinned at its bound does not stall the solve. The GNC solvers forward the mask
    to their inner solves (noise bound 10: every residual is an inlier)."""
    held, opt = _run_coupled(solver, True, max_iter=60)
    clipped, _ = _run_coupled(solver, False, max_iter=60)
    np.testing.assert_allclose(held.state, opt, atol=1e-6)
    if solver in ("dense", "schur"):
        assert held.status == CONVERGED and held.iterations < 15
        assert clipped.iterations > 2 * held.iterations
    else:
        assert held.status == CONVERGED
        assert np.abs(clipped.state - opt).max() > 1e3 * np.abs(held.state - opt).max()


def test_schur_and_dense_take_the_same_held_step():
    """Holding a shared coordinate in the Schur complement equals holding it in the full
    system: from a state with the slope at its bound, one step of each solver agrees."""
    x0 = np.r_[1.0, 0.4, 0.1, -0.2, 0.3]                  # a at its upper bound, pushed out
    dense, _ = _run_coupled("dense", True, x0=x0, max_iter=1, lam_init=1e-3)
    schur, _ = _run_coupled("schur", True, x0=x0, max_iter=1, lam_init=1e-3)
    assert dense.state[0] == 1.0 and not np.array_equal(dense.state, x0)
    np.testing.assert_allclose(schur.state, dense.state, rtol=0, atol=1e-12)


def test_box_active_only_exempts_outward_gradients():
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    x = np.array([0.0, 0.0, 1.0])
    g = np.array([1.0, -1.0, -1.0])     # descent -g: leaves, enters, leaves the box
    np.testing.assert_array_equal(box_active(x, lo, hi, g), [True, False, True])
    assert not box_active(np.array([0.5]), lo[:1], hi[:1], np.array([3.0])).any()


def _exp_groups():
    """Three groups y_i = a·exp(b_i x) sharing a, with two outliers in group 1."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 30)
    ys = [2.0 * np.exp(b * x) + 0.01 * rng.standard_normal(30) for b in (0.3, -0.5, 0.8)]
    ys[1][[3, 17]] += 0.5

    def residual(s):
        return np.concatenate([s[0] * np.exp(s[1 + i] * x) - ys[i] for i in range(3)])

    def linearize(s):
        es = [np.exp(s[1 + i] * x) for i in range(3)]
        return ([s[0] * e - y for e, y in zip(es, ys)], [e[:, None] for e in es],
                [(s[0] * x * e)[:, None] for e in es])

    def jacobian(s):
        J = np.zeros((90, 4))
        for i in range(3):
            e = np.exp(s[1 + i] * x)
            J[30 * i:30 * (i + 1), 0] = e
            J[30 * i:30 * (i + 1), 1 + i] = s[0] * x * e
        return J

    return np.array([1.0, 0.0, 0.0, 0.0]), residual, linearize, jacobian


@pytest.mark.req("NFR-NUM-008")
@pytest.mark.parametrize("kernel", ["none", "huber", "cauchy"])
def test_schur_lm_stops_on_relative_tests_without_the_step_test(kernel):
    """With the absolute step test disabled (tol=0), the statistical function test stops the
    solve at the optimum instead of letting it run to maximum damping."""
    x0, r, lin, J = _exp_groups()
    kw = {} if kernel == "none" else {"robust_kernel": kernel, "robust_scale": "auto"}
    res = schur_lm(x0, r, lin, lambda s, ds, dl: s + np.r_[ds, dl[:, 0]], n_groups=3,
                   shared_dim=1, local_dim=1, block=1, tol=0.0, max_iter=500, **kw)
    ref = lm_solve(x0, r, J, lambda s, d: s + d, block=1, tol=0.0, max_iter=500, **kw)
    assert res.stop_reason == "function_tol"
    assert res.status == CONVERGED and res.iterations < 20
    assert ref.iterations > res.iterations
    np.testing.assert_allclose(res.state, ref.state, atol=1e-4)
