"""Numerical guard rails (ds_msp.core.numerics): every failure is an attributable
NumericalError, and rank / conditioning verdicts do not depend on the problem's units."""
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.core.numerics import (
    NumericalError, conditioning, max_or_nan, psd_eigvalsh, require_finite, safe_eigh,
    se3_inverse, spd_inverse, spd_solve,
)

pytestmark = pytest.mark.req("NFR-NUM-009")


def _spd(n, cond, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return (Q * np.geomspace(1.0, 1.0 / cond, n)) @ Q.T


def test_require_finite_names_the_offending_array():
    require_finite("f", a=np.ones(3), b=2.0)
    with pytest.raises(NumericalError, match=r"f: b is not finite"):
        require_finite("f", a=np.ones(3), b=np.array([1.0, np.nan]))
    with pytest.raises(NumericalError):
        require_finite("f", c=np.array([np.inf]))


def test_spd_inverse_and_solve_match_numpy_on_a_well_conditioned_matrix():
    A = _spd(6, 1e4)
    b = np.arange(6.0)
    inv, info = spd_inverse(A)
    np.testing.assert_allclose(inv, np.linalg.inv(A), rtol=1e-9, atol=1e-12)
    x, _ = spd_solve(A, b)
    np.testing.assert_allclose(A @ x, b, rtol=0, atol=1e-9)
    X, _ = spd_solve(A, np.c_[b, 2 * b])
    np.testing.assert_allclose(A @ X, np.c_[b, 2 * b], rtol=0, atol=1e-9)
    assert info.full_rank and info.rank == 6 and 1e3 < info.cond < 1e5


def test_rank_deficiency_raises_instead_of_returning_garbage():
    A = _spd(5, 1e3)
    v = np.ones(5) / np.sqrt(5)
    A = A - (v @ A @ v) * np.outer(v, v)          # exactly rank 4 (up to roundoff)
    assert np.isfinite(np.linalg.inv(A)).all()    # numpy happily "inverts" it
    with pytest.raises(NumericalError, match="not positive definite") as e:
        spd_inverse(A, where="demo", what="H")
    assert e.value.rank == 4 and e.value.size == 5 and e.value.where == "demo"
    with pytest.raises(NumericalError):
        spd_solve(A, np.ones(5))


def test_conditioning_is_unit_invariant():
    """Rescaling the variables (mm vs m, rad vs deg) is a diagonal congruence D A D; the
    equilibrated rank and condition number must not change."""
    A = _spd(4, 1e6, seed=3)
    D = np.diag([1e-3, 1.0, 1e3, 57.3])
    a, b = conditioning(A), conditioning(D @ A @ D)
    assert a.rank == b.rank == 4
    assert b.cond == pytest.approx(a.cond, rel=1e-6)


def test_non_finite_and_non_symmetric_inputs_raise():
    A = _spd(3, 10.0)
    bad = A.copy()
    bad[0, 1] = bad[1, 0] = np.nan
    for fn in (spd_inverse, conditioning, safe_eigh, psd_eigvalsh):
        with pytest.raises(NumericalError):
            fn(bad)
    skew = A.copy()
    skew[0, 1] += 1.0
    with pytest.raises(NumericalError, match="not symmetric"):
        spd_inverse(skew)
    with pytest.raises(NumericalError):
        spd_solve(A, np.array([1.0, np.nan, 0.0]))


def test_psd_eigvalsh_zeroes_roundoff_and_rejects_indefinite():
    P = np.diag([4.0, 1.0, -1e-17])
    np.testing.assert_array_equal(psd_eigvalsh(P), [0.0, 1.0, 4.0])
    with pytest.raises(NumericalError, match="indefinite"):
        psd_eigvalsh(np.diag([4.0, 1.0, -1e-3]))
    np.testing.assert_array_equal(psd_eigvalsh(np.diag([4.0, -1e-9]), rtol=1e-8), [0.0, 4.0])


def test_se3_inverse_is_exact_and_validated():
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.array([0.3, -2.9, 0.4]))
    T[:3, 3] = [1e3, -2.0, 0.5]
    np.testing.assert_allclose(se3_inverse(T) @ T, np.eye(4), atol=1e-12)
    bad = T.copy()
    bad[3, 0] = 1.0
    with pytest.raises(NumericalError, match="bottom row"):
        se3_inverse(bad)
    bad = T.copy()
    bad[:3, :3] *= 1.1
    with pytest.raises(NumericalError, match="orthonormal"):
        se3_inverse(bad)
    bad = T.copy()
    bad[0, 3] = np.nan
    with pytest.raises(NumericalError, match="not finite"):
        se3_inverse(bad)


def test_max_or_nan_does_not_depend_on_order():
    assert max(float("nan"), 1.0) != max(1.0, float("nan"))  # the trap being guarded
    assert np.isnan(max_or_nan([float("nan"), 1.0]))
    assert np.isnan(max_or_nan([1.0, float("nan")]))
    assert max_or_nan([1.0, 3.0, 2.0]) == 3.0
    assert np.isnan(max_or_nan([]))
