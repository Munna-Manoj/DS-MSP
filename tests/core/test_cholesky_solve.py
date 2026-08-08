"""Regression coverage for the NumPy-only triangular Cholesky solve."""

import numpy as np
import pytest

from ds_msp.core.optimize import _cho_solve


def test_compensated_triangular_solve_matches_high_precision_reference():
    """Cancellation-prone Schur systems retain the high-precision solution.

    The fixture has ``cond(L @ L.T) ~= 2.56e10``.  A 100-decimal-place reference exposes
    accumulation loss that an ordinary BLAS dot product hides at easier condition numbers.
    """
    L = np.array([
        [0.5599439982665948, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.08104884785634878, 3.255218670799374, 0.0, 0.0, 0.0, 0.0],
        [0.6106743280680295, 9.00158271759616, 4.037585791489687, 0.0, 0.0, 0.0],
        [0.6761049430928088, -1.2742580357485136, 0.2805887128811007,
         0.06779795111415651, 0.0, 0.0],
        [0.2804562553050535, 0.25444573034295953, -0.04745960236472101,
         10.814603009890645, 0.21354433292846595, 0.0],
        [7.305612346810216, 2.0892625743273507, -0.021345495312479534,
         -0.1495571382856153, 1.038408219195437, 0.09036120376075164],
    ])
    b = np.array([
        5.392092392972249,
        0.11633293681736644,
        0.09566898251852121,
        -8.410121119477353,
        -0.2759065915044369,
        -0.08723920475534044,
    ])
    reference = np.array([
        1374958539.980859,
        -652473261.2826093,
        77786675.26518653,
        -1118252600.8506722,
        6990696.968167926,
        -1426898.3079636656,
    ])

    solved = _cho_solve(L, b)
    relative_inf_error = np.max(np.abs(solved - reference)) / np.max(np.abs(reference))
    assert relative_inf_error < 1e-16


pytestmark = pytest.mark.req("NFR-ARCH-002", "NFR-REPRO-001")
