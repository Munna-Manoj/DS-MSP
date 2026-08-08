"""Helpers shared by executable documentation snippets."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import numpy as np


def stable_display_round(values: np.ndarray, decimals: int) -> np.ndarray:
    """Round displayed results deterministically at a backend-sensitive tie.

    First quantize one guard digit, then apply decimal half-away-from-zero rounding.
    This affects presentation only; callers retain the full-precision solver result.
    """
    array = np.asarray(values, float)
    quantum = Decimal(1).scaleb(-decimals)
    rounded = [
        float(Decimal(f"{value:.{decimals + 1}f}").quantize(quantum, rounding=ROUND_HALF_UP))
        for value in array.ravel()
    ]
    return np.asarray(rounded, float).reshape(array.shape)


def zero_roundoff(value: float, *, atol: float) -> float:
    """Display numerical noise below a declared absolute floor as exact zero."""
    return 0.0 if abs(value) < atol else float(value)
