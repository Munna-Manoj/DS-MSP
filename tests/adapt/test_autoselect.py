"""Tests for global-optimal conversion and automatic target-model selection.

Anchored on a real-world failure: OpenCV-fisheye (Kannala-Brandt) lenses whose
radial curve inflects (large +k1 with strong negative higher terms) cannot be
held by the Double Sphere / UCM family at *any* parameters. ``convert_best`` must
escalate model capacity until the reprojection RMS clears the tolerance.
"""

import pytest
import numpy as np

from ds_msp.adapt import convert, convert_best
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.eucm import EUCMModel
from ds_msp.models.ucm import UCMModel

W, H = 2592, 1800

# A representative camera from the 8-fisheye rig (MC-Calib output): inflected
# radial profile that the sphere models provably cannot represent.
INFLECTED_KB = KannalaBrandtModel(
    999.98061593971704, 999.17381059829086, 1271.9840773628734, 878.03245259435948,
    0.23995536111100668, -0.031700115665275704, -0.085110530094180792, 0.027028681893684767)


def test_double_sphere_hits_capacity_floor_on_inflected_lens():
    # DS cannot represent this lens; even the global optimum is many pixels.
    _, rep = convert(INFLECTED_KB, DoubleSphereModel, width=W, height=H, n_samples=2000)
    assert rep["rms_px"] > 5.0, rep  # capacity limit, not an optimiser failure


def test_convert_best_meets_tolerance_on_inflected_lens():
    model, rep, results = convert_best(INFLECTED_KB, width=W, height=H, target_rms=3.0)
    assert rep["rms_px"] < 3.0, rep
    assert rep["selected"] is True
    # The escalation must have tried (and rejected) the cheaper sphere models.
    tried = [r[1]["target_model"] for r in results]
    assert tried[0] == "ucm" and rep["target_model"] == "kb", (tried, rep)


def test_convert_best_returns_best_when_tolerance_unreachable():
    # An impossible tolerance forces the "best available" branch, flagged honestly.
    model, rep, results = convert_best(
        INFLECTED_KB, width=W, height=H, target_rms=1e-6,
        candidates=(UCMModel, EUCMModel))
    assert rep["selected"] is True
    assert "not met" in rep["selected_reason"]
    # Best available is the more capable candidate (EUCM), not the first tried.
    assert rep["target_model"] == "eucm"


def test_convert_is_deterministic_across_runs():
    # The multi-start RNG is seeded, so repeated conversions are reproducible.
    m1, _ = convert(INFLECTED_KB, EUCMModel, width=W, height=H, n_samples=800)
    m2, _ = convert(INFLECTED_KB, EUCMModel, width=W, height=H, n_samples=800)
    assert np.allclose(m1.params, m2.params)


def test_multistart_never_worse_than_single_start():
    # Keeping the best of many starts can only match or beat the lone linear seed.
    _, rep_single = convert(INFLECTED_KB, DoubleSphereModel, width=W, height=H,
                            n_samples=1500, n_restarts=0)
    _, rep_multi = convert(INFLECTED_KB, DoubleSphereModel, width=W, height=H,
                           n_samples=1500, n_restarts=6)
    assert rep_multi["rms_px"] <= rep_single["rms_px"] + 1e-6

# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-ADAPT-002")
