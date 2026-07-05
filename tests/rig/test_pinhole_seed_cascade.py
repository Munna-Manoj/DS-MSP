"""Regression test for two real bugs found on 2026-07-02 in the same speed-optimization effort,
both stemming from the same root mistake: gating multi-start dispersion by *model class*
("RadTan/KB/OCam have no documented degenerate basin, so skip it") instead of verifying safety
empirically at realistic scale.

1. **Cascading seed sensitivity.** ``_robust_pinhole`` always calibrates a plain RadTan model
   internally to produce a focal/PP pre-seed that *every* downstream model initializes from,
   including sphere models documented to be chaotically sensitive to exactly this kind of seed
   perturbation. The model-class gate suppressed multi-start there too, and a 16-camera DS rig
   had one camera collapse to the UCM basin (xi~0) instead of converging to a genuine DS fit
   (xi=0.622).

2. **The gate's own premise was wrong at scale.** Even after fixing (1), a 100-camera RadTan-
   only benchmark showed 56/100 cameras with ~2x the reprojection error and 0/100 improved when
   multi-start was skipped for RadTan's *own* fit — not a documented-basin collapse, just an
   ordinary local optimum a single refine can land in on less-constrained (fewer-frame)
   synthetic data. The original "safe" measurement (real 2-camera/99-frame data) didn't
   generalize. The fix was to remove the model-class gate entirely: ``_shape_seeds`` now always
   disperses (``force`` defaults to ``True``) for every model with >4 params, matching the
   original pre-optimization behavior exactly.

This test covers the *mechanism* fast enough for PR-time CI; the end-to-end reproductions (16
and 100 camera rigs) that first surfaced both bugs needed a larger synthetic rig than is
practical to keep in the fast tier, so only the underlying dispersion invariant is asserted
here.
"""
import numpy as np
import pytest

from ds_msp.calib.bundle import _shape_seeds
from ds_msp.models.radtan import RadTanModel

pytestmark = pytest.mark.req("NFR-PERF-001", "NFR-PERF-003")


def test_multistart_always_disperses_by_default_no_model_class_gate():
    """The model-class gate (skip dispersion for "basin-free" models) is gone: ``_shape_seeds``
    must disperse for RadTan too by default, and only skip when a caller explicitly opts out
    with ``force=False`` — never based on which model is being calibrated."""
    base = np.array([500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    default = _shape_seeds(RadTanModel, base, n_restarts=4, seed=0)
    explicit_off = _shape_seeds(RadTanModel, base, n_restarts=4, seed=0, force=False)
    explicit_on = _shape_seeds(RadTanModel, base, n_restarts=4, seed=0, force=True)
    assert len(default) == 5, "the default must disperse for every model, RadTan included"
    assert len(explicit_off) == 1, "force=False is the only way to skip dispersion now"
    assert len(explicit_on) == 5


def test_robust_pinhole_forces_multistart_on_its_internal_seed():
    """``_robust_pinhole`` always calibrates a plain RadTan model internally to produce the
    focal/PP seed every downstream model (including sensitive sphere models) initializes from
    — it must pass ``force_multistart=True`` to ``calibrate()`` explicitly (belt-and-braces:
    true even though it now also matches the default, so the intent survives if the default
    ever changes again)."""
    import inspect

    import ds_msp.rig.calibrate as rc
    src = inspect.getsource(rc._robust_pinhole)
    assert "force_multistart=True" in src, (
        "_robust_pinhole must force full multi-start on its internal RadTan seed refine — "
        "its output feeds every downstream model's initialization (see module docstring)."
    )
