"""Convert robustness contract (FR-ADAPT-001, NFR-NUM-007).

``convert()`` must reach the global optimum deterministically, independent of the random
restart lottery. Two properties pin this:

* **Self-conversion is exact** — converting a model into its own class reproduces it to
  machine precision. This is the sharpest regression guard: it originally caught a case where
  a source shape parameter far from its linear seed settled in a wrong basin and self-converted
  at several pixels. The deterministic shape sweep fixes it.
* **KB is a faithful universal target** — every parametric fisheye model converts into
  Kannala-Brandt sub-pixel over its forward FOV (KB being the most expressive model here).

Synthetic and fast (no images), so it runs on every adapt change.
"""

import pytest

from ds_msp.adapt import convert
from ds_msp.adapt.evaluate import reprojection_report
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.eucm import EUCMModel
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.ucm import UCMModel

W, H = 1280, 960
_F = 320.0  # ~150 deg fisheye-scale focal for this resolution

# Realistic calibrated instances, incl. a DS with xi far from its neutral seed.
_INSTANCES = [
    ("ucm", UCMModel(_F, _F, W / 2, H / 2, 0.62)),
    ("eucm", EUCMModel(_F, _F, W / 2, H / 2, 0.60, 1.40)),
    ("ds", DoubleSphereModel(_F, _F, W / 2, H / 2, -0.18, 0.59)),
    ("kb", KannalaBrandtModel(_F, _F, W / 2, H / 2, 0.02, -0.004, 0.001, -0.0003)),
]

pytestmark = pytest.mark.req("FR-ADAPT-001", "NFR-NUM-007")


@pytest.mark.parametrize("name,model", _INSTANCES, ids=[n for n, _ in _INSTANCES])
def test_self_conversion_is_exact(name, model):
    """Converting a model into its own class reproduces it to ~machine precision."""
    target, _ = convert(model, type(model), width=W, height=H)
    rep = reprojection_report(model, target, W, H)
    assert rep["rms_px"] < 1e-3, (name, rep["rms_px"])


@pytest.mark.parametrize("name,model", _INSTANCES, ids=[n for n, _ in _INSTANCES])
def test_kb_is_faithful_universal_target(name, model):
    """Every parametric fisheye model converts into KB sub-pixel over its forward FOV."""
    target, _ = convert(model, KannalaBrandtModel, width=W, height=H)
    rep = reprojection_report(model, target, W, H)
    assert rep["rms_px"] < 0.5, (name, rep["rms_px"])


def test_ds_self_convert_is_deterministic_across_seeds():
    """A DS self-convert with xi far from the linear seed is exact regardless of seed (the
    deterministic sweep removes the restart lottery)."""
    src = DoubleSphereModel(_F, _F, W / 2, H / 2, -0.18, 0.59)
    for seed in (0, 1, 2, 3):
        target, _ = convert(src, DoubleSphereModel, width=W, height=H, seed=seed)
        rep = reprojection_report(src, target, W, H)
        assert rep["rms_px"] < 1e-3, (seed, rep["rms_px"])
