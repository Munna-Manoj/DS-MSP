"""Spherical rectification map generation on the OCam and DS+ camera models.

``tests/stereo/test_rectify.py`` covers the rectifying-rotation geometry (model-free) and one
``rectify_maps`` run on a ``DoubleSphereModel``. These add the same ``rectify_maps`` smoke
coverage for the polynomial OCam and DS+ models, which are otherwise absent from the stereo
suite: the map must be the chart's shape, have imageable pixels, and be finite where valid.
"""

import numpy as np
import pytest

from ds_msp.ops import Equirectangular
from ds_msp.stereo.rectify import rectify_maps, rectifying_rotation
from ds_msp.models.ocam import OCamModel
from ds_msp.models.dsplus import DSPlusModel


def _ocam_120():
    """OCam sized for a 120x120 fisheye (centre 60,60; focal ~ |a0|=60)."""
    m = OCamModel(60.0, 60.0, 1.0, 0.0, 0.0, -60.0, 0.0, 6.0, -1.5, 0.25)
    m.width, m.height = 120, 120
    return m


def _dsplus_120():
    m = DSPlusModel(130.0, 130.0, 60.0, 60.0, alpha=0.55)
    m.width, m.height = 120, 120
    return m


@pytest.mark.parametrize("factory", [_ocam_120, _dsplus_120], ids=["ocam", "dsplus"])
def test_rectify_maps_runs_on_camera_model(factory):
    cam = factory()
    chart = Equirectangular(360, 180, hfov_deg=200)
    R_rect = rectifying_rotation(np.array([0.0, -1.0, 0.0]))
    mapx, mapy, valid = rectify_maps(cam, R_rect, chart)
    assert mapx.shape == (180, 360) and mapy.shape == (180, 360)
    assert valid.any()                                       # some rays are imageable
    assert np.isfinite(mapx[valid]).all() and np.isfinite(mapy[valid]).all()

# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-STEREO-002")
