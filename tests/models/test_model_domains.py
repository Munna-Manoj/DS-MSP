"""Every point a model calls valid must unproject back to its own ray.

Random in-bounds parameters times random rays over the whole sphere. A model whose
validity test is too loose (a point past a fold, on the far root of the distortion
polynomial, or behind the camera) fails here, because two rays share one pixel there.
"""

import numpy as np
import pytest

from ds_msp.models.eucm_math import eucm_project, eucm_unproject
from ds_msp.models.kb_math import kb_project, kb_unproject

pytestmark = pytest.mark.req("NFR-NUM-014")

K = (500.0, 500.0, 640.0, 480.0)


def _sphere(rng, n, max_deg=180.0):
    th = rng.uniform(0.0, np.deg2rad(max_deg), n)
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]


def _round_trip(project, unproject, params, P):
    u, v, ok = project(P, *K, *params)
    assert np.isfinite(u[ok]).all() and np.isfinite(v[ok]).all()
    rays, back = unproject(np.c_[u, v][ok], *K, *params)
    assert back.all(), f"{(~back).sum()} valid points do not unproject, params {params}"
    np.testing.assert_allclose(rays, P[ok], atol=1e-6, err_msg=f"params {params}")


def test_eucm_round_trip_over_the_sphere():
    rng = np.random.default_rng(0)
    for _ in range(200):
        params = (rng.uniform(0.01, 0.99), rng.uniform(0.01, 3.0))
        _round_trip(eucm_project, eucm_unproject, params, _sphere(rng, 300))


def test_kb_round_trip_over_the_sphere():
    rng = np.random.default_rng(1)
    for _ in range(200):
        params = tuple(rng.uniform(-0.5, 0.5, 4) * [1.0, 0.3, 0.1, 0.03])
        _round_trip(kb_project, kb_unproject, params, _sphere(rng, 300))

