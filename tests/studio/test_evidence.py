"""One-line answers of Studio's evidence views (FR-UI-005)."""
import numpy as np
import pytest

from ds_msp.studio.evidence import error_pattern, lens_zones

pytestmark = pytest.mark.req("FR-UI-005")
W, H = 1200, 800


def _corners(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, W, n), rng.uniform(0, H, n), rng


def test_random_errors_read_as_no_pattern():
    u, v, rng = _corners()
    p = error_pattern(u, v, rng.normal(0, 0.2, len(u)), rng.normal(0, 0.2, len(u)), W, H)
    assert p["kind"] == "none" and not any(c["biased"] for c in p["grid"]["cells"])


def test_radial_growth_reads_as_edges():
    u, v, rng = _corners()
    rho = np.hypot(u - W / 2, v - H / 2) / np.hypot(W / 2, H / 2)
    s = 0.1 + 0.6 * rho ** 2
    p = error_pattern(u, v, rng.normal(0, 1, len(u)) * s, rng.normal(0, 1, len(u)) * s, W, H)
    assert p["kind"] == "edges" and p["edge_growth"] > 1.5


def test_a_shifted_region_reads_as_regional_and_names_it():
    u, v, rng = _corners()
    dx, dy = rng.normal(0, 0.2, len(u)), rng.normal(0, 0.2, len(u))
    m = (u < W / 6) & (v < H / 4)
    dx[m] += 0.5
    p = error_pattern(u, v, dx, dy, W, H)
    assert p["kind"] == "regional" and "top-left" in p["answer"]
    assert [(c["i"], c["j"]) for c in p["grid"]["cells"] if c["biased"]] == [(0, 0)]


def test_lens_zones_split_and_place_the_uncovered_part():
    nx, ny = 24, 18
    uu, vv = np.meshgrid((np.arange(nx) + 0.5) * W / nx, (np.arange(ny) + 0.5) * H / ny)
    u, v = uu.ravel(), vv.ravel()
    observed = u > W / 3                                   # left third never seen
    sigma = np.where(u > W * 0.9, 2.0, 0.1)                # right strip over a 1 px limit
    z = lens_zones({"u": u.tolist(), "v": v.tolist(), "sigma_px": sigma.tolist(),
                    "observed": observed.tolist(), "limit_px": 1.0, "unbounded": False}, W, H)
    assert z["share"]["extrapolated"] == pytest.approx(8 / 24)
    assert z["share"]["uncertain"] == pytest.approx(2 / 24)      # cell centres beyond 0.9 W
    assert "left" in z["answer"]
    whole = lens_zones({"u": u.tolist(), "v": v.tolist(), "sigma_px": [0.1] * len(u),
                        "observed": [True] * len(u), "limit_px": 1.0, "unbounded": False}, W, H)
    assert whole["answer"] == "Trusted across the whole image."
