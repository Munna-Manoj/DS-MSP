"""core/observability.py — equilibration, weak-direction detection, coverage (FR-RIG-021)."""
import numpy as np
import pytest

from ds_msp.core.observability import (eigen_weakness, equilibrate, orientation_spread,
                                       radial_occupancy)


@pytest.mark.req("FR-RIG-021")
def test_equilibrate_unit_diagonal_and_scale_invariance():
    """Ĥ has unit diagonal, and rescaling a column's units (m→mm, 1e6 in H) leaves Ĥ
    unchanged — the exact mixed-units artefact the equilibration removes."""
    rng = np.random.default_rng(0)
    J = rng.normal(size=(60, 6))
    H = J.T @ J
    Hh, _s = equilibrate(H)
    assert np.allclose(np.diag(Hh), 1.0, atol=1e-9)
    D = np.diag([1.0, 1e3, 1.0, 1e-3, 1.0, 1.0])       # a units change of two columns
    Hh2, _ = equilibrate(D @ H @ D)
    assert np.allclose(Hh, Hh2, atol=1e-9)


@pytest.mark.req("FR-RIG-021")
def test_eigen_weakness_finds_planted_null_direction():
    """A rank-deficient H (one exactly-dependent column pair) yields exactly one weak
    direction whose participation energy concentrates on the two dependent columns."""
    rng = np.random.default_rng(1)
    B = rng.normal(size=(80, 5))
    J = np.column_stack([B, B[:, 0]])                   # col 5 duplicates col 0
    H = J.T @ J
    out = eigen_weakness(H, tau_rel=1e-6)
    assert out["n_weak"] == 1
    energy = out["weak"][0]["energy"]
    assert energy[0] + energy[5] > 0.95
    assert any({i, j} == {0, 5} and corr > 0.99 for i, j, corr in out["pairs"])


@pytest.mark.req("FR-RIG-021")
def test_eigen_weakness_silent_on_well_conditioned_problem():
    rng = np.random.default_rng(2)
    J = rng.normal(size=(200, 8))
    out = eigen_weakness(J.T @ J, tau_rel=1e-6)
    assert out["n_weak"] == 0
    assert out["cond"] < 1e3


@pytest.mark.req("FR-RIG-021")
def test_radial_occupancy_flags_missing_periphery():
    rng = np.random.default_rng(3)
    center = np.array([640.0, 480.0])
    R = float(np.hypot(640, 480))
    # points confined to the central 40% of the radius
    ang = rng.uniform(0, 2 * np.pi, 500)
    rad = rng.uniform(0, 0.4 * R, 500)
    uv = center + np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    occ, periph = radial_occupancy(uv, center, R=R)
    assert periph == 0.0
    assert occ[-1] == 0.0                                # outer equal-area annulus empty
    # uniform-over-disk control: every equal-area annulus roughly equally occupied
    rad_u = R * np.sqrt(rng.uniform(0, 1, 4000))
    uv_u = center + np.column_stack([rad_u * np.cos(ang := rng.uniform(0, 2 * np.pi, 4000)),
                                     rad_u * np.sin(ang)])
    occ_u, periph_u = radial_occupancy(uv_u, center, R=R)
    assert np.all(occ_u > 0.1) and periph_u > 0.2


@pytest.mark.req("FR-RIG-021")
def test_orientation_spread_separates_aligned_from_diverse():
    aligned = np.tile([0.0, 0.0, 1.0], (30, 1))
    _eig, div0 = orientation_spread(aligned)
    assert div0 < 1e-9
    # sign flips do not fake diversity (a normal and its negative are the same plane)
    flipped = aligned.copy()
    flipped[::2] *= -1.0
    _eig, div_f = orientation_spread(flipped)
    assert div_f < 1e-9
    rng = np.random.default_rng(4)
    diverse = rng.normal(size=(200, 3))
    _eig, div1 = orientation_spread(diverse)
    assert div1 > 0.4
