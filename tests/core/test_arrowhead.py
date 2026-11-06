"""Block-arrowhead spectrum (ds_msp.core.arrowhead): exact inertia counts and the audit's
weak-direction analysis without forming the K x K matrix must agree with the dense solver."""

import numpy as np
import pytest

from ds_msp.core.arrowhead import ArrowheadSym, spectrum_bounds
from ds_msp.core.observability import eigen_weakness, eigen_weakness_arrowhead

pytestmark = pytest.mark.req("NFR-PERF-008")


def _arrow_jacobian(seed=0, c0=12, F=40, P=9, weak_block=True):
    """Stacked per-observation Jacobian rows with the rig layout [cams | poses | intrinsics]:
    each row block touches one pose, at most one camera and the intrinsics. Planted weak
    directions: two near-collinear intrinsics, a camera column tracking a pose column, and
    (``weak_block``) a near-singular pose block, which forces pivoting in the counts."""
    rng = np.random.default_rng(seed)
    K = c0 + 6 * F + P
    rows = []
    for f in range(F):
        for _ in range(rng.integers(2, 5)):
            J = np.zeros((10, K))
            J[:, c0 + 6 * f:c0 + 6 * f + 6] = rng.normal(size=(10, 6))
            cam = rng.integers(0, c0 // 6 + 1)
            if cam < c0 // 6:
                J[:, 6 * cam:6 * cam + 6] = rng.normal(size=(10, 6))
            J[:, c0 + 6 * F:] = rng.normal(size=(10, P))
            rows.append(J)
    J = np.vstack(rows)
    J[:, -1] = J[:, -2] * (1 + 1e-3) + 3e-3 * rng.normal(size=len(J))
    J[:, 0] = J[:, c0 + 6] + 1e-2 * rng.normal(size=len(J))
    if weak_block:
        fw = c0 + 6 * (F // 2)
        r = np.nonzero(J[:, fw + 4])[0]
        J[r, fw + 5] = J[r, fw + 4] + 1e-4 * rng.normal(size=len(r))
    J *= rng.uniform(0.1, 1e3, K)[None, :]                     # mixed units
    H = (J * rng.uniform(0.2, 2, len(J))[:, None]).T @ J
    return 0.5 * (H + H.T), c0, 6 * F


def _arrow(H, c0, n_pose):
    K = H.shape[0]
    core = np.r_[np.arange(c0), np.arange(c0 + n_pose, K)]
    blk = np.arange(c0, c0 + n_pose)
    D = np.stack([H[c0 + 6 * f:c0 + 6 * f + 6, c0 + 6 * f:c0 + 6 * f + 6]
                  for f in range(n_pose // 6)])
    A = ArrowheadSym(A=H[np.ix_(core, core)].copy(), B=H[np.ix_(core, blk)].copy(), D=D.copy(),
                     core=core, blocks=blk)
    np.testing.assert_array_equal(A.to_dense(), H)
    return A


@pytest.mark.parametrize("weak_block", [False, True])
def test_counts_equal_dense_inertia(weak_block):
    """ν(σ) is the number of eigenvalues below σ, including next to a near-singular block."""
    H, c0, n_pose = _arrow_jacobian(weak_block=weak_block)
    s = 1 / np.sqrt(np.diag(H))
    Hh = 0.5 * ((H * s[:, None]) * s[None, :] + ((H * s[:, None]) * s[None, :]).T)
    w = np.linalg.eigvalsh(Hh)
    sp = spectrum_bounds(_arrow(Hh, c0, n_pose))
    for k in (0, 1, 2, 3, len(w) // 2, len(w) - 1):
        lo, hi = w[k] - 1e-9 * w[-1], w[k] + 1e-9 * w[-1]
        assert sp.count(lo) == k and sp.count(hi) == k + 1
        assert abs(sp.kth(k) - w[k]) <= 1e-14 * w[-1]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_arrowhead_weakness_matches_dense(seed):
    H, c0, n_pose = _arrow_jacobian(seed=seed, F=30 + 10 * seed)
    arr = _arrow(H, c0, n_pose)
    a = eigen_weakness_arrowhead(arr, tau_rel=1e-3)
    d = eigen_weakness(H.copy(), tau_rel=1e-3)
    lmax = d["eigvals"][-1]
    nw = d["n_weak"]
    assert a["n_weak"] == nw >= 2
    assert a["pairs"] == d["pairs"]
    np.testing.assert_array_equal(a["scale"], d["scale"])
    np.testing.assert_allclose(a["eigvals"][:nw], d["eigvals"][:nw], rtol=0, atol=1e-14 * lmax)
    assert abs(a["median"] - np.median(d["eigvals"])) <= 1e-14 * lmax
    eps = np.finfo(float).eps
    n = H.shape[0]
    # cond = λmax/λmin: two backward-stable solvers agree to ~n·ε·‖H‖/λmin
    assert np.isclose(a["cond"], d["cond"], rtol=4 * n * eps * d["cond"])
    assert np.isclose(a["cond_raw"], d["cond_raw"], rtol=4 * n * eps * d["cond_raw"])
    for wa, wd in zip(a["weak"], d["weak"]):
        assert wa["participating"] == wd["participating"]
        np.testing.assert_allclose(wa["energy"], wd["energy"], atol=1e-8)


def test_arrowhead_weakness_stores_no_dense_matrix():
    """Peak allocation stays far below one K x K matrix on a pose-heavy layout."""
    import tracemalloc
    H, c0, n_pose = _arrow_jacobian(F=300, weak_block=True)
    arr = _arrow(H, c0, n_pose)
    K = H.shape[0]
    del H
    tracemalloc.start()
    eigen_weakness_arrowhead(arr, tau_rel=1e-3)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 0.25 * K * K * 8, (peak, K * K * 8)
