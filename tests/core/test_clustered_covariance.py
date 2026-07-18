"""Frame-clustered sandwich covariance: closes the real-data under-coverage the plain
(unclustered) sandwich measures when per-block scores are correlated within a cluster.

Derivation + full Monte-Carlo numbers:
.ai/experiments/2026-07-17-stage-I-frame-clustered-sandwich-derivation.md. This module is
the fast (reduced seed-count) regression version of that derivation's verification script.
"""

from __future__ import annotations

import numpy as np
import pytest

from ds_msp.core.covariance import clustered_sandwich_covariance, sandwich_covariance

pytestmark = pytest.mark.req("FR-RIG-020")

P = 4
N_FRAMES = 30
CORNERS_PER_FRAME = 20
BLOCK = 2
SIGMA_E = 0.10                  # independent per-corner noise
SIGMA_B = 0.30                  # shared per-frame offset -- the correlation source
THETA_STAR = np.array([1.0, -2.0, 0.5, 3.0])

N_CORNERS = N_FRAMES * CORNERS_PER_FRAME
N_ROWS = N_CORNERS * BLOCK
FRAME_ID = np.repeat(np.arange(N_FRAMES), CORNERS_PER_FRAME)


def _design(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N_ROWS, P))
    # last column is frame-constant: a shared parameter identified through frame-level
    # structure (effective sample size G, not N) -- the calibration analogue (e.g. a
    # camera extrinsic identified only through many per-frame board-pose constraints).
    frame_cov = rng.normal(size=N_FRAMES)
    A[:, -1] = np.repeat(frame_cov, CORNERS_PER_FRAME * BLOCK)
    return A


def _draw_y(A, seed):
    rng = np.random.default_rng(seed)
    clean = A @ THETA_STAR
    e = rng.normal(scale=SIGMA_E, size=N_ROWS)
    b_frame = rng.normal(scale=SIGMA_B, size=(N_FRAMES, BLOCK))
    b = np.repeat(b_frame, CORNERS_PER_FRAME, axis=0).reshape(N_ROWS)
    return clean + e + b


def test_clustered_reduces_to_plain_sandwich_when_every_block_is_its_own_cluster():
    """cluster_id = arange(n) (every block its own cluster) must reproduce the plain
    (unclustered) sandwich, up to the differing HC1 dof-correction convention (rows vs
    blocks -- negligible at this N,P)."""
    A = _design()
    rng = np.random.default_rng(11)
    y = A @ THETA_STAR + rng.normal(scale=SIGMA_E, size=N_ROWS)
    th = np.linalg.lstsq(A, y, rcond=None)[0]
    r = A @ th - y
    trivial_clusters = np.arange(N_CORNERS)
    d_clu = np.diag(clustered_sandwich_covariance(A, r, trivial_clusters, kernel="none",
                                                   small_cluster_correction=False))
    d_sw = np.diag(sandwich_covariance(A, r, kernel="none"))
    assert np.allclose(d_clu, d_sw, rtol=0.05), (d_clu, d_sw)


def test_clustering_closes_frame_level_undercoverage():
    """Manufacture KNOWN within-frame-correlated noise (independent per-corner draw PLUS
    a shared per-frame offset). Over seeded refits, the plain sandwich badly under-covers
    the frame-level parameter direction (predicted std << true empirical std); the
    frame-clustered sandwich closes it to within a loose acceptance band. Reduced-seed
    fast regression of the full 4000-seed derivation result (unclustered 0.161, clustered
    0.979 on that run)."""
    A = _design()
    N_MC = 500
    thetas, sw_var, clu_var = [], [], []
    for k in range(N_MC):
        y = _draw_y(A, 10_000 + k)
        th = np.linalg.lstsq(A, y, rcond=None)[0]
        r = A @ th - y
        thetas.append(th)
        sw_var.append(np.diag(sandwich_covariance(A, r, kernel="none")))
        clu_var.append(np.diag(clustered_sandwich_covariance(A, r, FRAME_ID, kernel="none")))

    thetas = np.asarray(thetas)
    mc_std = thetas.std(axis=0, ddof=1)
    sw_std = np.sqrt(np.mean(sw_var, axis=0))
    clu_std = np.sqrt(np.mean(clu_var, axis=0))

    ratio_sw = sw_std / mc_std
    ratio_clu = clu_std / mc_std

    # frame-level param (index 3): plain sandwich badly under-covers ...
    assert ratio_sw[3] < 0.4, f"expected plain sandwich to under-cover: ratio={ratio_sw[3]}"
    # ... clustering closes it to within a loose acceptance band around 1.0
    assert 0.6 < ratio_clu[3] < 1.5, f"clustered ratio out of band: {ratio_clu[3]}"
    # per-corner params (0-2): both estimators already fine (no fix needed there)
    assert np.all(ratio_sw[:3] > 0.7) and np.all(ratio_sw[:3] < 1.3)
    assert np.all(ratio_clu[:3] > 0.7) and np.all(ratio_clu[:3] < 1.3)


def test_clustered_covariance_rejects_mismatched_cluster_id_length():
    A = _design()
    rng = np.random.default_rng(3)
    y = A @ THETA_STAR + rng.normal(scale=SIGMA_E, size=N_ROWS)
    r = A @ np.zeros(P) - y
    with pytest.raises(ValueError):
        clustered_sandwich_covariance(A, r, FRAME_ID[:-1], kernel="none")


def test_clustered_covariance_symmetric_psd():
    A = _design()
    rng = np.random.default_rng(4)
    y = A @ THETA_STAR + rng.normal(scale=SIGMA_E, size=N_ROWS)
    th = np.linalg.lstsq(A, y, rcond=None)[0]
    r = A @ th - y
    cov = clustered_sandwich_covariance(A, r, FRAME_ID, kernel="none")
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-9)
