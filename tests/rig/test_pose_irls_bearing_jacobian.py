"""The critical test: the chordal-bearing pose Jacobian must match finite differences.

``bearing_chordal_residual_jacobian`` (ADR-0020) is the analytic core of the now fully
bearing-native ``robust_pose_irls`` -- a regression here is the classic source of an IRLS that
"converges" to the wrong pose, silently, because the normal equations are built from a wrong
gradient.
"""
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp, se3_exp
from ds_msp.rig.pose_init import bearing_chordal_residual_jacobian


def _random_scene(seed=0, n=40, include_peripheral=True):
    """World points and their observed unit bearings under a ground-truth pose, spanning the
    full sphere (some >90 deg off-axis) when ``include_peripheral``."""
    rng = np.random.default_rng(seed)
    R_gt, t_gt = so3_exp([0.3, -0.2, 0.15]), np.array([0.1, -0.05, 0.4])
    if include_peripheral:
        dirs = rng.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        depths = rng.uniform(0.5, 3.0, n)
        Xc = dirs * depths[:, None]
    else:
        Xc = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n),
                              rng.uniform(0.5, 3.0, n)])
    X = (Xc - t_gt) @ R_gt                            # world points: Xc = R_gt X + t_gt
    f = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    return X, f, R_gt, t_gt


def test_scene_genuinely_spans_past_90deg():
    X, f, R_gt, t_gt = _random_scene()
    off = np.degrees(np.arccos(np.clip(f[:, 2], -1, 1)))
    assert (off > 90).sum() >= 10 and off.max() > 150.0


@pytest.mark.parametrize("include_peripheral", [True, False])
def test_jacobian_matches_finite_difference(include_peripheral):
    X, f, R_gt, t_gt = _random_scene(seed=1, include_peripheral=include_peripheral)
    # evaluate at a pose *near* but not at ground truth (a realistic mid-iteration state)
    T = np.eye(4)
    T[:3, :3] = R_gt @ so3_exp([0.02, -0.01, 0.03])
    T[:3, 3] = t_gt + np.array([0.01, -0.02, 0.03])
    foc = 400.0

    _, J = bearing_chordal_residual_jacobian(T, X, f, foc)
    eps = 1e-6
    for k in range(6):                                # 6 tangent directions: [δt(3), δω(3)]
        d = np.zeros(6)
        d[k] = eps
        Tp = se3_exp(d) @ T
        Tm = se3_exp(-d) @ T
        ep, _ = bearing_chordal_residual_jacobian(Tp, X, f, foc)
        em, _ = bearing_chordal_residual_jacobian(Tm, X, f, foc)
        fd = (ep - em) / (2 * eps)                    # (n, 3)
        analytic = J[:, :, k]                          # (n, 3)
        assert np.allclose(analytic, fd, atol=1e-5, rtol=1e-4), \
            f"column {k} mismatch (max err {np.abs(analytic - fd).max():.2e}, " \
            f"include_peripheral={include_peripheral})"


def test_residual_is_zero_at_ground_truth():
    """Sanity: e_i = 0 exactly when the predicted bearing equals the observed one."""
    X, f, R_gt, t_gt = _random_scene(seed=2)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_gt, t_gt
    e, _ = bearing_chordal_residual_jacobian(T, X, f, 400.0)
    assert np.abs(e).max() < 1e-10


def test_antipodal_prediction_is_not_a_perfect_fit():
    """The corrected chordal residual assigns the antipode its maximum cost."""
    X = np.array([[0.0, 0.0, -2.0]])
    f = np.array([[0.0, 0.0, 1.0]])
    e, _ = bearing_chordal_residual_jacobian(np.eye(4), X, f, 400.0)
    assert np.linalg.norm(e[0]) == 800.0


# Traceability: same requirement as the wide-FOV PnP suite this Jacobian serves.
pytestmark = pytest.mark.req("FR-CALIB-002")
