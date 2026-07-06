"""Chapter 8 -- the eight-point estimator on rays, and its epipolar residual.

`essential_from_rays` solves f2^T E f1 = 0 in least squares for the essential matrix
`E`. `epipolar_residual` returns f2^T E f1 per pair -- zero for a perfect fit. Rays are
regenerated with the same recipe as the chapter's opening example (`recover_pose_basic.py`).
"""

import numpy as np

from ds_msp.mvg import essential_from_rays, epipolar_residual


def rodrigues(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def main() -> None:
    rng = np.random.default_rng(1)
    R_true = rodrigues(rng.standard_normal(3), 0.6)
    t_true = rng.standard_normal(3)
    t_true /= np.linalg.norm(t_true)

    X1 = np.column_stack([rng.uniform(-2, 2, 40), rng.uniform(-2, 2, 40), rng.uniform(2, 8, 40)])
    X2 = (R_true @ X1.T).T + t_true
    f1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    f2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)

    E = essential_from_rays(f1, f2)                 # (3, 3), rank 2
    residual = epipolar_residual(E, f1, f2)         # (40,) algebraic residual, one per pair
    print(f"max epipolar residual: {np.abs(residual).max():.2e}")


if __name__ == "__main__":
    main()
