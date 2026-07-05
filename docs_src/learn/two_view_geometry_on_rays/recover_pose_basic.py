"""Chapter 8 -- the smallest thing that works: rays in, pose out.

`recover_pose` takes >=8 unit bearing-ray correspondences and returns the relative
rotation, the unit translation direction, and the triangulated 3D points. Rays here
are built directly from ground-truth 3D points (no pixels, no lens model), so with no
noise to absorb the recovered pose is exact to float64 round-off.
"""

import numpy as np

from ds_msp.mvg import recover_pose


def rodrigues(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rot_err_deg(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))


def dir_err_deg(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1)))


def main() -> None:
    # A ground-truth relative pose (R, t) maps a point from camera 1 to camera 2: X2 = R @ X1 + t.
    rng = np.random.default_rng(1)
    R_true = rodrigues(rng.standard_normal(3), 0.6)                     # 0.6 rad rotation
    t_true = rng.standard_normal(3)
    t_true /= np.linalg.norm(t_true)                                   # unit translation

    # 40 random 3D points in front of camera 1, seen by both cameras.
    X1 = np.column_stack([rng.uniform(-2, 2, 40), rng.uniform(-2, 2, 40), rng.uniform(2, 8, 40)])
    X2 = (R_true @ X1.T).T + t_true

    # Turn each point into a unit BEARING RAY in its camera (no pixels, no model -- just directions).
    f1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)                # (40, 3) unit rays, camera 1
    f2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)                # (40, 3) unit rays, camera 2

    R, t, X = recover_pose(f1, f2)                                     # R: (3,3), t: (3,), X: (40,3)

    print(f"rotation error       : {rot_err_deg(R_true, R):.2e} deg")
    print(f"translation-dir error: {dir_err_deg(t_true, t):.2e} deg")


if __name__ == "__main__":
    main()
