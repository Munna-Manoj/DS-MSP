"""Chapter 8 -- RANSAC against wrong matches, vs. the naive eight-point.

Corrupts 30% of camera-2 rays with random directions (the "wrong matches"), then
compares the naive least-squares eight-point against `ransac_relative_pose`, which
scores candidates by the angular Sampson residual on the sphere.
"""

import numpy as np

from ds_msp.mvg import recover_pose, essential_from_rays, ransac_relative_pose


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
    rng = np.random.default_rng(3)
    R_true = rodrigues(rng.standard_normal(3), 0.6)
    t_true = rng.standard_normal(3)
    t_true /= np.linalg.norm(t_true)
    X1 = np.column_stack([rng.uniform(-2, 2, 120), rng.uniform(-2, 2, 120), rng.uniform(2, 8, 120)])
    X2 = (R_true @ X1.T).T + t_true
    f1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    f2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)

    # Corrupt 30% of camera-2 rays with random directions (the "wrong matches").
    rng2 = np.random.default_rng(4)
    outlier = rng2.random(120) < 0.30
    f2_bad = f2.copy()
    f2_bad[outlier] = rng2.standard_normal((int(outlier.sum()), 3))
    f2_bad /= np.linalg.norm(f2_bad, axis=1, keepdims=True)

    # Naive eight-point on the contaminated rays:
    R_naive, _, _ = recover_pose(f1, f2_bad, essential_from_rays(f1, f2_bad))

    # Robust: RANSAC with a 0.005 rad (~0.3 deg) angular inlier threshold.
    R_rob, t_rob, inliers = ransac_relative_pose(f1, f2_bad, threshold=0.005, seed=0)

    truth = ~outlier
    precision = (inliers & truth).sum() / max(inliers.sum(), 1)
    recall = (inliers & truth).sum() / truth.sum()

    print(f"naive  rotation error : {rot_err_deg(R_true, R_naive):.2f} deg")
    print(f"RANSAC rotation error : {rot_err_deg(R_true, R_rob):.3f} deg")
    print(f"RANSAC trans-dir error: {dir_err_deg(t_true, t_rob):.3f} deg")
    print(f"inlier precision/recall: {precision:.3f} / {recall:.3f}  ({int(inliers.sum())}/120)")


if __name__ == "__main__":
    main()
