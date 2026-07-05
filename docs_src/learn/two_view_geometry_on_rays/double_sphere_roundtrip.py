"""Chapter 8 -- two-view pose recovery through a real Double Sphere fisheye camera.

Projects 3D points to fisheye pixels in two views, unprojects back to rays, and
recovers the pose with `recover_pose` -- proving the pipeline is model-agnostic. This
mirrors `tests/mvg/test_two_view.py::test_recover_pose_through_a_real_double_sphere_camera`.
"""

import numpy as np

from ds_msp.models import DoubleSphereModel
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
    cam = DoubleSphereModel(fx=300.0, fy=300.0, cx=320.0, cy=320.0, xi=0.3, alpha=0.6)
    rng = np.random.default_rng(7)
    R_true = rodrigues(rng.standard_normal(3), 0.5)
    t_true = rng.standard_normal(3)
    t_true /= np.linalg.norm(t_true)

    X1 = np.column_stack([rng.uniform(-3, 3, 60), rng.uniform(-3, 3, 60), rng.uniform(2, 9, 60)])
    X2 = (R_true @ X1.T).T + t_true

    uv1, ok1 = cam.project(X1)            # 3D -> fisheye pixels, view 1; ok1: (60,) valid mask
    uv2, ok2 = cam.project(X2)            # 3D -> fisheye pixels, view 2
    ok = ok1 & ok2
    f1, _ = cam.unproject(uv1[ok])        # pixels -> unit rays, view 1
    f2, _ = cam.unproject(uv2[ok])        # pixels -> unit rays, view 2

    R, t, X = recover_pose(f1, f2)
    print(f"valid pairs          : {int(ok.sum())}")
    print(f"rotation error       : {rot_err_deg(R_true, R):.2e} deg")
    print(f"translation-dir error: {dir_err_deg(t_true, t):.2e} deg")


if __name__ == "__main__":
    main()
