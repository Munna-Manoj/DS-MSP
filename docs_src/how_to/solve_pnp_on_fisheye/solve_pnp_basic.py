"""Recover a fisheye camera's pose from 3D<->2D correspondences with `cam.solve_pnp`.

Builds a known ground-truth pose, projects 3D points through a Double Sphere model to
make 2D fisheye correspondences, asks `solve_pnp` to recover the pose, then measures
the error against ground truth. No external data needed -- everything is synthetic.
"""
import cv2
import numpy as np

from ds_msp import DoubleSphereCamera


def main() -> None:
    cam = DoubleSphereCamera(fx=711.57, fy=711.24, cx=949.18, cy=518.81,
                             xi=0.183, alpha=0.809, width=1920, height=1080)

    # 1. A ground-truth pose (what we want to recover).
    rvec_gt = np.array([0.05, -0.10, 0.02])      # Rodrigues vector, rad
    tvec_gt = np.array([0.30, -0.20, 1.00])      # translation, metres
    R_gt, _ = cv2.Rodrigues(rvec_gt)

    # 2. 40 world points spread in front of the camera.
    rng = np.random.default_rng(0)
    points_3d = rng.uniform([-2, -2, 4], [2, 2, 8], size=(40, 3))   # (40, 3) metres

    # 3. Project them through the fisheye to get 2D correspondences.
    P_cam = (R_gt @ points_3d.T + tvec_gt[:, None]).T               # (40, 3) camera frame
    uv, valid = cam.project(P_cam)                                  # uv: (40, 2) pixels
    points_2d = uv[valid]
    points_3d = points_3d[valid]

    # 4. Recover the pose from the 3D<->2D correspondences.
    success, rvec, tvec = cam.solve_pnp(points_3d, points_2d)
    print(success, len(points_3d))

    # 5. Measure the error against ground truth.
    R, _ = cv2.Rodrigues(rvec)
    rot_err_deg = np.degrees(np.arccos(np.clip((np.trace(R @ R_gt.T) - 1) / 2, -1, 1)))
    t_err_m = np.linalg.norm(tvec - tvec_gt)
    print(f"rotation error: {rot_err_deg:.2e} deg")
    print(f"translation error: {t_err_m:.2e} m")


if __name__ == "__main__":
    main()
