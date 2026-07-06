"""Hand the same fisheye pixels to `cv2.solvePnP` with a pinhole `K` -- it fits the wrong model.

Same synthetic scene as `solve_pnp_basic.py`, but solved with plain `cv2.solvePnP`
(pinhole assumption, zero distortion) instead of `cam.solve_pnp`. Contrasts the
recovered-pose error against the fisheye-aware solve.
"""
import cv2
import numpy as np

from ds_msp import DoubleSphereCamera


def main() -> None:
    cam = DoubleSphereCamera(fx=711.57, fy=711.24, cx=949.18, cy=518.81,
                             xi=0.183, alpha=0.809, width=1920, height=1080)

    rvec_gt = np.array([0.05, -0.10, 0.02])
    tvec_gt = np.array([0.30, -0.20, 1.00])
    R_gt, _ = cv2.Rodrigues(rvec_gt)

    rng = np.random.default_rng(0)
    points_3d = rng.uniform([-2, -2, 4], [2, 2, 8], size=(40, 3))

    P_cam = (R_gt @ points_3d.T + tvec_gt[:, None]).T
    uv, valid = cam.project(P_cam)
    points_2d = uv[valid]
    points_3d = points_3d[valid]

    ok, rv, tv = cv2.solvePnP(points_3d.astype(np.float64),
                              points_2d.astype(np.float64),
                              cam.K, np.zeros(5))               # pinhole assumption
    R_bad, _ = cv2.Rodrigues(rv)
    bad_rot = np.degrees(np.arccos(np.clip((np.trace(R_bad @ R_gt.T) - 1) / 2, -1, 1)))
    print(f"cv2 rotation error: {bad_rot:.2f} deg")
    print(f"cv2 translation error: {np.linalg.norm(tv.squeeze() - tvec_gt):.2f} m")


if __name__ == "__main__":
    main()
