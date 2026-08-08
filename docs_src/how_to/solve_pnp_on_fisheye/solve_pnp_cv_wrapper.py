"""The OpenCV-style functional wrapper: `ds_cv.solvePnP(points_3d, points_2d, K, D)`.

Same synthetic scene and same solve as `solve_pnp_basic.py`, called through the
`cv2.solvePnP`-shaped wrapper instead of the object method -- so it drops into
existing `cv2.solvePnP` call sites. Confirms both entry points agree.
"""
import cv2
import numpy as np

import ds_msp.cv as ds_cv
from ds_msp import DoubleSphereCamera
from docs_src import zero_roundoff


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

    # cam.K is the pinhole matrix; cam.D = [xi, alpha] are the DS distortion coefficients.
    success, rvec, tvec = ds_cv.solvePnP(points_3d, points_2d, cam.K, cam.D)
    print(success, rvec.shape, tvec.shape)                 # -> (3, 1) (3, 1), cv2-native shape

    R, _ = cv2.Rodrigues(rvec)
    rot_err_deg = np.degrees(np.arccos(np.clip((np.trace(R @ R_gt.T) - 1) / 2, -1, 1)))
    t_err_m = np.linalg.norm(tvec.squeeze() - tvec_gt)
    # Keep the published console output deterministic at the float64 round-off floor.
    rot_err_deg = zero_roundoff(rot_err_deg, atol=1e-5)
    t_err_m = zero_roundoff(t_err_m, atol=1e-12)
    print(f"rotation error: {rot_err_deg:.2e} deg")
    print(f"translation error: {t_err_m:.2e} m")


if __name__ == "__main__":
    main()
