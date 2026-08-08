"""Recover pose from 3D<->2D correspondences, for any fisheye/omni model.

`solve_pnp` depends only on the `CameraModel` contract: it unprojects to bearing
rays and solves on bearings across the full model-valid field of view -- so it works
unchanged for any registered model, not just Double Sphere.
"""
import numpy as np

from ds_msp import DoubleSphereModel, solve_pnp
from docs_src import stable_display_round


def main() -> None:
    cam = DoubleSphereModel.sample()

    object_points = np.array([[0, 0, 0], [0.1, 0, 0],      # (N, 3) known 3D points, metres
                              [0, 0.1, 0], [0.1, 0.1, 0]], dtype=float)
    image_points = np.array([[610, 480], [720, 470],       # (N, 2) their pixels
                             [600, 590], [715, 580]], dtype=float)

    ok, rvec, tvec = solve_pnp(cam, object_points, image_points)
    print(f"ok={ok}")
    # Display-only quantization: supported numerical backends place rvec[1] on opposite
    # sides of a four-decimal tie.  One guard digit plus an explicit tie rule keeps the
    # documented output stable without changing the full-precision pose.
    print(f"rvec={stable_display_round(rvec, 4).tolist()}")
    print(f"tvec={stable_display_round(tvec, 4).tolist()}")


if __name__ == "__main__":
    main()
