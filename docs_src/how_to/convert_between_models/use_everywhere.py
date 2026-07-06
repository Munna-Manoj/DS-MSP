"""How-to: use a converted model with the rest of DS-MSP -- e.g. solve_pnp.

Every DS-MSP service depends only on the CameraModel contract, so a converted
model works everywhere the source model did. Converting is a one-line swap in
a pipeline.
"""
import numpy as np

from ds_msp import DoubleSphereModel, KannalaBrandtModel, convert, solve_pnp


def main() -> None:
    ds = DoubleSphereModel.sample()
    kb, _ = convert(ds, KannalaBrandtModel, width=1920, height=1080)   # DS -> OpenCV fisheye

    object_points = np.array([[0, 0, 0], [0.1, 0, 0],           # (N, 3) known 3D points, metres
                              [0, 0.1, 0], [0.1, 0.1, 0]], dtype=float)
    image_points = np.array([[610, 480], [720, 470],            # (N, 2) their pixels
                             [600, 590], [715, 580]], dtype=float)

    ok, rvec, tvec = solve_pnp(kb, object_points, image_points)   # same call as for the DS model
    print(f"ok={ok}")


if __name__ == "__main__":
    main()
