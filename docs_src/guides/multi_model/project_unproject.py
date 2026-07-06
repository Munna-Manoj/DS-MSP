"""The core 2D<->3D geometry: project and unproject, identical on every model.

`project` maps camera-frame 3D points to pixels; `unproject` is its inverse. Every
DS-MSP model implements both, so this exact code works unchanged if `cam` were a
KannalaBrandtModel, EUCMModel, or any other registered model instead.
"""
import numpy as np

from ds_msp import DoubleSphereModel


def main() -> None:
    cam = DoubleSphereModel.sample()   # any model works here -- swap this one line

    # 3D camera-frame points (N, 3) -> pixels (N, 2) + per-point validity mask
    pts_3d = np.array([[0.1, 0.0, 2.0], [0.4, -0.2, 3.0]])
    uv, valid = cam.project(pts_3d)
    print(uv.round(3).tolist())
    print(valid.tolist())

    # pixels (N, 2) -> unit bearing rays (N, 3) + validity
    rays, valid = cam.unproject(uv)              # rays are unit-norm
    print(rays.round(6).tolist())
    print(np.linalg.norm(rays, axis=1).round(6).tolist())   # -> [1.0, 1.0]


if __name__ == "__main__":
    main()
