"""Minimal quickstart shown on the docs site landing page (docs/index.md).

Builds a Double Sphere camera from real calibrated intrinsics, projects a
batch of 3D points to pixels, and unprojects them back to unit bearing rays
-- verifying project() and unproject() are exact inverses of each other.
"""
import numpy as np

from ds_msp import DoubleSphereCamera


def main() -> None:
    cam = DoubleSphereCamera(fx=711.57, fy=711.24, cx=949.18, cy=518.81,
                              xi=0.183, alpha=0.809, width=1920, height=1080)

    rng = np.random.default_rng(0)
    pts_3d = rng.uniform(-2.0, 2.0, size=(500, 3))
    pts_3d[:, 2] = np.abs(pts_3d[:, 2]) + 0.5  # (500, 3) camera-frame pts, in front of the lens

    px, ok = cam.project(pts_3d)      # (500, 2) pixels
    rays, ok2 = cam.unproject(px)     # (500, 3) unit bearing rays
    px2, ok3 = cam.project(rays)      # re-project -- should land back on `px`

    valid = ok & ok2 & ok3
    print(f"{valid.sum()}/{len(pts_3d)} points valid through the round trip")
    print(f"max round-trip error: {np.abs(px2 - px)[valid].max():.2e} px")


if __name__ == "__main__":
    main()
