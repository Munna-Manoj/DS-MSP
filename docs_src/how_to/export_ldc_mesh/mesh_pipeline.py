"""Export a TI Jacinto LDC displacement mesh from a calibrated Double Sphere camera.

One continuous pipeline, shown incrementally on the how-to page: build the camera
(`cam`) -> wrap it in the mesh generator (`gen`) -> generate the mesh dict (`res`) ->
read the Q3 `mesh_lut` values -> sweep `downsample_factor` -> undistort keypoints with
the closed form at the same `K_new`. No external data -- the intrinsics are a
representative 1920x1080 fisheye calibration.
"""

import numpy as np

from ds_msp import DoubleSphereCamera
from ds_msp.ldc import TI_LDC_MeshGenerator


def main() -> None:
    # A calibrated Double Sphere camera (1920x1080 fisheye).
    # width/height are optional here -- the mesh generator ignores them. They matter
    # only if you also call cam.compute_K_new() / cam.get_undistortion_maps().
    cam = DoubleSphereCamera(fx=711.57, fy=711.24, cx=949.18, cy=518.81,
                             xi=0.183, alpha=0.809, width=1920, height=1080)

    gen = TI_LDC_MeshGenerator(cam)
    res = gen.generate_mesh_and_intrinsics(1920, 1080, downsample_factor=4, balance=0.5)

    mesh_lut = res["mesh_lut"]   # (69, 121, 2) int16 -- Q3 (h, v) displacements
    K_new = res["K_new"]         # (3, 3) rectified pinhole intrinsics

    print(mesh_lut.shape, mesh_lut.dtype)   # -> (69, 121, 2) int16
    print(round(float(K_new[0, 0]), 2))     # -> 426.84  (new focal length, px)

    # What the dict contains -- continues from `res`.
    print(list(res.keys()))
    print(res["config"]["mesh_size"])
    print(res["config"]["downsample_factor"])

    # Read the Q3 fixed-point format -- continues from `mesh_lut`. Each node holds
    # two int16 values (h, v) in Q3: pixels * 8, rounded. Divide by 8 to recover px.
    node = mesh_lut[34, 60]                  # the node at output pixel (960, 544)
    print(node)                             # -> (h, v) in Q3 units
    print(node / 8.0)                       # -> displacement in pixels

    print(int(mesh_lut.min()), int(mesh_lut.max()))   # Q3 range across the whole mesh

    # Trade mesh size against accuracy -- continues from `gen`.
    for m in (3, 4, 5):
        r = gen.generate_mesh_and_intrinsics(1920, 1080, downsample_factor=m, balance=0.5)
        print(m, 2**m, r["mesh_lut"].shape)

    # Undistort keypoints with the closed form, not the mesh -- continues from
    # `cam` and `K_new` (the SAME K_new the mesh above was generated with).
    pts = np.array([[960.0, 540.0],     # (N, 2) distorted fisheye keypoints, px
                    [1400.0, 800.0],
                    [600.0, 300.0]])
    und, valid = cam.undistort_points(pts, K_new)   # und: (N, 2) px, rectified frame
    print(und.round(2))
    print(valid)


if __name__ == "__main__":
    main()
