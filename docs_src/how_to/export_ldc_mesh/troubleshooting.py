"""Troubleshooting: `width`/`height` are optional for the mesh generator, but not for
image-level camera helpers.

`TI_LDC_MeshGenerator` sizes its mesh from the explicit `output_width`/`output_height`
arguments to `generate_mesh_and_intrinsics`, so it never needs `cam.width`/`cam.height`.
`DoubleSphereCamera.compute_K_new()` and `get_undistortion_maps()` are image-level helpers
that do need them, and raise `ValueError` without them.
"""

from ds_msp import DoubleSphereCamera
from ds_msp.ldc import TI_LDC_MeshGenerator


def main() -> None:
    cam = DoubleSphereCamera(711.57, 711.24, 949.18, 518.81, 0.183, 0.809)   # no width/height

    # Fine -- the mesh generator uses its own output_width/output_height arguments.
    gen = TI_LDC_MeshGenerator(cam)
    res = gen.generate_mesh_and_intrinsics(1920, 1080)
    print(res["mesh_lut"].shape)   # -> (69, 121, 2)  works without cam.width/cam.height

    # Raises -- cam.compute_K_new() needs the sensor dimensions.
    try:
        cam.compute_K_new()
    except ValueError as exc:
        print(f"ValueError: {exc}")

    # Fix: supply width and height on the camera if you also call image-level ops.
    cam_sized = DoubleSphereCamera(711.57, 711.24, 949.18, 518.81, 0.183, 0.809,
                                    width=1920, height=1080)
    K = cam_sized.compute_K_new()   # now works
    print(round(float(K[0, 0]), 2))   # -> 426.84, same balance=0.5 default as generate_mesh_and_intrinsics


if __name__ == "__main__":
    main()
