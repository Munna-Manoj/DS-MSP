"""The LDC mesh recipe is the same for every camera model.

`TI_LDC_MeshGenerator` uses only the `CameraModel` contract (`project()` and `K`), so a
Kannala-Brandt fisheye and an OCam polynomial camera go through exactly the code path a
Double Sphere camera does -- and so will any model added later. Keypoints are undistorted
with the model-agnostic closed form at the same `K_new`.
"""

import numpy as np

from ds_msp.ldc import TI_LDC_MeshGenerator
from ds_msp.models import KannalaBrandtModel, OCamModel
from ds_msp.ops.undistort import Undistorter


def main() -> None:
    for cam in (KannalaBrandtModel.sample(), OCamModel.sample()):
        gen = TI_LDC_MeshGenerator(cam)   # any CameraModel -- no model-specific code
        res = gen.generate_mesh_and_intrinsics(640, 480, downsample_factor=4, balance=0.5)

        print(cam.name, res["mesh_lut"].shape, res["mesh_lut"].dtype,
              bool(res["valid_mask"].all()))
        print(round(float(res["K_new"][0, 0]), 2), res["config"]["camera_model"]["name"])

        # Keypoints: the closed form for ANY model, at the mesh's own K_new.
        pts = np.array([[320.0, 240.0], [500.0, 150.0]])
        uv, valid = Undistorter(cam, 640, 480).undistort_points(pts, res["K_new"])
        print(uv.round(2).tolist(), valid.tolist())


if __name__ == "__main__":
    main()
