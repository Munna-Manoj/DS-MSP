"""Load a camera back out of calibrated_cameras_data.yml with rig.load_camera().

A real ds-msp-calibrate-rig run produces a RigState (per-camera CameraModel +
extrinsics) and writes it with ds_msp.io.mccalib.save_mccalib_cameras(). This
builds a minimal one-camera RigState by hand -- the same file shape a real rig
run writes, just without needing a multi-camera image capture -- so the round
trip through calibrated_cameras_data.yml runs standalone.
"""
import os
import tempfile

import numpy as np

from ds_msp.data.observations import RigState
from ds_msp.io.mccalib import save_mccalib_cameras
from ds_msp.models import KannalaBrandtModel
import ds_msp.rig as rig


def main() -> None:
    truth = KannalaBrandtModel(fx=900.0, fy=900.0, cx=960.0, cy=540.0,
                                k1=0.01, k2=-0.02, k3=0.0, k4=0.0)
    state = RigState(
        cameras={0: truth},
        T_c_g={0: np.eye(4)},
        ref_cam_id=0,
        object_poses={},
        objects={},
        img_size={0: (1920, 1080)},
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "calibrated_cameras_data.yml")
        save_mccalib_cameras(state, path)

        cam0 = rig.load_camera(path, 0)
        print(cam0)

        points_3d = np.array([[0.0, 0.0, 2.0], [0.3, -0.2, 1.5]])  # (2, 3) camera-frame, metres
        uv, valid = cam0.project(points_3d)
        print(f"uv[0] = ({uv[0, 0]:.3f}, {uv[0, 1]:.3f})   valid={valid[0]}")
        print(f"uv[1] = ({uv[1, 0]:.3f}, {uv[1, 1]:.3f})   valid={valid[1]}")


if __name__ == "__main__":
    main()
