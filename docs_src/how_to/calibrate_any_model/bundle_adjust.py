"""Bundle-adjust a Double Sphere model from correspondences you already have.

``calibrate`` doesn't care how ``(X_world, keypoints, visibility)`` were built --
AprilGrid detection is one source (see ``examples/03_calibrate_tumvi_aprilgrid.py``),
but any known board geometry works. Here correspondences come straight from the
bundled test fixture (``test_config.json``): a 5x6 checkerboard, 30 pre-detected
corners in each of its two real photographed views.
"""
import json

import numpy as np

from ds_msp.calib import calibrate
from ds_msp.models import DoubleSphereModel


def main() -> None:
    cfg = json.load(open("test_config.json"))
    intr = cfg["intrinsics"]
    board = cfg["checkerboard"]
    rows, cols, sq = board["rows"], board["cols"], board["square_size"]

    # 3D board points (metres), row-major to match the bundled keypoint order.
    X = np.array([[c * sq, r * sq, 0.0] for r in range(rows) for c in range(cols)])
    visible = np.ones(len(X), dtype=bool)
    X_world = [X, X]                                                  # same board, 2 views
    keypoints = [np.array(t["keypoints_2d"]) for t in cfg["test_images"]]  # (30, 2) each
    visibility = [visible, visible]

    seed = DoubleSphereModel(fx=intr["fx"] * 0.9, fy=intr["fy"] * 0.9,
                             cx=intr["width"] / 2, cy=intr["height"] / 2,
                             xi=0.0, alpha=0.5)
    result = calibrate(seed, X_world, keypoints, visibility, loss="cauchy", f_scale=0.5)

    print(result["success"])           # -> True
    print(f"{result['rms_px']:.4f}")   # -> reprojection RMS, pixels
    print(result["model"])             # -> fitted DoubleSphereModel


if __name__ == "__main__":
    main()
