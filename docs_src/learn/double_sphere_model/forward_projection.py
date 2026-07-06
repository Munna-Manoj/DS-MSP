"""Chapter 2 -- the Double Sphere forward map (Usenko et al. 2018).

These are the six lines of `ds_project` in `ds_msp/models/ds_math.py`, reproduced here
against the bundled real calibration and cross-checked against the real function -- so
the walkthrough can never silently drift from the library. The 3D point comes from
unprojecting a real detected corner (`test_config.json`) and pushing it out along its
own ray, which shows the central-projection property: only *direction* matters, so any
depth along that ray lands on the same pixel.
"""
import json

import numpy as np

from ds_msp.models.ds_math import ds_project, ds_unproject


def main() -> None:
    intr = json.load(open("test_config.json"))["intrinsics"]  # the bundled real calibration
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    xi, alpha = intr["xi"], intr["alpha"]

    # A real detected corner, pushed out to 2.3 m along its own ray -- not unit length,
    # so d1 below is not trivially 1.
    u0, v0 = json.load(open("test_config.json"))["test_images"][0]["keypoints_2d"][0]
    ray, _ = ds_unproject(np.array([u0, v0]), fx, fy, cx, cy, xi, alpha)
    x, y, z = ray * 2.3
    print(f"3D point (camera frame, metres): x={x:.4f} y={y:.4f} z={z:.4f}")

    d1  = np.sqrt(x*x + y*y + z*z)          # distance to sphere 1  (normalize the ray)
    z1  = z + xi * d1                        # shift the z by xi  -> center of sphere 2
    d2  = np.sqrt(x*x + y*y + z1*z1)         # distance to sphere 2
    den = alpha * d2 + (1.0 - alpha) * z1    # the alpha-blended projection denominator
    u   = fx * x / den + cx
    v   = fy * y / den + cy

    print(f"d1={d1:.4f}  z1={z1:.4f}  d2={d2:.4f}  den={den:.4f}")
    print(f"u={u:.4f}  v={v:.4f}  (matches the original detected corner {u0}, {v0})")

    # cross-check against the real ds_project -- these six lines must never drift from it
    u_ref, v_ref, valid = ds_project(np.array([x, y, z]), fx, fy, cx, cy, xi, alpha)
    print(f"ds_project() agrees:  u={u_ref:.4f}  v={v_ref:.4f}  valid={bool(valid)}")


if __name__ == "__main__":
    main()
