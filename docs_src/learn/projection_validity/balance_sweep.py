"""Chapter 3 -- the `balance` knob trades rectified field of view for black border.

A pinhole image plane is infinite at 90 degrees, so a Double Sphere lens's >180 deg
field of view can never fit into one rectified frame without cropping. `balance`
controls where on that trade-off the output sits.
"""

import json

import cv2
import numpy as np

from ds_msp import DoubleSphereCamera


def main() -> None:
    intr = json.load(open("test_config.json"))["intrinsics"]  # the bundled real calibration
    cam = DoubleSphereCamera(intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                             intr["xi"], intr["alpha"],
                             width=intr["width"], height=intr["height"])

    img = cv2.imread("assets/test_image.jpg")   # the bundled real fisheye frame
    for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
        K_new = cam.compute_K_new(balance=b)
        rect, _ = cam.undistort_image(img, K_new)
        hfov = np.degrees(2 * np.arctan((cam.width / 2) / K_new[0, 0]))
        filled = float((cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY) > 0).mean()) * 100
        print(f"balance={b:.2f}  hfov={hfov:.1f} deg  filled={filled:.1f}%")


if __name__ == "__main__":
    main()
