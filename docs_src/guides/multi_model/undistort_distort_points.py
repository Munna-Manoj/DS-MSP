"""Move keypoints between the distorted (fisheye) frame and a rectified pinhole frame.

`Undistorter.undistort_points` moves detections into a pinhole frame for classic
algorithms; `distort_points` is its exact inverse, for drawing pinhole-space
results back onto the original fisheye image. Both round-trip to sub-pixel.
"""
import numpy as np

from ds_msp import DoubleSphereModel, Undistorter


def main() -> None:
    W, H = 1920, 1080
    cam = DoubleSphereModel.sample()
    und = Undistorter(cam, width=W, height=H)
    K_new = und.new_K(balance=0.5)

    distorted_kpts = np.array([[640.0, 480.0], [900.0, 300.0]])   # e.g. detected features (N, 2)

    # distorted pixels  ->  rectified pinhole pixels (in the K_new frame)
    kp_rect, valid = und.undistort_points(distorted_kpts, K_new)
    print(kp_rect.round(3).tolist())
    print(valid.tolist())

    # rectified pinhole pixels  ->  distorted pixels (exact inverse)
    kp_dist, valid = und.distort_points(kp_rect, K_new)
    print(kp_dist.round(3).tolist())

    max_err = float(np.max(np.abs(kp_dist - distorted_kpts)))
    print(f"round-trip max error: {max_err:.2e} px")


if __name__ == "__main__":
    main()
