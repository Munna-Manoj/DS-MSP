"""Measure the `balance` FOV-vs-border trade-off instead of taking it on faith.

`balance` slides between two extremes of the same rectified image: `0.0` keeps the widest
field of view at the cost of black corners, `1.0` crops in until the borders are gone. This
script builds all three reference points (`0.0`, `0.5`, `1.0`) and measures the
black-border fraction of each -- the share of output pixels that fell outside the fisheye's
coverage and were filled with black.
"""
import cv2
import numpy as np

from ds_msp import DoubleSphereCamera
import ds_msp.cv as ds_cv


def black_fraction(im: np.ndarray) -> float:
    """Fraction of pixels that are pure black (outside the fisheye coverage)."""
    return float(np.mean(np.all(im == 0, axis=2)))


def main() -> None:
    cam = DoubleSphereCamera(fx=711.57, fy=711.24, cx=949.18, cy=518.81,
                             xi=0.183, alpha=0.809, width=1920, height=1080)
    img = cv2.imread("assets/test_image.jpg")

    focals = {}
    for balance in (0.0, 0.5, 1.0):
        K_new = ds_cv.estimateNewCameraMatrixForUndistortRectify(
            cam.K, cam.D, (1920, 1080), balance=balance)
        und = ds_cv.undistortImage(img, cam.K, cam.D, Knew=K_new)
        fx_new = K_new[0, 0]
        frac = black_fraction(und)
        focals[balance] = fx_new
        print(f"balance={balance:.1f}  fx_new={fx_new:.2f} px  black_fraction={frac:.3f}")

    # Focal length exactly doubles from balance=0.0 to balance=1.0.
    print(f"fx_new(1.0) / fx_new(0.0) = {focals[1.0] / focals[0.0]:.2f}")
    # balance=0.5 lands exactly at the midpoint of the two extremes.
    midpoint = (focals[0.0] + focals[1.0]) / 2
    print(f"midpoint(0.0, 1.0) = {midpoint:.2f}  (balance=0.5 gives {focals[0.5]:.2f})")


if __name__ == "__main__":
    main()
