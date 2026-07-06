"""Undistort a fisheye frame to a pinhole view, for any model, via `Undistorter`.

`Undistorter` depends only on the `CameraModel` contract (`project`), so it works
for any of DS-MSP's models -- not just Double Sphere. It caches the resampling map
internally, keyed by the target `K_new`.
"""
import cv2

from ds_msp import DoubleSphereModel, Undistorter


def main() -> None:
    W, H = 1920, 1080
    cam = DoubleSphereModel.sample()
    img = cv2.imread("assets/test_image.jpg")           # bundled fisheye frame

    und = Undistorter(cam, width=W, height=H)            # stateful map cache lives here
    K_new = und.new_K(balance=0.5)                        # 0.0 widest FOV ... 1.0 tightest crop
    img_rect, K_new = und.undistort_image(img, K_new)     # cv2.remap under the hood

    print(img_rect.shape)                # -> (1080, 1920, 3)
    print(round(float(K_new[0, 0]), 2))  # new focal length, px


if __name__ == "__main__":
    main()
