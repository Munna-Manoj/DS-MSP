"""Undistort a fisheye frame in three calls: build the camera, estimate a new pinhole
matrix, remap.

``ds_msp.cv`` mirrors ``cv2.fisheye``'s function signatures (``estimateNewCameraMatrixFor
UndistortRectify``, ``undistortImage``), so it drops into existing OpenCV pipelines. Uses
the bundled real fisheye frame (``assets/test_image.jpg``) and a calibrated Double Sphere
camera.
"""
import os
import tempfile

import cv2

from ds_msp import DoubleSphereCamera
import ds_msp.cv as ds_cv


def main() -> None:
    # A calibrated Double Sphere camera (1920x1080 fisheye).
    cam = DoubleSphereCamera(fx=711.57, fy=711.24, cx=949.18, cy=518.81,
                             xi=0.183, alpha=0.809, width=1920, height=1080)

    img = cv2.imread("assets/test_image.jpg")     # (1080, 1920, 3) BGR
    K, D = cam.K, cam.D                            # K: (3,3); D = [xi, alpha]
    print(f"D = {D.tolist()}")

    # balance=0.0 -> widest FOV (keeps the most scene; leaves black borders)
    K_new = ds_cv.estimateNewCameraMatrixForUndistortRectify(K, D, (1920, 1080), balance=0.0)
    img_undist = ds_cv.undistortImage(img, K, D, Knew=K_new)   # (1080, 1920, 3)

    out_path = os.path.join(tempfile.gettempdir(), "undistorted.jpg")
    cv2.imwrite(out_path, img_undist)
    print(img_undist.shape)          # -> (1080, 1920, 3)
    print(round(K_new[0, 0], 2))     # -> 284.56  (new focal length, px)


if __name__ == "__main__":
    main()
