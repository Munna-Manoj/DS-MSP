"""The object API: ``DoubleSphereCamera.undistort_image`` does the same job as the three
OpenCV-style calls, in one call, and hands back the ``K_new`` it chose.

Called with ``K_new=None`` it builds a balanced matrix at ``balance=0.5``. There is no
``balance=`` keyword on the method itself -- to pick a different balance, build the matrix
explicitly with ``estimateNewCameraMatrixForUndistortRectify(..., balance=...)`` and pass
it in as ``K_new=``.
"""
import cv2

from ds_msp import DoubleSphereCamera
import ds_msp.cv as ds_cv


def main() -> None:
    cam = DoubleSphereCamera(fx=711.57, fy=711.24, cx=949.18, cy=518.81,
                             xi=0.183, alpha=0.809, width=1920, height=1080)
    img = cv2.imread("assets/test_image.jpg")

    # K_new=None -> built internally at balance=0.5
    img_undist, K_new = cam.undistort_image(img)
    print(img_undist.shape)          # -> (1080, 1920, 3)
    print(round(K_new[0, 0], 2))     # -> 426.84  (between the widest and tightest focals)

    # cam.undistort_image(img, balance=0.3) raises TypeError -- no such keyword on the method
    try:
        cam.undistort_image(img, balance=0.3)   # type: ignore[call-arg]
    except TypeError as exc:
        print(f"TypeError: {exc}")

    # Pick a different balance by building the matrix explicitly and passing it as K_new=.
    K_tight = ds_cv.estimateNewCameraMatrixForUndistortRectify(
        cam.K, cam.D, (1920, 1080), balance=1.0)
    img_tight, _ = cam.undistort_image(img, K_new=K_tight)
    print(round(K_tight[0, 0], 2))   # -> 569.12  (tightest crop, no borders)


if __name__ == "__main__":
    main()
