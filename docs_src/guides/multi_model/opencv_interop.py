"""Plug a converted model's `K`/`distortion` straight into OpenCV.

`convert` first (see the how-to guide), then `cam.K` and `cam.distortion` are exactly
what `cv2.fisheye.*` (Kannala-Brandt) or `cv2.projectPoints`/`cv2.solvePnP` (RadTan)
expect -- no reformatting.
"""
import cv2
import numpy as np

from ds_msp import DoubleSphereModel, KannalaBrandtModel, RadTanModel, convert, solve_pnp


def main() -> None:
    W, H = 1920, 1080
    cam = DoubleSphereModel.sample()
    img = cv2.imread("assets/test_image.jpg")

    # -> cv2.fisheye: convert to Kannala-Brandt, then use OpenCV's own undistort call.
    kb, _ = convert(cam, KannalaBrandtModel, width=W, height=H)
    K_new = kb.K.copy()
    K_new[0, 0] *= 0.6                                     # widen the output FOV a bit
    K_new[1, 1] *= 0.6
    img_rect = cv2.fisheye.undistortImage(img, kb.K, kb.distortion, Knew=K_new)
    print(img_rect.shape)                                  # -> (1080, 1920, 3)

    # -> cv2 pinhole: convert to RadTan (bounded to a representable FOV), reuse a PnP pose.
    object_points = np.array([[0, 0, 0], [0.1, 0, 0],
                              [0, 0.1, 0], [0.1, 0.1, 0]], dtype=float)
    image_points = np.array([[610, 480], [720, 470],
                             [600, 590], [715, 580]], dtype=float)
    ok, rvec, tvec = solve_pnp(cam, object_points, image_points)

    rt, _ = convert(cam, RadTanModel, width=W, height=H, max_fov_deg=120)
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, rt.K, rt.distortion)
    print(proj.reshape(-1, 2).round(2).tolist())


if __name__ == "__main__":
    main()
