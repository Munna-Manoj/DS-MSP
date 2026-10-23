"""OpenCV-version normalization at the ChArUco detector boundary (pixel-centre convention)."""

import cv2
import numpy as np
import pytest

from ds_msp.detect.charuco import _canonical_charuco_corners


def test_opencv_4_charuco_half_pixel_offset_is_removed():
    corners = np.array([[10.25, 20.75]], dtype=np.float32)
    actual = _canonical_charuco_corners(corners, opencv_version="4.13.0")
    assert np.array_equal(actual, np.array([[9.75, 20.25]], dtype=np.float32))


def test_opencv_5_0_output_is_already_pixel_centred():
    corners = np.array([[9.75, 20.25]], dtype=np.float32)
    actual = _canonical_charuco_corners(corners, opencv_version="5.0.0")
    assert np.array_equal(actual, corners)


def test_unmeasured_opencv_version_is_refused():
    with pytest.raises(ValueError, match="not measured"):
        _canonical_charuco_corners(np.zeros((1, 2)), opencv_version="5.1.0")


def test_detected_charuco_corners_match_rendered_truth():
    """Render a ChArUco board at exact integer-pixel squares (supersampled) and require the
    detected corners to sit on the pixel-centre truth, with no half-pixel bias."""
    from ds_msp.detect.charuco import BoardSpec, detect_image, make_detectors
    nx, ny, sq, margin, ss = 9, 7, 64, 32, 4
    det = make_detectors([BoardSpec(nx, ny, 0.04, 0.03, 0.04)])
    w, h = nx * sq + 2 * margin, ny * sq + 2 * margin
    big = det[0].getBoard().generateImage((w * ss, h * ss), marginSize=margin * ss, borderBits=1)
    img = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    gx, gy = np.meshgrid(np.arange(1, nx), np.arange(1, ny))
    truth = np.c_[margin + gx.ravel() * sq - 0.5, margin + gy.ravel() * sq - 0.5]
    (_b, _ids, uv), = detect_image(det, img)
    err = uv - truth[np.linalg.norm(uv[:, None] - truth[None], axis=2).argmin(1)]
    assert np.all(np.abs(err.mean(0)) < 0.05)
    assert np.sqrt((err ** 2).sum(1).mean()) < 0.1


pytestmark = pytest.mark.req("FR-CALIB-004")
