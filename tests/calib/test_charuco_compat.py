"""OpenCV-version normalization at the ChArUco detector boundary."""

import numpy as np
import pytest

from ds_msp.detect.charuco import _canonical_charuco_corners


def test_opencv_4_charuco_coordinates_are_already_canonical():
    corners = np.array([[10.25, 20.75]], dtype=np.float32)
    actual = _canonical_charuco_corners(corners, opencv_version="4.13.0")
    assert np.array_equal(actual, corners)


def test_opencv_5_0_half_pixel_shift_is_normalized_at_source():
    corners = np.array([[9.75, 20.25]], dtype=np.float32)
    actual = _canonical_charuco_corners(corners, opencv_version="5.0.0")
    assert np.array_equal(actual, np.array([[10.25, 20.75]], dtype=np.float32))


pytestmark = pytest.mark.req("FR-CALIB-004")
