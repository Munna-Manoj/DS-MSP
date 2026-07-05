import pytest
import numpy as np
import cv2

from ds_msp.model import DoubleSphereCamera, ds_unproject
import ds_msp.cv as ds_camera_cv

# ============================================================================
# 1. Test Unprojection NaN Bug
# ============================================================================
def test_unproject_nan_bug():
    """Test that ds_unproject handles extreme inputs without returning nan with valid=True."""
    fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
    xi = 2.0      # xi > 1.0 triggers negative sqrt under specific points
    alpha = 0.5
    
    # A pixel near the border that causes a large r2 and negative sqrt argument
    points_2d = np.array([[640.0, 480.0]], dtype=np.float32)
    
    # Run unproject
    rays, valid = ds_unproject(points_2d, fx, fy, cx, cy, xi, alpha)
    
    # Asserts
    assert not np.isnan(rays).any(), "Rays contain NaN values!"
    assert not valid[0], "Point should be marked as INVALID!"
    assert (rays[0] == 0.0).all(), "Invalid ray should be masked with zero fallback!"

# ============================================================================
# 2. Test is_flip 1-Pixel Registration Shift
# ============================================================================
def test_is_flip_registration():
    """Test that is_flip=True maintains sub-pixel round-trip accuracy without 1-pixel shift."""
    fx, fy, cx, cy = 711.57, 711.24, 949.18, 518.81
    xi, alpha = 0.183, 0.809
    width, height = 1920, 1080
    
    # Unflipped camera
    cam_unflipped = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha, width, height, is_flip=False)
    # Flipped camera
    cam_flipped = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha, width, height, is_flip=True)
    
    # A point exactly on the optical axis
    pt_3d = np.array([[0.0, 0.0, 5.0]])
    
    p2d_unflipped, _ = cam_unflipped.project(pt_3d)
    p2d_flipped, _ = cam_flipped.project(pt_3d)
    
    # Flipped X coordinate should be EXACTLY (width - 1) - unflipped X coordinate
    expected_flipped_x = (width - 1.0) - p2d_unflipped[0, 0]
    assert np.isclose(p2d_flipped[0, 0], expected_flipped_x), (
        f"Flipped X coordinate shift: got {p2d_flipped[0, 0]}, expected {expected_flipped_x}"
    )
    
    # Sub-pixel round-trip unproject -> project check for flipped camera
    test_pixel = np.array([[450.5, 320.2]])
    ray, valid = cam_flipped.unproject(test_pixel)
    reprojected, _ = cam_flipped.project(ray)
    
    assert np.allclose(test_pixel, reprojected, atol=1e-10), "Flipped round trip failed to preserve sub-pixel accuracy!"

# ============================================================================
# 3. Test Redundant Normalization and Map Correctness
# ============================================================================
def test_undistortion_map_correctness_and_cache():
    """Test that get_undistortion_maps matches OpenCV outputs and caches correctly."""
    fx, fy, cx, cy = 711.57, 711.24, 949.18, 518.81
    xi, alpha = 0.183, 0.809
    width, height = 1920, 1080
    
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha, width, height)
    K_new = cam.compute_K_new(balance=0.5)
    
    # 1. Compare maps between OO and functional implementation
    mapx_oo, mapy_oo, _ = cam.get_undistortion_maps(K_new)
    mapx_cv, mapy_cv = ds_camera_cv.initUndistortRectifyMap(
        np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
        np.array([xi, alpha]),
        np.eye(3),
        K_new,
        (width, height),
        cv2.CV_32FC1
    )
    
    assert np.allclose(mapx_oo, mapx_cv, atol=1e-3), "mapx mismatch between OO and CV wrappers!"
    assert np.allclose(mapy_oo, mapy_cv, atol=1e-3), "mapy mismatch between OO and CV wrappers!"
    
    # 2. Test cache hits
    # Clear cache first
    cam._mapx = None
    cam._mapy = None
    
    # Run once to populate cache
    mapx1, mapy1, k_used1 = cam.get_undistortion_maps()
    # Modify cache manually to verify it is reused
    cam._mapx = np.zeros_like(mapx1)
    
    # Run again with same parameters
    mapx2, mapy2, k_used2 = cam.get_undistortion_maps()
    assert (mapx2 == 0.0).all(), "Cache was not reused when K_new is None!"
    
    # Run again by passing computed K_new explicitly
    mapx3, mapy3, k_used3 = cam.get_undistortion_maps(k_used1)
    assert (mapx3 == 0.0).all(), "Cache was not reused when matching K_new is passed!"

# ============================================================================
# 4. Test TI LDC Mesh LUT Generator and Point Undistorter
# ============================================================================
def test_ti_ldc_mesh_and_undistort():
    """Test the TI LDC Mesh LUT generation and point undistorter correctness."""
    from ds_msp.ldc import TI_LDC_MeshGenerator, TI_LDC_PointUndistorter
    
    fx, fy, cx, cy = 711.57, 711.24, 949.18, 518.81
    xi, alpha = 0.183, 0.809
    width, height = 1920, 1080
    
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha, width, height)
    mesh_gen = TI_LDC_MeshGenerator(cam)
    
    # Generate mesh
    res = mesh_gen.generate_mesh_and_intrinsics(width, height, downsample_factor=4, balance=0.5)
    
    mesh_lut = res["mesh_lut"]
    mesh_lut_float = res["mesh_lut_float"]
    K_new = res["K_new"]
    
    # 1. Assert Mesh properties
    # J7 LDC spacing step is 2^4 = 16 pixels
    step = 16
    padded_width = ((width + step - 1) // step) * step
    padded_height = ((height + step - 1) // step) * step
    expected_mesh_width = (padded_width // step) + 1
    expected_mesh_height = (padded_height // step) + 1
    assert mesh_lut.shape == (expected_mesh_height, expected_mesh_width, 2), "Incorrect Mesh shape!"
    
    # Int mesh_lut displacement must be Q3 format (float * 8)
    assert np.allclose(mesh_lut, np.round(mesh_lut_float * 8.0).astype(np.int16)), "Mesh integer LUT is not in J7 Q3 format!"
    
    # 2. Point Undistorter sub-pixel accuracy verification
    undistorter = TI_LDC_PointUndistorter(mesh_lut_float, K_new, downsample_factor=4, output_width=width, output_height=height)
    
    # Sample distorted keypoints
    pts_dist = np.array([[800.0, 600.0], [1200.0, 450.0]], dtype=np.float32)
    
    pts_undist_ldc, valid_ldc = undistorter.undistort_points(pts_dist)
    pts_undist_gt, valid_gt = cam.undistort_points(pts_dist, K_new)
    
    # Bilinear interpolation error on 16px grid should be < 0.2 px for typical lens coordinates
    assert np.allclose(pts_undist_ldc[valid_ldc], pts_undist_gt[valid_ldc], atol=0.2), (
        f"LDC point undistortion differs too much from analytical: LDC={pts_undist_ldc}, GT={pts_undist_gt}"
    )

# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-INTEROP-002")
