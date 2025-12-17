
import numpy as np
import cv2
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ds_camera import DoubleSphereCamera
import ds_camera_cv

def test_project_points():
    print("Testing projectPoints...")
    fx, fy, cx, cy = 500, 500, 320, 240
    xi, alpha = 0.5, 0.6
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.array([xi, alpha])
    
    points_3d = np.array([
        [0, 0, 10],
        [10, 10, 10],
        [-5, 5, 20]
    ], dtype=np.float64)
    
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    
    # Test ds_camera_cv
    points_2d_cv, _ = ds_camera_cv.projectPoints(points_3d, rvec, tvec, K, D)
    
    # Test original class
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha, 640, 480)
    points_2d_orig, valid = cam.project(points_3d)
    
    diff = np.linalg.norm(points_2d_cv.squeeze() - points_2d_orig)
    print(f"Difference between wrapper and original: {diff}")
    assert diff < 1e-5, "projectPoints mismatch"
    print("projectPoints passed.")

def test_undistort_points():
    print("\nTesting undistortPoints...")
    fx, fy, cx, cy = 500, 500, 320, 240
    xi, alpha = 0.5, 0.6
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.array([xi, alpha])
    
    # Create some distorted points
    points_distorted = np.array([
        [320, 240],
        [400, 300],
        [100, 100]
    ], dtype=np.float64)
    
    # Undistort using wrapper
    points_undist_cv = ds_camera_cv.undistortPoints(points_distorted, K, D)
    
    # Undistort using original class
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha, 640, 480)
    rays, valid = cam.unproject(points_distorted)
    rays_norm = rays / (rays[:, 2:3] + 1e-10)
    points_undist_orig = rays_norm[:, :2]
    
    diff = np.linalg.norm(points_undist_cv.squeeze() - points_undist_orig)
    print(f"Difference between wrapper and original: {diff}")
    assert diff < 1e-5, "undistortPoints mismatch"
    print("undistortPoints passed.")

def test_round_trip():
    print("\nTesting round trip (distort -> undistort)...")
    fx, fy, cx, cy = 500, 500, 320, 240
    xi, alpha = 0.5, 0.6
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.array([xi, alpha])
    
    # Normalized points (z=1)
    points_norm = np.array([
        [0, 0],
        [0.5, 0.5],
        [-0.2, 0.3]
    ], dtype=np.float64)
    
    # Distort
    points_distorted = ds_camera_cv.distortPoints(points_norm, K, D)
    
    # Undistort
    points_undistorted = ds_camera_cv.undistortPoints(points_distorted, K, D)
    
    diff = np.linalg.norm(points_norm - points_undistorted.squeeze())
    print(f"Round trip error: {diff}")
    assert diff < 1e-5, "Round trip failed"
    print("Round trip passed.")

def test_undistort_image():
    print("\nTesting undistortImage...")
    fx, fy, cx, cy = 500, 500, 320, 240
    xi, alpha = 0.5, 0.6
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.array([xi, alpha])
    
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 10, (255, 255, 255), -1)
    
    Knew = ds_camera_cv.estimateNewCameraMatrixForUndistortRectify(K, D, (640, 480), balance=0.5)
    img_undist = ds_camera_cv.undistortImage(img, K, D, Knew)
    
    print(f"Undistorted image shape: {img_undist.shape}")
    assert img_undist.shape == img.shape, "Shape mismatch"
    print("undistortImage passed.")

if __name__ == "__main__":
    test_project_points()
    test_undistort_points()
    test_round_trip()
    test_undistort_image()
    print("\nAll tests passed!")
