
import json
import cv2
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ds_camera_cv

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def draw_reprojection(img, points_2d, points_reproj, axis_reproj=None):
    img_draw = img.copy()
    
    # Draw original points (Green)
    for pt in points_2d:
        cv2.circle(img_draw, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
        
    # Draw reprojected points (Red)
    for pt in points_reproj:
        cv2.circle(img_draw, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        
    # Draw axes if provided
    if axis_reproj is not None:
        origin = tuple(axis_reproj[0].astype(int))
        cv2.arrowedLine(img_draw, origin, tuple(axis_reproj[1].astype(int)), (0, 0, 255), 3) # X - Red
        cv2.arrowedLine(img_draw, origin, tuple(axis_reproj[2].astype(int)), (0, 255, 0), 3) # Y - Green
        cv2.arrowedLine(img_draw, origin, tuple(axis_reproj[3].astype(int)), (255, 0, 0), 3) # Z - Blue
        
    return img_draw

def main():
    # 1. Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_config.json')
    config = load_config(config_path)
    
    # Parameters
    intr = config['intrinsics']
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]])
    D = np.array([intr['xi'], intr['alpha']])
    w, h = intr['width'], intr['height']
    
    # Load image
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['image_file'])
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return

    # 3D Points
    rows = config['checkerboard']['rows']
    cols = config['checkerboard']['cols']
    square_size = config['checkerboard']['square_size']
    points_3d = []
    for i in range(rows):
        for j in range(cols):
            points_3d.append([j * square_size, i * square_size, 0.0])
    points_3d = np.array(points_3d, dtype=np.float64)
    
    # 2D Points
    points_2d = np.array(config['keypoints_2d'], dtype=np.float64)
    
    # Axes for visualization
    axis_length = 0.4
    axis_points = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ], dtype=np.float64)

    print("--- Task 1: Pose on Distorted Image ---")
    
    # Undistort points to normalized coordinates for PnP
    points_2d_norm = ds_camera_cv.undistortPoints(points_2d, K, D)
    
    # Ensure contiguous arrays
    points_3d = np.ascontiguousarray(points_3d).reshape(-1, 3)
    points_2d_norm = np.ascontiguousarray(points_2d_norm).reshape(-1, 2)
    
    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(points_3d, points_2d_norm, np.eye(3), None)
    
    if not success:
        print("PnP failed")
        return
        
    print(f"rvec: {rvec.flatten()}")
    print(f"tvec: {tvec.flatten()}")
    
    # Reproject
    points_reproj, _ = ds_camera_cv.projectPoints(points_3d, rvec, tvec, K, D)
    points_reproj = points_reproj.reshape(-1, 2)
    
    axis_reproj, _ = ds_camera_cv.projectPoints(axis_points, rvec, tvec, K, D)
    axis_reproj = axis_reproj.reshape(-1, 2)
    
    # Visualize
    img_dist_vis = draw_reprojection(img, points_2d, points_reproj, axis_reproj)
    cv2.imwrite('part1_calibration/result_distorted.jpg', img_dist_vis)
    print("Saved result_distorted.jpg")
    
    
    print("\n--- Task 2: Undistort (Optimal Crop) ---")
    
    # Estimate new camera matrix (balance=0.5 for optimal crop usually, or 0 for full FOV, 1 for crop)
    # User asked for "optimal crop" and "keep whole pixel".
    # Usually balance=0 is "keep all pixels" (but might have black borders), balance=1 is "crop to valid pixels".
    # Let's assume:
    # 1. Optimal Crop (balance=1.0 - no black borders, but loses some FOV)
    # 2. Keep Whole Pixel (balance=0.0 - all pixels visible, black borders)
    
    # Case 1: Optimal Crop (balance=1.0)
    K_new_crop = ds_camera_cv.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), balance=1.0)
    
    # Undistort image
    img_undist_crop = ds_camera_cv.undistortImage(img, K, D, K_new_crop)
    
    # Project points to this new image
    # We need to project 3D points using K_new_crop (pinhole model)
    # Since it's pinhole, we can use cv2.projectPoints with 0 distortion
    points_reproj_crop, _ = cv2.projectPoints(points_3d, rvec, tvec, K_new_crop, np.zeros(4))
    points_reproj_crop = points_reproj_crop.reshape(-1, 2)
    
    axis_reproj_crop, _ = cv2.projectPoints(axis_points, rvec, tvec, K_new_crop, np.zeros(4))
    axis_reproj_crop = axis_reproj_crop.reshape(-1, 2)
    
    # We also need to transform the original 2D observations to this new image to verify
    # Distorted (u,v) -> Normalized (x,y) -> Undistorted (u', v')
    points_2d_undist_crop = ds_camera_cv.undistortPoints(points_2d, K, D, P=K_new_crop)
    points_2d_undist_crop = points_2d_undist_crop.reshape(-1, 2)
    
    img_crop_vis = draw_reprojection(img_undist_crop, points_2d_undist_crop, points_reproj_crop, axis_reproj_crop)
    cv2.imwrite('part1_calibration/result_undistort_crop.jpg', img_crop_vis)
    print("Saved result_undistort_crop.jpg")
    
    
    print("\n--- Task 3: Undistort (Keep Whole Pixel) ---")
    
    # Case 2: Keep Whole Pixel (balance=0.0)
    K_new_whole = ds_camera_cv.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), balance=0.0)
    
    # Undistort image
    img_undist_whole = ds_camera_cv.undistortImage(img, K, D, K_new_whole)
    
    # Project points
    points_reproj_whole, _ = cv2.projectPoints(points_3d, rvec, tvec, K_new_whole, np.zeros(4))
    points_reproj_whole = points_reproj_whole.reshape(-1, 2)
    
    axis_reproj_whole, _ = cv2.projectPoints(axis_points, rvec, tvec, K_new_whole, np.zeros(4))
    axis_reproj_whole = axis_reproj_whole.reshape(-1, 2)
    
    # Transform observed points
    points_2d_undist_whole = ds_camera_cv.undistortPoints(points_2d, K, D, P=K_new_whole)
    points_2d_undist_whole = points_2d_undist_whole.reshape(-1, 2)
    
    img_whole_vis = draw_reprojection(img_undist_whole, points_2d_undist_whole, points_reproj_whole, axis_reproj_whole)
    cv2.imwrite('part1_calibration/result_undistort_whole.jpg', img_whole_vis)
    print("Saved result_undistort_whole.jpg")

    print("\n--- Task 4: Undistort (Zoom Out 4x) ---")
    
    # Case 3: Zoom Out (Reduce focal length by 4x)
    K_new_zoom = K_new_whole.copy()
    K_new_zoom[0, 0] /= 4.0
    K_new_zoom[1, 1] /= 4.0
    K_new_zoom[0, 2] = w / 2.0 # Center principal point
    K_new_zoom[1, 2] = h / 2.0
    
    # Undistort image
    img_undist_zoom = ds_camera_cv.undistortImage(img, K, D, K_new_zoom)
    
    # Project points
    points_reproj_zoom, _ = cv2.projectPoints(points_3d, rvec, tvec, K_new_zoom, np.zeros(4))
    points_reproj_zoom = points_reproj_zoom.reshape(-1, 2)
    
    axis_reproj_zoom, _ = cv2.projectPoints(axis_points, rvec, tvec, K_new_zoom, np.zeros(4))
    axis_reproj_zoom = axis_reproj_zoom.reshape(-1, 2)
    
    # Transform observed points
    points_2d_undist_zoom = ds_camera_cv.undistortPoints(points_2d, K, D, P=K_new_zoom)
    points_2d_undist_zoom = points_2d_undist_zoom.reshape(-1, 2)
    
    img_zoom_vis = draw_reprojection(img_undist_zoom, points_2d_undist_zoom, points_reproj_zoom, axis_reproj_zoom)
    cv2.imwrite('part1_calibration/result_undistort_zoom.jpg', img_zoom_vis)
    print("Saved result_undistort_zoom.jpg")

if __name__ == "__main__":
    main()
