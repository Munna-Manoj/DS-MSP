
import json
import cv2
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ds_camera_cv
from ds_camera import DoubleSphereCamera

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def visualize_coverage(img, K, D, K_new, w, h, name):
    # Create grid of pixels in the UNDISTORTED image
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    points_undist = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2).astype(np.float32)
    
    fx_new, fy_new = K_new[0, 0], K_new[1, 1]
    cx_new, cy_new = K_new[0, 2], K_new[1, 2]
    
    mx = (points_undist[:, 0] - cx_new) / fx_new
    my = (points_undist[:, 1] - cy_new) / fy_new
    
    rays = np.stack([mx, my, np.ones_like(mx)], axis=-1)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    
    cam = DoubleSphereCamera(K[0,0], K[1,1], K[0,2], K[1,2], D[0], D[1], w, h)
    points_dist, valid = cam.project(rays)
    
    coverage_mask = np.zeros((h, w), dtype=np.uint8)
    points_dist_int = np.round(points_dist).astype(int)
    valid_bounds = (points_dist_int[:, 0] >= 0) & (points_dist_int[:, 0] < w) & \
                   (points_dist_int[:, 1] >= 0) & (points_dist_int[:, 1] < h)
    valid_points = points_dist_int[valid & valid_bounds]
    coverage_mask[valid_points[:, 1], valid_points[:, 0]] = 255
    kernel = np.ones((3,3), np.uint8)
    coverage_mask = cv2.dilate(coverage_mask, kernel, iterations=2)
    
    vis_img = img.copy()
    vis_img = (vis_img * 0.3).astype(np.uint8)
    vis_img[coverage_mask > 0] = img[coverage_mask > 0]
    contours, _ = cv2.findContours(coverage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)
    
    cv2.imwrite(f'part1_calibration/coverage_{name}.jpg', vis_img)
    print(f"Saved coverage_{name}.jpg")

def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_config.json')
    config = load_config(config_path)
    
    intr = config['intrinsics']
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]])
    D = np.array([intr['xi'], intr['alpha']])
    w, h = intr['width'], intr['height']
    
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['image_file'])
    img = cv2.imread(img_path)
    
    # 1. Standard "Whole" (balance=0.0)
    K_new_whole = ds_camera_cv.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), balance=0.0)
    visualize_coverage(img, K, D, K_new_whole, w, h, "normal")
    
    # 2. Extreme Zoom Out (Reduce focal length by 4x)
    K_new_zoom = K_new_whole.copy()
    K_new_zoom[0, 0] /= 4.0
    K_new_zoom[1, 1] /= 4.0
    
    # Generate undistorted image for zoom
    img_undist_zoom = ds_camera_cv.undistortImage(img, K, D, K_new_zoom)
    cv2.imwrite('part1_calibration/result_undistort_zoom.jpg', img_undist_zoom)
    print("Saved result_undistort_zoom.jpg")
    
    visualize_coverage(img, K, D, K_new_zoom, w, h, "zoom")

if __name__ == "__main__":
    main()
