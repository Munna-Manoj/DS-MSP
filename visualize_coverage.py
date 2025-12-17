
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

def main():
    # 1. Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_config.json')
    config = load_config(config_path)
    
    # Parameters
    intr = config['intrinsics']
    fx, fy = intr['fx'], intr['fy']
    cx, cy = intr['cx'], intr['cy']
    xi, alpha = intr['xi'], intr['alpha']
    w, h = intr['width'], intr['height']
    
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.array([xi, alpha])
    
    # Load image
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['image_file'])
    img = cv2.imread(img_path)
    
    # 2. Get K_new used for "whole" (balance=0.0)
    K_new_whole = ds_camera_cv.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), balance=0.0)
    
    # 3. Create grid of pixels in the UNDISTORTED image
    # These are the pixels that actually exist in the output
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    points_undist = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2).astype(np.float32)
    
    # 4. Distort these points back to the ORIGINAL image
    # Undistorted (u', v') -> Normalized (x, y) -> Distorted (u, v)
    
    # Inverse of Pinhole Projection (K_new)
    fx_new, fy_new = K_new_whole[0, 0], K_new_whole[1, 1]
    cx_new, cy_new = K_new_whole[0, 2], K_new_whole[1, 2]
    
    mx = (points_undist[:, 0] - cx_new) / fx_new
    my = (points_undist[:, 1] - cy_new) / fy_new
    
    # Create rays (z=1)
    rays = np.stack([mx, my, np.ones_like(mx)], axis=-1)
    # Normalize rays
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    
    # Project back to distorted image using DS model
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha, w, h)
    points_dist, valid = cam.project(rays)
    
    # 5. Visualize Coverage
    # Create a mask of covered pixels on the original image
    coverage_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Round to nearest pixel
    points_dist_int = np.round(points_dist).astype(int)
    
    # Filter out of bounds
    valid_bounds = (points_dist_int[:, 0] >= 0) & (points_dist_int[:, 0] < w) & \
                   (points_dist_int[:, 1] >= 0) & (points_dist_int[:, 1] < h)
    
    valid_points = points_dist_int[valid & valid_bounds]
    
    # Mark covered pixels
    coverage_mask[valid_points[:, 1], valid_points[:, 0]] = 255
    
    # Dilate slightly to fill gaps from sampling
    kernel = np.ones((3,3), np.uint8)
    coverage_mask = cv2.dilate(coverage_mask, kernel, iterations=2)
    
    # Create visualization
    # Original image
    vis_img = img.copy()
    
    # Darken the whole image
    vis_img = (vis_img * 0.3).astype(np.uint8)
    
    # Restore brightness where covered
    vis_img[coverage_mask > 0] = img[coverage_mask > 0]
    
    # Draw red border around the covered area
    contours, _ = cv2.findContours(coverage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)
    
    # Save
    out_path = 'part1_calibration/coverage_vis.jpg'
    cv2.imwrite(out_path, vis_img)
    print(f"Saved coverage visualization to {out_path}")

if __name__ == "__main__":
    main()
