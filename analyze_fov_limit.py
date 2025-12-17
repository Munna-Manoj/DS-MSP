
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
    
    # Load image
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['image_file'])
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return

    # 2. Create grid of all pixels
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    points_2d = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2).astype(np.float32)
    
    # 3. Unproject all pixels to 3D rays
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha, w, h)
    rays, valid_mask = cam.unproject(points_2d)
    
    # 4. Analyze rays
    # Condition 1: Model Validity (already in valid_mask)
    # Condition 2: Pinhole Validity (z > 0)
    
    z_coords = rays[:, 2]
    pinhole_valid_mask = z_coords > 0
    
    # Combined mask: Valid in model AND Valid for pinhole projection
    final_valid_mask = valid_mask & pinhole_valid_mask
    
    # Reshape to image dimensions
    valid_mask_img = valid_mask.reshape(h, w)
    pinhole_valid_mask_img = pinhole_valid_mask.reshape(h, w)
    final_valid_mask_img = final_valid_mask.reshape(h, w)
    
    # 5. Visualization
    # Create an overlay
    # Green: Valid and Projectable (z > 0)
    # Red: Valid but NOT Projectable (z <= 0) - These are the "missing" pixels
    # Blue: Invalid in model (s < 0)
    
    vis_img = img.copy()
    
    # Create color masks
    # We'll blend them:
    # - Pixels that are valid model but z <= 0 -> Red tint
    # - Pixels that are invalid model -> Blue tint
    # - Pixels that are valid and z > 0 -> No tint (original image)
    
    # Mask for Red (Valid Model, z <= 0)
    mask_red = valid_mask_img & (~pinhole_valid_mask_img)
    
    # Mask for Blue (Invalid Model)
    mask_blue = ~valid_mask_img
    
    # Apply tints
    overlay = vis_img.copy()
    overlay[mask_red] = [0, 0, 255] # Red
    overlay[mask_blue] = [255, 0, 0] # Blue
    
    # Blend
    alpha_blend = 0.5
    cv2.addWeighted(overlay, alpha_blend, vis_img, 1 - alpha_blend, 0, vis_img)
    
    # Save result
    out_path = 'part1_calibration/fov_analysis.jpg'
    cv2.imwrite(out_path, vis_img)
    print(f"Saved FOV analysis to {out_path}")
    
    # Statistics
    total_pixels = w * h
    valid_model_pixels = np.sum(valid_mask)
    projectable_pixels = np.sum(final_valid_mask)
    missing_pixels = valid_model_pixels - projectable_pixels
    
    print(f"Total Pixels: {total_pixels}")
    print(f"Valid Model Pixels: {valid_model_pixels} ({valid_model_pixels/total_pixels*100:.1f}%)")
    print(f"Projectable (z>0) Pixels: {projectable_pixels} ({projectable_pixels/total_pixels*100:.1f}%)")
    print(f"Missing Pixels (Valid but z<=0): {missing_pixels} ({missing_pixels/total_pixels*100:.1f}%)")

if __name__ == "__main__":
    main()
