import argparse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from ds_msp.model import DoubleSphereCamera, ds_project, ds_unproject

def viz_fov_zones(args):
    """Visualize FOV zones (Green/Yellow/Red) on an image."""
    print(f"Visualizing FOV zones for {args.image}...")
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read {args.image}")
        return

    # Load camera (hardcoded for demo or load from args)
    # Using Sample 96 params for demo
    fx, fy = 711.57, 711.24
    cx, cy = 949.18, 518.81
    xi, alpha = 0.183, 0.809
    
    h, w = img.shape[:2]
    
    # Create grid
    y, x = np.mgrid[0:h, 0:w]
    
    # Unproject all pixels
    u = x.flatten()
    v = y.flatten()
    
    # Custom unprojection to get raw values for analysis
    mx = (u - cx) / fx
    my = (v - cy) / fy
    r2 = mx*mx + my*my
    
    # Validity check (s >= 0)
    s = 1.0 - (2.0 * alpha - 1.0) * r2
    valid_mask = s >= 0
    
    # Calculate theta for valid points
    # ray_z = (k * mz - xi) / norm
    # theta = acos(ray_z)
    
    # For visualization, we just color based on validity and theta
    # Green: Valid & Frontal (theta < 90)
    # Yellow: Valid & Back (theta >= 90)
    # Red: Invalid (s < 0)
    
    vis_img = img.copy()
    
    # Red overlay for invalid
    vis_img[~valid_mask.reshape(h, w)] = vis_img[~valid_mask.reshape(h, w)] * 0.5 + np.array([0, 0, 128]) * 0.5
    
    # We need full unprojection for Yellow/Green distinction
    rays, valid = ds_unproject(np.stack([u, v], axis=-1), fx, fy, cx, cy, xi, alpha)
    
    # Check Z component of ray
    # If Z > 0 (Frontal) -> Green (Leave as is or tint green)
    # If Z <= 0 (Back) -> Yellow
    
    z_vals = rays[:, 2]
    is_back = (z_vals <= 0) & valid
    
    # Apply Yellow tint to back-facing pixels
    mask_back = is_back.reshape(h, w)
    vis_img[mask_back] = vis_img[mask_back] * 0.7 + np.array([0, 255, 255]) * 0.3
    
    # Apply Green tint to frontal pixels (optional, to make it clear)
    mask_front = (z_vals > 0) & valid
    mask_front = mask_front.reshape(h, w)
    # vis_img[mask_front] = vis_img[mask_front] * 0.9 + np.array([0, 255, 0]) * 0.1

    out_path = "results/visualizations/fov_zones.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis_img)
    print(f"Saved to {out_path}")

def viz_undistort(args):
    """Demonstrate undistortion modes."""
    print(f"Undistorting {args.image}...")
    img = cv2.imread(args.image)
    if img is None: return
    
    h, w = img.shape[:2]
    # Sample 11 params
    fx, fy, cx, cy = 711.57, 711.24, 949.18, 518.81
    xi, alpha = 0.183, 0.809
    
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha, w, h)
    
    # Modes
    modes = [
        ("whole", 0.0),
        ("balanced", 0.5),
        ("crop", 1.0)
    ]
    
    out_dir = "results/visualizations"
    os.makedirs(out_dir, exist_ok=True)
    
    for name, balance in modes:
        K_new = cam.compute_K_new(balance=balance)
        img_undist, _ = cam.undistort_image(img, K_new)
        cv2.imwrite(f"{out_dir}/undistort_{name}.jpg", img_undist)
        print(f"Saved {name} to {out_dir}/undistort_{name}.jpg")

def main():
    parser = argparse.ArgumentParser(description="DS-MSP Visualization Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # FOV Zones
    parser_fov = subparsers.add_parser("fov", help="Visualize FOV zones")
    parser_fov.add_argument("--image", type=str, default="test_image_96.jpg", help="Input image")
    parser_fov.set_defaults(func=viz_fov_zones)
    
    # Undistort Demo
    parser_undist = subparsers.add_parser("undistort", help="Run undistortion demo")
    parser_undist.add_argument("--image", type=str, default="test_image.jpg", help="Input image")
    parser_undist.set_defaults(func=viz_undistort)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
