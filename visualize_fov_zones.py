
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ds_camera import DoubleSphereCamera

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_config.json')
    config = load_config(config_path)
    
    intr = config['intrinsics']
    xi = intr['xi']
    alpha = intr['alpha']
    w, h = intr['width'], intr['height']
    
    cam = DoubleSphereCamera(intr['fx'], intr['fy'], intr['cx'], intr['cy'], xi, alpha, w, h)
    
    # Load Image
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['image_file'])
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Compute Ray Angles for every pixel
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    points_2d = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2).astype(np.float32)
    
    rays, valid = cam.unproject(points_2d)
    
    # Angle with optical axis (Z-axis)
    # rays are unit vectors (x, y, z)
    # cos(theta) = z
    z = rays[:, 2]
    theta_rad = np.arccos(np.clip(z, -1.0, 1.0))
    theta_deg = np.degrees(theta_rad)
    
    theta_map = theta_deg.reshape(h, w)
    valid_map = valid.reshape(h, w)
    
    # 2. Define Zones
    # Zone 1: Frontal (0 to 90 deg) -> Green
    # Zone 2: Back/Side (90 to Limit) -> Yellow
    # Zone 3: Invalid -> Red
    
    # Create an overlay image
    overlay = np.zeros_like(img_rgb)
    
    # Green Zone (Theta < 90)
    mask_green = (theta_map < 90) & valid_map
    overlay[mask_green] = [0, 255, 0]
    
    # Yellow Zone (Theta >= 90 and Valid)
    mask_yellow = (theta_map >= 90) & valid_map
    overlay[mask_yellow] = [255, 255, 0]
    
    # Red Zone (Invalid)
    mask_red = ~valid_map
    overlay[mask_red] = [255, 0, 0]
    
    # Blend with original image
    alpha_blend = 0.4
    blended = cv2.addWeighted(img_rgb, 1.0 - alpha_blend, overlay, alpha_blend, 0)
    
    # 3. Plotting
    plt.figure(figsize=(12, 8))
    plt.imshow(blended)
    
    # Plot Keypoints
    keypoints = np.array(config['keypoints_2d'])
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='white', s=50, marker='*', edgecolors='black', label='Real Keypoints')
    
    # Add Contours for 90 degrees
    # We can use contour plot on theta_map
    plt.contour(theta_map, levels=[90], colors='white', linestyles='dashed', linewidths=2)
    
    # Add Text Annotations
    # Find a point in the Green Zone
    y_g, x_g = np.where(mask_green)
    if len(y_g) > 0:
        idx = len(y_g) // 2
        plt.text(x_g[idx], y_g[idx], 'Frontal FOV\n(< 90°)', color='white', ha='center', va='center', fontweight='bold', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))
        
    # Find a point in the Yellow Zone
    y_y, x_y = np.where(mask_yellow)
    if len(y_y) > 0:
        # Pick a point somewhat far from the 90 boundary
        idx = len(y_y) // 2
        plt.text(x_y[idx], y_y[idx], 'Side/Back FOV\n(> 90°)', color='black', ha='center', va='center', fontweight='bold', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
        
    # Find a point in the Red Zone
    y_r, x_r = np.where(mask_red)
    if len(y_r) > 0:
        idx = len(y_r) // 2
        plt.text(x_r[idx], y_r[idx], 'Invalid Cone', color='white', ha='center', va='center', fontweight='bold', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.title('Double Sphere FOV Zones Augmented with Real Data', fontsize=16)
    plt.legend(loc='upper right')
    plt.axis('off')
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fov_zones_augmented.jpg')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
