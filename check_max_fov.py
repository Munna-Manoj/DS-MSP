
import json
import numpy as np
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
    
    # Grid of pixels
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    points_2d = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2).astype(np.float32)
    
    rays, valid = cam.unproject(points_2d)
    
    # Filter valid rays
    valid_rays = rays[valid]
    
    # Calculate angles
    z = valid_rays[:, 2]
    theta_deg = np.degrees(np.arccos(np.clip(z, -1.0, 1.0)))
    
    max_fov = theta_deg.max()
    min_fov = theta_deg.min()
    
    print(f"Max Valid Angle (Half-FOV): {max_fov:.2f} degrees")
    print(f"Min Valid Angle: {min_fov:.2f} degrees")
    
    if max_fov > 90:
        print("✅ Camera has > 90 degree Half-FOV (Yellow Zone exists).")
    else:
        print("⚠️ Camera has < 90 degree Half-FOV (Yellow Zone empty).")

if __name__ == "__main__":
    main()
