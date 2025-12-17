
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
    
    # 1. Generate a dense grid of rays in the X-Z plane (Y=0)
    # Angles from -180 to 180 degrees
    thetas = np.linspace(-np.pi, np.pi, 720)
    
    # Convert to unit vectors (x, 0, z)
    # theta=0 -> z=1 (Forward)
    # theta=90 -> x=1 (Right)
    # theta=180 -> z=-1 (Backward)
    
    # Note: standard convention: z is forward. x is right.
    # So x = sin(theta), z = cos(theta)
    
    rays_x = np.sin(thetas)
    rays_z = np.cos(thetas)
    rays_y = np.zeros_like(rays_x)
    
    rays = np.stack([rays_x, rays_y, rays_z], axis=-1)
    
    # 2. Check validity using Double Sphere projection logic
    # We use the internal logic of 'project' but just check the validity condition
    
    x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
    
    # DS Projection Math
    d1 = np.sqrt(x*x + y*y + z*z) # Should be 1.0 for unit rays
    z1 = z + xi * d1
    d2 = np.sqrt(x*x + y*y + z1*z1)
    den = alpha * d2 + (1.0 - alpha) * z1
    
    valid_mask = den > 1e-8
    
    # 3. Also check if it projects onto the image sensor
    # Project to pixels
    points_2d, valid_project = cam.project(rays)
    
    # Check if inside image bounds
    in_image = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
               (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
               
    valid_sensor = valid_mask & valid_project & in_image
    
    # 4. Plotting
    plt.figure(figsize=(10, 10))
    
    # Plot Camera Center
    plt.plot(0, 0, 'ko', markersize=10, label='Camera Center')
    
    # Plot Rays
    # We plot lines from (0,0) to (x, z)
    
    # Valid Sensor Rays (Green)
    for i in range(len(thetas)):
        if valid_sensor[i]:
            plt.plot([0, rays_x[i]], [0, rays_z[i]], 'g-', alpha=0.1)
            
    # Valid Model but Out of Sensor (Blue)
    for i in range(len(thetas)):
        if valid_mask[i] and not valid_sensor[i]:
            plt.plot([0, rays_x[i]], [0, rays_z[i]], 'b-', alpha=0.1)
            
    # Invalid Model (Red) - The Cone
    for i in range(len(thetas)):
        if not valid_mask[i]:
            plt.plot([0, rays_x[i]], [0, rays_z[i]], 'r-', alpha=0.3)

    # 5. Plot Original Data Keypoints (Projected to X-Z plane)
    # We unproject the 2D keypoints to get their rays
    keypoints = np.array(config['keypoints_2d'])
    rays_kp, valid_kp = cam.unproject(keypoints)
    
    # Normalize
    rays_kp = rays_kp / np.linalg.norm(rays_kp, axis=1, keepdims=True)
    
    # Project to X-Z plane (ignore Y) and normalize again for visualization
    kp_x = rays_kp[:, 0]
    kp_z = rays_kp[:, 2]
    norm_xz = np.sqrt(kp_x**2 + kp_z**2)
    kp_x /= norm_xz
    kp_z /= norm_xz
    
    plt.plot(kp_x, kp_z, 'k*', markersize=10, label='Real Data Keypoints (Projected to XZ)')
    
    # Add labels and legend
    plt.xlabel('X (Lateral)')
    plt.ylabel('Z (Forward)')
    plt.title(f'Double Sphere FOV Analysis (Top-Down View)\nxi={xi:.3f}, alpha={alpha:.3f}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Add annotations
    plt.text(0, 0.5, 'Valid FOV', ha='center', color='green', fontweight='bold')
    plt.text(0, -0.5, 'Invalid Cone', ha='center', color='red', fontweight='bold')
    
    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fov_cone_diagram.jpg')
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
