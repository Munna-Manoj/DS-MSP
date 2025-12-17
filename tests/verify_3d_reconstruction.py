
import json
import cv2
import numpy as np
import os
import sys

# Add current directory to path
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ds_msp.cv as ds_camera_cv
from ds_msp.model import DoubleSphereCamera

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def verify_3d_reconstruction(points_2d_orig, points_3d_gt, K, D, K_new, rvec, tvec, w, h, name):
    print(f"\n--- Verifying 3D Reconstruction: {name} ---")
    
    # 1. Undistort original 2D points to the new image frame
    # We use our wrapper which handles the mapping correctly
    points_undist = ds_camera_cv.undistortPoints(points_2d_orig, K, D, P=K_new)
    points_undist = points_undist.reshape(-1, 2)
    
    # 2. Unproject to Unit Rays using K_new^-1 (Pinhole)
    fx_new, fy_new = K_new[0, 0], K_new[1, 1]
    cx_new, cy_new = K_new[0, 2], K_new[1, 2]
    
    mx = (points_undist[:, 0] - cx_new) / fx_new
    my = (points_undist[:, 1] - cy_new) / fy_new
    mz = np.ones_like(mx)
    
    rays_cam = np.stack([mx, my, mz], axis=-1)
    # Normalize? Not strictly necessary for intersection if we use z=1 plane logic, 
    # but let's normalize to be proper unit vectors
    rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=-1, keepdims=True)
    
    # 3. Intersect Rays with Checkerboard Plane
    # Plane is defined by R, t.
    # Normal in camera frame: n_c = R * [0, 0, 1]^T = r3 (3rd col of R)
    # Point on plane: t
    # Ray: lambda * d
    # lambda = (t . n_c) / (d . n_c)
    
    R, _ = cv2.Rodrigues(rvec)
    n_c = R[:, 2] # Normal vector
    
    # Dot products
    numer = np.dot(tvec.flatten(), n_c)
    denom = np.dot(rays_cam, n_c)
    
    lambdas = numer / denom
    
    # Reconstructed points in Camera Frame
    points_recon_cam = rays_cam * lambdas[:, np.newaxis]
    
    # 4. Transform back to World Frame
    # X_c = R * X_w + t  =>  X_w = R^T * (X_c - t)
    points_recon_world = (points_recon_cam - tvec.flatten()) @ R
    
    # 5. Compare with Ground Truth
    # Z should be 0
    z_error = np.abs(points_recon_world[:, 2]).mean()
    pos_error = np.linalg.norm(points_recon_world - points_3d_gt, axis=1).mean()
    
    print(f"Mean Position Error: {pos_error:.6f} meters")
    print(f"Mean Z-plane Error:  {z_error:.6f} meters")
    
    # 6. Check Square Size Consistency
    # Compute distances between adjacent points
    # Grid is 6 cols, 5 rows.
    # Horizontal neighbors: indices i and i+1 (if not at end of row)
    # Vertical neighbors: indices i and i+6
    
    distances = []
    
    # Reshape to (rows, cols, 3)
    grid_recon = points_recon_world.reshape(5, 6, 3)
    
    # Horizontal
    h_dists = np.linalg.norm(grid_recon[:, :-1] - grid_recon[:, 1:], axis=2)
    distances.extend(h_dists.flatten())
    
    # Vertical
    v_dists = np.linalg.norm(grid_recon[:-1, :] - grid_recon[1:, :], axis=2)
    distances.extend(v_dists.flatten())
    
    distances = np.array(distances)
    mean_dist = distances.mean()
    std_dist = distances.std()
    
    print(f"Reconstructed Square Size: {mean_dist:.6f} +/- {std_dist:.6f} meters")
    print(f"Target Square Size: 0.200000 meters")
    
    if pos_error < 0.001 and abs(mean_dist - 0.2) < 0.001:
        print("✅ Verification Successful: 3D Geometry Preserved.")
    else:
        print("❌ Verification Failed: Geometry mismatch.")

def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_config.json')
    config = load_config(config_path)
    
    intr = config['intrinsics']
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]])
    D = np.array([intr['xi'], intr['alpha']])
    w, h = intr['width'], intr['height']
    
    # Load Image 96 Data (Sample 96 is the second entry in test_config.json, or we search for it)
    # Actually, let's just use the first entry which is test_image.jpg (Sample 11)
    # Or if we want Sample 96 specifically as the code implies (it says "Load Image 96 Data"),
    # we should find it. But test_config.json has test_image.jpg first.
    # Let's use the first image in the config.
    image_entry = config['test_images'][0]
    points_2d = np.array(image_entry['keypoints_2d'], dtype=np.float64)
    
    # 3D Points
    rows = config['checkerboard']['rows']
    cols = config['checkerboard']['cols']
    square_size = config['checkerboard']['square_size']
    points_3d = []
    for i in range(rows):
        for j in range(cols):
            points_3d.append([j * square_size, i * square_size, 0.0])
    points_3d = np.array(points_3d, dtype=np.float64)
    
    # Estimate Pose first (using standard pipeline)
    points_2d_norm = ds_camera_cv.undistortPoints(points_2d, K, D)
    points_3d_cont = np.ascontiguousarray(points_3d).reshape(-1, 3)
    points_2d_norm_cont = np.ascontiguousarray(points_2d_norm).reshape(-1, 2)
    success, rvec, tvec = cv2.solvePnP(points_3d_cont, points_2d_norm_cont, np.eye(3), None)
    
    if not success:
        print("Pose estimation failed")
        return

    # 1. Optimal Crop
    K_new_crop = ds_camera_cv.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), balance=1.0)
    verify_3d_reconstruction(points_2d, points_3d, K, D, K_new_crop, rvec, tvec, w, h, "Optimal Crop")
    
    # 2. Keep Whole
    K_new_whole = ds_camera_cv.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), balance=0.0)
    verify_3d_reconstruction(points_2d, points_3d, K, D, K_new_whole, rvec, tvec, w, h, "Keep Whole")
    
    # 3. Zoom Out
    K_new_zoom = K_new_whole.copy()
    K_new_zoom[0, 0] /= 4.0
    K_new_zoom[1, 1] /= 4.0
    K_new_zoom[0, 2] = w / 2.0
    K_new_zoom[1, 2] = h / 2.0
    verify_3d_reconstruction(points_2d, points_3d, K, D, K_new_zoom, rvec, tvec, w, h, "Zoom Out (4x)")

if __name__ == "__main__":
    main()
