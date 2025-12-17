
import json
import cv2
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ds_camera_cv

def main():
    # 1. Load calibration parameters (User provided)
    fx = 711.5744706559915
    fy = 711.2367154139102
    cx = 949.1837602591455
    cy = 518.8057004536004
    xi = 0.18321185451070932
    alpha = 0.8086089938575695
    width, height = 1920, 1080
    
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.array([xi, alpha])
    
    # 2. Load 2D keypoints from anns.json
    anns_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'anns.json')
    with open(anns_path, 'r') as f:
        data = json.load(f)
    
    # Find image 1 (samples/000011.jpg)
    img_info = next(img for img in data['images'] if img['file_name'] == 'samples/000011.jpg')
    img_id = img_info['id']
    ann = next(a for a in data['annotations'] if a['image_id'] == img_id)
    
    # Extract keypoints (x, y, v) -> (x, y)
    keypoints = np.array(ann['keypoints']).reshape(-1, 3)[:, :2]
    
    # 3. Define 3D points (6x5 grid, 0.2m square)
    # Based on visual inspection: 6 columns, 5 rows
    rows = 5
    cols = 6
    square_size = 0.2
    
    points_3d = []
    for i in range(rows):
        for j in range(cols):
            points_3d.append([j * square_size, i * square_size, 0.0])
    points_3d = np.array(points_3d, dtype=np.float64)
    
    # Ensure we have 30 points
    if len(keypoints) != 30:
        print(f"Error: Expected 30 keypoints, found {len(keypoints)}")
        return
    
    # 4. Undistort points to normalized coordinates
    # P=None returns normalized coordinates (x, y)
    points_2d_norm = ds_camera_cv.undistortPoints(keypoints, K, D)
    
    # 5. Solve PnP
    # We use identity camera matrix because points are already normalized
    print(f"points_3d shape: {points_3d.shape}, dtype: {points_3d.dtype}")
    print(f"points_2d_norm shape: {points_2d_norm.shape}, dtype: {points_2d_norm.dtype}")
    
    # Ensure contiguous arrays
    points_3d = np.ascontiguousarray(points_3d).reshape(-1, 3)
    points_2d_norm = np.ascontiguousarray(points_2d_norm).reshape(-1, 2)
    
    success, rvec, tvec = cv2.solvePnP(points_3d, points_2d_norm, np.eye(3), None)
    
    if not success:
        print("Error: solvePnP failed")
        return
    
    print("Pose estimated successfully:")
    print(f"rvec: {rvec.flatten()}")
    print(f"tvec: {tvec.flatten()}")
    
    # 6. Reproject points
    points_reproj, _ = ds_camera_cv.projectPoints(points_3d, rvec, tvec, K, D)
    points_reproj = points_reproj.reshape(-1, 2)
    
    # 7. Calculate error
    error = np.linalg.norm(points_reproj - keypoints, axis=1).mean()
    print(f"Reprojection error: {error:.4f} pixels")
    
    # 8. Visualize
    img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), img_info['file_name'])
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return
        
    # Draw original points (Green)
    for pt in keypoints:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
        
    # Draw reprojected points (Red)
    for pt in points_reproj:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        
    # Draw axes
    axis_length = 0.4 # 2 squares
    axis_points = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ], dtype=np.float64)
    
    axis_reproj, _ = ds_camera_cv.projectPoints(axis_points, rvec, tvec, K, D)
    axis_reproj = axis_reproj.reshape(-1, 2).astype(int)
    
    origin = tuple(axis_reproj[0])
    cv2.arrowedLine(img, origin, tuple(axis_reproj[1]), (0, 0, 255), 3) # X - Red
    cv2.arrowedLine(img, origin, tuple(axis_reproj[2]), (0, 255, 0), 3) # Y - Green
    cv2.arrowedLine(img, origin, tuple(axis_reproj[3]), (255, 0, 0), 3) # Z - Blue
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reprojection_result.jpg')
    cv2.imwrite(output_path, img)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    main()
