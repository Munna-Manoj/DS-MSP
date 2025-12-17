"""
Part 1.3 – Validation & Visualization
-------------------------------------

This script loads calibration results from `calibration.main_real()`,
computes reprojection error, visualizes the reprojected corners over images,
and writes:

    results/calibration_params.json
    results/poses.json
    results/metrics.json
    results/visualizations/*.png

"""

import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple

from calibrate import (
    ds_project,
    ds_unproject,
    load_coco_calibration,
    unpack_params,
    build_checkerboard_points,
    main_real,
)


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ============================================================================
# Utility: ensure results folder exists
# ============================================================================
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


# ============================================================================
# Draw reprojected vs observed points
# ============================================================================
def draw_reprojection(image, uv_obs, uv_proj, valid_mask):
    """
    Draws:
        - observed 2D points in BLUE
        - reprojected points in RED
    """
    img = image.copy()

    for (uo, vo), (up, vp), v in zip(uv_obs, uv_proj, valid_mask):
        if not v:
            continue

        cv2.circle(img, (int(uo), int(vo)), 5, (255, 0, 0), -1)  # observed (blue)
        cv2.circle(img, (int(up), int(vp)), 4, (0, 0, 255), -1)  # projected (red)
        cv2.line(img, (int(uo), int(vo)), (int(up), int(vp)), (0, 255, 0), 2)

    return img


# ============================================================================
# Compute per-image RMS error
# ============================================================================
def compute_rms_error(uv_proj, uv_obs, valid_mask):
    diff = uv_proj[valid_mask] - uv_obs[valid_mask]
    if diff.size == 0:
        return 0.0
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


# ============================================================================
# MAIN VALIDATION PIPELINE
# ============================================================================
def main_validate():
    """
    Validation pipeline.

    Loads:
        - results/calibration_params.json
        - results/poses.json

    Computes:
        - Per-image reprojection visualizations
        - Per-image RMS, global statistics

    Saves:
        results/visualizations/*.png
        results/metrics.json
    """

    # --------------------------------------------------------------
    # 1. Load saved calibration parameters & poses
    # --------------------------------------------------------------
    with open("results/calibration_params.json", "r") as f:
        calib = json.load(f)

    with open("results/poses.json", "r") as f:
        poses = json.load(f)

    # intrinsics + DS
    fx = calib["fx"]
    fy = calib["fy"]
    cx = calib["cx"]
    cy = calib["cy"]
    xi = calib["xi"]
    alpha = calib["alpha"]

    ph = calib["checkerboard_rows"]
    pw = calib["checkerboard_cols"]
    pLength = calib["pLength"]

    # --------------------------------------------------------------
    # 2. Load original COCO annotation for ground-truth keypoints
    # --------------------------------------------------------------
    
    json_path = project_root + "/anns.json"

    (
        X_world_list,
        keypoints_list,
        visibility_list,
        image_info_list
    ) = load_coco_calibration(json_path, ph, pw, pLength)

    num_images = len(image_info_list)

    # --------------------------------------------------------------
    # 3. Prepare output directories
    # --------------------------------------------------------------
    out_dir = "results"
    vis_dir = os.path.join(out_dir, "visualizations")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # --------------------------------------------------------------
    # 4. Loop through each image and compute projection error
    # --------------------------------------------------------------
    all_rms = []

    for i in range(num_images):

        img_rel = image_info_list[i]["file_name"]     # e.g., samples/000011.jpg
        # Always resolve paths relative to project root
        img_path = os.path.join(project_root, img_rel)


        img = cv2.imread(img_path)
        if img is None:
            print(f"WARNING: Could not load image {img_path}")
            continue

        # 2D annotations
        uv_obs = keypoints_list[i]
        vis = visibility_list[i]

        # 3D board points
        Xw = X_world_list[i]

        # Extrinsics from saved JSON
        pose = poses[img_rel]
        r_i = np.array(pose["r"], dtype=np.float64)
        t_i = np.array(pose["t"], dtype=np.float64)

        # world → camera
        R_i, _ = cv2.Rodrigues(r_i)
        Xc = (R_i @ Xw.T).T + t_i.reshape(1, 3)

        # project via Double Sphere
        uv_proj, valid_ds = ds_project(Xc, fx, fy, cx, cy, xi, alpha)
        valid_mask = vis & valid_ds

        # RMS error
        rms = compute_rms_error(uv_proj, uv_obs, valid_mask)
        all_rms.append(rms)

        # Visualization
        vis_img = draw_reprojection(img, uv_obs, uv_proj, valid_mask)
        out_file = os.path.join(vis_dir, f"reproj_{i:03d}.png")
        cv2.imwrite(out_file, vis_img)

        print(f"Saved visualization {out_file} (RMS={rms:.4f})")

    # --------------------------------------------------------------
    # 5. Save metrics.json
    # --------------------------------------------------------------
    metrics = {
        "num_images": num_images,
        "mean_rms": float(np.mean(all_rms)),
        "median_rms": float(np.median(all_rms)),
        "max_rms": float(np.max(all_rms)),
        "min_rms": float(np.min(all_rms)),
        "per_image_rms": [float(v) for v in all_rms],
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nValidation complete!")
    print("Saved:")
    print("  results/metrics.json")
    print("  results/visualizations/*.png")

    return metrics


def validate_single_config(config_path: str):
    """
    Validate images using a standalone config file (e.g., test_config.json).
    This does NOT require previous calibration results.
    """
    print(f"Validating images from config: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # 1. Load Intrinsics
    intr = config['intrinsics']
    fx, fy = intr['fx'], intr['fy']
    cx, cy = intr['cx'], intr['cy']
    xi, alpha = intr['xi'], intr['alpha']
    
    # 2. Load Checkerboard
    cb = config['checkerboard']
    ph, pw = cb['rows'], cb['cols']
    pLength = cb['square_size']
    Xw = build_checkerboard_points(ph, pw, pLength)
    
    # 3. Iterate over images
    # Support both old format (single image) and new format (list of images)
    if 'test_images' in config:
        images = config['test_images']
    else:
        # Backward compatibility
        images = [{
            "file": config['image_file'],
            "keypoints_2d": config['keypoints_2d']
        }]
        
    base_dir = os.path.dirname(os.path.abspath(config_path))
    out_dir = os.path.join(base_dir, "results", "visualizations")
    os.makedirs(out_dir, exist_ok=True)
    
    for idx, img_entry in enumerate(images):
        img_rel = img_entry['file']
        print(f"\nProcessing: {img_rel}")
        
        # Load Image
        img_path = os.path.join(base_dir, img_rel)
        if not os.path.exists(img_path):
            assets_path = os.path.join(base_dir, "assets", img_rel)
            if os.path.exists(assets_path):
                img_path = assets_path
                
        img = cv2.imread(img_path)
        if img is None:
            print(f"WARNING: Could not load image: {img_path}")
            continue
            
        # Load Keypoints
        uv_obs = np.array(img_entry['keypoints_2d'], dtype=np.float32)
        
        # Estimate Pose (PnP)
        # Unproject to unit rays
        rays, valid_unproj = ds_unproject(uv_obs, fx, fy, cx, cy, xi, alpha)
        
        # Filter valid rays
        rays_valid = rays[valid_unproj]
        uv_obs_valid = uv_obs[valid_unproj]
        Xw_valid = Xw[valid_unproj]
        
        if len(rays_valid) < 4:
            print("Not enough valid points for PnP.")
            continue

        # Project to normalized plane (x/z, y/z)
        xn = rays_valid[:, 0] / rays_valid[:, 2]
        yn = rays_valid[:, 1] / rays_valid[:, 2]
        pts_norm = np.stack([xn, yn], axis=-1)
        
        # Solve PnP with Identity matrix
        ret, rvec, tvec = cv2.solvePnP(Xw_valid, pts_norm, np.eye(3), None)
        
        if not ret:
            print("PnP failed.")
            continue
            
        # Reproject and Compute Error
        R, _ = cv2.Rodrigues(rvec)
        Xc = (R @ Xw.T).T + tvec.reshape(1, 3)
        
        uv_proj, valid_ds = ds_project(Xc, fx, fy, cx, cy, xi, alpha)
        valid_mask = valid_ds
        rms = compute_rms_error(uv_proj, uv_obs, valid_mask)
        
        print(f"Pose Estimation Success.")
        print(f"Translation: {tvec.flatten()}")
        print(f"RMS Reprojection Error: {rms:.4f} px")
        
        # Visualize
        vis_img = draw_reprojection(img, uv_obs, uv_proj, valid_mask)
        
        # Save with index or filename
        name_no_ext = os.path.splitext(os.path.basename(img_rel))[0]
        out_file = os.path.join(out_dir, f"validate_{name_no_ext}.png")
        cv2.imwrite(out_file, vis_img)
        print(f"Saved visualization to {out_file}")


# ============================================================================
# Run directly
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to test_config.json for single image validation")
    args = parser.parse_args()
    
    if args.config:
        validate_single_config(args.config)
    else:
        # Default behavior: load from results/
        if os.path.exists("results/calibration_params.json"):
            main_validate()
        else:
            print("No calibration results found. Run 'python calibrate.py' first, or use '--config test_config.json' to validate the test image.")
