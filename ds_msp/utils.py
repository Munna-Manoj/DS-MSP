import numpy as np
import json
from typing import List, Tuple, Dict

def build_checkerboard_points(ph: int, pw: int, pLength: float) -> np.ndarray:
    """
    Build 3D world coordinates for a ph x pw checkerboard.
    """
    xs = np.arange(pw) * pLength
    ys = np.arange(ph) * pLength
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = np.zeros_like(X)

    Xw = np.stack([X, Y, Z], axis=-1)  # (ph, pw, 3)
    return Xw.reshape(-1, 3)  # (ph*pw, 3)

def pack_params(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    xi: float,
    alpha: float,
    r_list: List[np.ndarray],
    t_list: List[np.ndarray],
) -> np.ndarray:
    """
    Pack all calibration parameters into a single 1D vector.
    """
    num_images = len(r_list)
    assert num_images == len(t_list), "r_list and t_list must have same length"

    # First 6 params: intrinsics + DS
    param_list = [fx, fy, cx, cy, xi, alpha]

    # Then 6 params per image: [r_i(3), t_i(3)]
    for i in range(num_images):
        r_i = np.asarray(r_list[i]).reshape(3)
        t_i = np.asarray(t_list[i]).reshape(3)
        param_list.extend(r_i.tolist())
        param_list.extend(t_i.tolist())

    return np.array(param_list, dtype=np.float64)

def unpack_params(
    params: np.ndarray,
    num_images: int,
) -> Tuple[
    float, float, float, float, float, float, List[np.ndarray], List[np.ndarray]
]:
    """
    Unpack 1D parameter vector into intrinsics, DS params, and per-image extrinsics.
    """
    params = np.asarray(params).ravel()
    expected_len = 6 + 6 * num_images
    assert (
        params.shape[0] == expected_len
    ), f"Expected param length {expected_len}, got {params.shape[0]}"

    # First 6: intrinsics + DS
    fx, fy, cx, cy, xi, alpha = params[:6]

    r_list = []
    t_list = []

    # Each image contributes 6 params
    offset = 6
    for _ in range(num_images):
        r_i = params[offset : offset + 3]
        t_i = params[offset + 3 : offset + 6]
        r_list.append(r_i.copy())
        t_list.append(t_i.copy())
        offset += 6

    return fx, fy, cx, cy, xi, alpha, r_list, t_list

def load_coco_calibration(
    json_path: str,
    ph: int,
    pw: int,
    pLength: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[Dict]]:
    """
    Load COCO-style calibration annotations.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    # Map image_id -> annotation (assuming one board per image)
    ann_by_img: Dict[int, dict] = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id in ann_by_img:
            raise ValueError(f"Multiple annotations for image_id={img_id}")
        ann_by_img[img_id] = ann

    Xw_board = build_checkerboard_points(ph, pw, pLength)
    N_expected = ph * pw

    X_world_list: List[np.ndarray] = []
    keypoints_list: List[np.ndarray] = []
    visibility_list: List[np.ndarray] = []
    image_info_list: List[Dict] = []

    # Iterate images in order of id
    for img in sorted(images, key=lambda im: im["id"]):
        img_id = img["id"]
        if img_id not in ann_by_img:
            # No board for this image; skip
            continue

        ann = ann_by_img[img_id]
        kps = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        if kps.shape[0] != N_expected:
            raise ValueError(
                f"Image {img_id}: expected {N_expected} keypoints, got {kps.shape[0]}"
            )

        uv = kps[:, :2]  # (N, 2)
        vis = kps[:, 2] == 2  # (N,)

        X_world_list.append(Xw_board.copy())
        keypoints_list.append(uv)
        visibility_list.append(vis)
        image_info_list.append(img)

    return X_world_list, keypoints_list, visibility_list, image_info_list
