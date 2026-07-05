"""Config-agnostic single-camera calibration driver — the literal realization of "front end
produces correspondences, backend is the same": :func:`calibrate_camera` is the same three
lines regardless of whether ``board`` is a checkerboard, ChArUco, or AprilGrid target.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from ..geometry.resection import intrinsics_seed
from ..models.registry import model_class, seed_from_K
from .board import Board, ProgressCB, to_correspondences
from .bundle import calibrate


def calibrate_camera(board: Board, image_paths: Sequence[str], model_name: str,
                     width: int, height: int, *,
                     progress_cb: Optional[ProgressCB] = None, **calib_kwargs) -> Dict:
    """Detect ``board`` in every image, seed a from-scratch pinhole ``K``, and refine the
    named camera model with :func:`ds_msp.calib.bundle.calibrate`. ``calib_kwargs`` are
    forwarded to it verbatim (``robust``, ``robust_scale``, ``gnc``, ``multi_start``,
    ``n_restarts``, ``seed``, ``max_nfev``, ``verbose``, ...). Adds ``n_images_detected``/
    ``n_images_total`` to the result so a caller can report detection yield. ``progress_cb``
    (if given) fires once per completed image during detection — see ``ds_msp.calib.report``
    for a terminal-line implementation."""
    obs = board.detect(image_paths, progress_cb=progress_cb)
    if not obs:
        raise ValueError("no board detected in any of the given images")
    X_world_list, keypoints_list, visibility_list = to_correspondences(obs)
    K0, _ = intrinsics_seed(X_world_list, keypoints_list, width, height)
    init_model = seed_from_K(model_class(model_name), K0)
    result = calibrate(init_model, X_world_list, keypoints_list, visibility_list, **calib_kwargs)
    result["n_images_detected"] = len(obs)
    result["n_images_total"] = len(image_paths)
    return result
