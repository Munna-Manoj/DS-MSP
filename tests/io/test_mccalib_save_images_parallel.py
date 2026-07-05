"""Tests for the parallelized ``save_reprojection_images`` / ``save_detection_images`` —
these used to run fully serially *after* the bundle adjustment already converged (measured
~12.5s each for 255 images on this repo's real MC-Calib dataset), which is why the live view
looked like it "sat there doing nothing" right when a run was actually almost done. Mirrors
``tests/calib/test_charuco.py``'s parallel-vs-serial bit-identical proof for ``detect_rig``.
"""
import hashlib
import os

import cv2
import numpy as np
import pytest

from ds_msp.io.mccalib import save_detection_images, save_reprojection_images
from ds_msp.rig.calibrate import calibrate_rig, make_bundle_front_end

from tests.rig._synth import make_rig

pytestmark = pytest.mark.req("NFR-PERF-004")

W, H = 320, 240


def _write_fake_images(tmp_path, object_obs, cam_prefix="Cam_"):
    """One tiny real PNG per (cam, frame) actually referenced, named to match
    ``_obs_image_path``'s 0-indexed fallback convention -- enough for cv2.imread to succeed."""
    root = tmp_path / "images"
    seen = set()
    for o in object_obs:
        key = (o.cam_id, o.frame_id)
        if key in seen:
            continue
        seen.add(key)
        cam_dir = root / f"{cam_prefix}{o.cam_id + 1:03d}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        img = np.full((H, W, 3), 40, dtype=np.uint8)
        cv2.imwrite(str(cam_dir / f"{o.frame_id:05d}.png"), img)
    return str(root)


def _hash_dir(d):
    h = hashlib.sha256()
    for root, _dirs, files in os.walk(d):
        for f in sorted(files):
            h.update(open(os.path.join(root, f), "rb").read())
    return h.hexdigest()


def _small_calibrated_rig():
    obj, obs, img_size, _gt_ext, _gtm = make_rig(n_cam=3, n_frame=8, noise_px=0.2, seed=0,
                                                 w=W, h=H)
    rig = calibrate_rig(obj, obs, img_size, front_end=make_bundle_front_end(
        {c: "radtan" for c in range(3)}))
    return rig, obj, obs


@pytest.mark.parametrize("fn_name", ["save_reprojection_images", "save_detection_images"])
def test_save_images_parallel_matches_serial_bit_for_bit(tmp_path, fn_name):
    rig, obj, obs = _small_calibrated_rig()
    image_root = _write_fake_images(tmp_path, obs)
    fn = {"save_reprojection_images": save_reprojection_images,
         "save_detection_images": save_detection_images}[fn_name]

    def call(save_dir, workers):
        if fn_name == "save_reprojection_images":
            return fn(rig, obs, image_root, save_dir, workers=workers)
        return fn(obs, image_root, save_dir, workers=workers)

    serial_dir, parallel_dir = tmp_path / "serial", tmp_path / "parallel"
    n1 = call(str(serial_dir), 1)
    n2 = call(str(parallel_dir), 4)

    assert n1 == n2 > 0
    assert _hash_dir(str(serial_dir)) == _hash_dir(str(parallel_dir))


@pytest.mark.parametrize("fn_name", ["save_reprojection_images", "save_detection_images"])
def test_save_images_progress_cb_fires_exactly_once_per_image_under_parallel_workers(
        tmp_path, fn_name):
    rig, obj, obs = _small_calibrated_rig()
    image_root = _write_fake_images(tmp_path, obs)
    fn = {"save_reprojection_images": save_reprojection_images,
         "save_detection_images": save_detection_images}[fn_name]

    calls = []

    def cb(cam_id, i, n, frame_id):
        calls.append((cam_id, i, n, frame_id))

    if fn_name == "save_reprojection_images":
        n = fn(rig, obs, image_root, str(tmp_path / "out"), workers=4, progress_cb=cb)
    else:
        n = fn(obs, image_root, str(tmp_path / "out"), workers=4, progress_cb=cb)

    assert len(calls) == n > 0
    # per-camera "i" counters must be a clean 1..count sequence with no duplicates/gaps even
    # though workers race -- proves the lock correctly serializes the counter.
    by_cam = {}
    for cam_id, i, cnt, _fr in calls:
        by_cam.setdefault(cam_id, []).append(i)
    for cam_id, seq in by_cam.items():
        assert sorted(seq) == list(range(1, len(seq) + 1)), f"cam {cam_id}: {sorted(seq)}"
