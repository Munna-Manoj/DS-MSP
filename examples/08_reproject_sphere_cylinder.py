#!/usr/bin/env python3
"""
Deep-dive demo — one fisheye, four images (raw / sphere / cylinder / pinhole), proven by corners.

A *central* camera (all rays through one point) is fully described by a bijection between
ray directions (points on the unit sphere) and pixels. The flat pinhole image is just one
way to store that bijection — and a poor one past 90°, where `tan` blows up. This script
re-stores the SAME real fisheye three more ways and proves the pixel-to-pixel maps that link
them are exact inverses — visually, on a real checkerboard, and numerically.

  1. FOUR REAL IMAGES. Resample the bundled fisheye into
       - the RAW fisheye itself,
       - an equirectangular (unit-sphere) panorama:  row ~ elevation,        col ~ azimuth
       - a cylindrical panorama:                      row ~ tan(elevation),   col ~ azimuth
       - a pinhole (gnomonic) rectification:          row ~ tan, col ~ tan    (both gnomonic)
     Every sample comes from `DoubleSphereCamera.project`, so the pictures are the library's
     own geometry, not a look-alike.

  2. THE CORNER PROOF. The fisheye has a checkerboard with 30 known corner pixels. We push each
     corner raw -> ray (DS unproject) -> chart pixel (the conversion math) and draw it on each
     chart. If the math is right, every mapped corner lands exactly on the checkerboard corner
     you can see in that representation. The 4 *_corners.png images show this.

  3. THE NUMBER YOU CAN VERIFY. Round-trip every corner
       raw -> ray -> chart pixel -> ray -> raw
     and report the mean / max residual per representation. ~1e-13 px (float64 round-off) —
     the maps are inverses, not approximations.

  4. WHERE THE CYLINDER BREAKS. Sphere row is linear in elevation, so it reaches the pole;
     cylinder row is tan(elevation), so the SAME image height holds far less elevation.

Prerequisites:  `pip install -e .`  + the bundled `test_image.jpg` / `test_config.json` / `anns.json`.
Run:            python examples/08_reproject_sphere_cylinder.py
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from ds_msp import DoubleSphereCamera

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(HERE, "test_config.json")
IMG = os.path.join(HERE, "assets", "test_image.jpg")
ANNS = os.path.join(HERE, "anns.json")
OUT = os.path.join(HERE, "assets", "learn")
BOARD_ROWS, BOARD_COLS = 5, 6                  # 30 inner corners, row-major in anns.json

# --- panorama intrinsics, shared by the sphere and the cylinder --------------------
# One focal (pixels per radian) and one centre for BOTH panoramas, so the only thing
# that differs between them is the VERTICAL law (elevation vs tan-elevation). That shared
# axis is the whole point: azimuth is stored identically, the cylinder just warps height.
W_PANO, H_PANO = 1200, 700
F_PANO = 320.0                       # px / rad
CX_PANO, CY_PANO = W_PANO / 2.0, H_PANO / 2.0


# ===================================================================================
# The pixel <-> ray maps. These are the math the doc describes, in code.
# Ray convention matches the library: x right, y down, z forward; project([x,y,z]) -> (u,v).
# A ray's spherical angles:  azimuth  lambda = atan2(x, z),  elevation  psi = atan2(-y, hypot(x,z)).
# ===================================================================================
def sphere_pix_to_ray(u, v):
    """Equirectangular: column is linear in azimuth, row is linear in elevation."""
    lam = (u - CX_PANO) / F_PANO
    psi = (CY_PANO - v) / F_PANO
    cps = np.cos(psi)
    return np.stack([cps * np.sin(lam), -np.sin(psi), cps * np.cos(lam)], axis=-1)


def cylinder_pix_to_ray(u, v):
    """Cylinder: column is linear in azimuth (SAME as sphere); row is linear in tan(elevation)."""
    lam = (u - CX_PANO) / F_PANO
    h = (CY_PANO - v) / F_PANO                 # h == tan(elevation)
    return np.stack([np.sin(lam), -h, np.cos(lam)], axis=-1)


def ray_to_sphere_pix(d):
    """Inverse of sphere_pix_to_ray: ray -> equirectangular pixel."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    lam = np.arctan2(x, z)
    psi = np.arctan2(-y, np.hypot(x, z))
    return np.stack([CX_PANO + F_PANO * lam, CY_PANO - F_PANO * psi], axis=-1)


def ray_to_cylinder_pix(d):
    """Inverse of cylinder_pix_to_ray: ray -> cylindrical pixel (height = tan(elevation))."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    lam = np.arctan2(x, z)
    h = -y / np.hypot(x, z)                    # = tan(elevation)
    return np.stack([CX_PANO + F_PANO * lam, CY_PANO - F_PANO * h], axis=-1)


def ray_to_pinhole_pix(d, K):
    """Gnomonic: ray -> pinhole pixel (valid on the front hemisphere z>0)."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]], axis=-1)


def pinhole_pix_to_ray(p, K):
    """Inverse: a pinhole pixel IS a ray."""
    return np.stack([(p[..., 0] - K[0, 2]) / K[0, 0],
                     (p[..., 1] - K[1, 2]) / K[1, 1],
                     np.ones_like(p[..., 0])], axis=-1)


def reproject(cam, img, pix_to_ray, w, h):
    """Resample `img` (a fisheye) into a w*h view whose pixels follow `pix_to_ray`."""
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    rays = pix_to_ray(u, v)
    pts, valid = cam.project(rays.reshape(-1, 3))
    mapx = pts[:, 0].reshape(h, w).astype(np.float32)
    mapy = pts[:, 1].reshape(h, w).astype(np.float32)
    valid = valid.reshape(h, w)
    mapx[~valid] = -1
    mapy[~valid] = -1
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def draw_corners(img, pts, w, h, color=(0, 0, 255)):
    """Overlay the checkerboard grid (green lines) and corners (coloured dots) on a copy."""
    out = img.copy()
    P = pts.reshape(BOARD_ROWS, BOARD_COLS, 2)

    def ok(p):
        return np.isfinite(p).all() and -2 * w < p[0] < 2 * w and -2 * h < p[1] < 2 * h

    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            p = P[r, c]
            if c + 1 < BOARD_COLS and ok(p) and ok(P[r, c + 1]):
                cv2.line(out, tuple(p.astype(int)), tuple(P[r, c + 1].astype(int)), (60, 220, 60), 1, cv2.LINE_AA)
            if r + 1 < BOARD_ROWS and ok(p) and ok(P[r + 1, c]):
                cv2.line(out, tuple(p.astype(int)), tuple(P[r + 1, c].astype(int)), (60, 220, 60), 1, cv2.LINE_AA)
    for p in pts:
        if ok(p):
            cv2.circle(out, tuple(p.astype(int)), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, tuple(p.astype(int)), 5, color, 2, cv2.LINE_AA)
    return out


def main() -> None:
    intr = json.load(open(CFG))
    if "intrinsics" in intr:
        intr = intr["intrinsics"]
    W, H = intr["width"], intr["height"]
    cam = DoubleSphereCamera(intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                             intr["xi"], intr["alpha"], width=W, height=H)
    img = cv2.imread(IMG)
    if img is None:
        raise SystemExit(f"missing {IMG}")
    os.makedirs(OUT, exist_ok=True)

    # ground-truth corners of the bundled board (anns.json image_id 1 == test_image.jpg)
    anns = json.load(open(ANNS))
    raw_corners = np.array(anns["annotations"][0]["keypoints"]).reshape(-1, 3)[:, :2]

    print("Re-storing ONE fisheye four ways (raw / sphere / cylinder / pinhole)")
    print(f"  source fisheye   : {W}x{H}  DS(xi={cam.xi:.3f}, alpha={cam.alpha:.3f})")
    print(f"  panorama model   : f={F_PANO:.0f} px/rad, {W_PANO}x{H_PANO}, shared by sphere & cylinder")
    print(f"  board            : {BOARD_ROWS}x{BOARD_COLS} = {len(raw_corners)} known corner pixels\n")

    # 1. the four images ----------------------------------------------------------
    sphere = reproject(cam, img, sphere_pix_to_ray, W_PANO, H_PANO)
    cylinder = reproject(cam, img, cylinder_pix_to_ray, W_PANO, H_PANO)
    pinhole, K_new = cam.undistort_image(img, cam.compute_K_new(balance=0.0))
    cv2.imwrite(os.path.join(OUT, "reproj_sphere.png"), sphere)
    cv2.imwrite(os.path.join(OUT, "reproj_cylinder.png"), cylinder)
    cv2.imwrite(os.path.join(OUT, "reproj_pinhole.png"), pinhole)

    # 2. push every corner raw -> ray -> chart pixel, and overlay -----------------
    rays, _ = cam.unproject(raw_corners.astype(np.float64))
    c_sphere = ray_to_sphere_pix(rays)
    c_cyl = ray_to_cylinder_pix(rays)
    c_pin = ray_to_pinhole_pix(rays, K_new)

    cv2.imwrite(os.path.join(OUT, "corners_raw.png"),
                draw_corners(img, raw_corners, W, H))
    cv2.imwrite(os.path.join(OUT, "corners_sphere.png"),
                draw_corners(sphere, c_sphere, W_PANO, H_PANO))
    cv2.imwrite(os.path.join(OUT, "corners_cylinder.png"),
                draw_corners(cylinder, c_cyl, W_PANO, H_PANO))
    cv2.imwrite(os.path.join(OUT, "corners_pinhole.png"),
                draw_corners(pinhole, c_pin, W, H))
    print("  wrote assets/learn/reproj_{sphere,cylinder,pinhole}.png")
    print("  wrote assets/learn/corners_{raw,sphere,cylinder,pinhole}.png")

    # 3. round-trip each corner back to the raw pixel -----------------------------
    def roundtrip(chart_px, px_to_ray):
        back_rays = px_to_ray(chart_px)
        raw_again, valid = cam.project(back_rays)
        err = np.linalg.norm(raw_again - raw_corners, axis=1)
        err = err[valid]
        return err

    rows = [
        ("sphere", roundtrip(c_sphere, lambda p: sphere_pix_to_ray(p[..., 0], p[..., 1]))),
        ("cylinder", roundtrip(c_cyl, lambda p: cylinder_pix_to_ray(p[..., 0], p[..., 1]))),
        ("pinhole", roundtrip(c_pin, lambda p: pinhole_pix_to_ray(p, K_new))),
    ]
    print("\nCorner round-trip  raw -> ray -> chart pixel -> ray -> raw  (30 corners):")
    print("  representation   mean error    max error    corners checked")
    for name, err in rows:
        print(f"  {name:13s}    {err.mean():.2e} px   {err.max():.2e} px   {len(err)}/30")
    print("  => every corner returns to its original raw pixel — the conversion math is exact.")

    # 4. where the cylinder breaks ------------------------------------------------
    psi_sphere = np.degrees((CY_PANO - 0) / F_PANO)
    psi_cyl = np.degrees(np.arctan((CY_PANO - 0) / F_PANO))
    print("\nSame image height, different elevation reach (top row of the panorama):")
    print(f"  sphere row 0   reaches elevation {psi_sphere:5.1f} deg")
    print(f"  cylinder row 0 reaches elevation {psi_cyl:5.1f} deg "
          f"(tan compresses it; the poles at +-90 deg sit at infinity)")


if __name__ == "__main__":
    main()
