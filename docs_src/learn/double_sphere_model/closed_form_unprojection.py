"""Chapter 2 -- Double Sphere's closed-form unprojection (Usenko et al. 2018).

`mz` below is the solved quadratic from `ds_unproject` in `ds_msp/models/ds_math.py` --
one `np.sqrt`, no Newton iteration. Walked explicitly for one real detected corner, then
cross-checked as a full project(unproject(u)) round trip over all 30 bundled corners
(`test_config.json`) to show the closed form is the *exact* analytic inverse, not an
approximation.
"""
import json

import numpy as np

from ds_msp.models.ds_math import ds_project, ds_unproject


def main() -> None:
    intr = json.load(open("test_config.json"))["intrinsics"]  # the bundled real calibration
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    xi, alpha = intr["xi"], intr["alpha"]
    pixels = np.array(json.load(open("test_config.json"))["test_images"][0]["keypoints_2d"])

    # walk the closed form for the first detected corner
    u, v = pixels[0]
    mx = (u - cx) / fx
    my = (v - cy) / fy
    r2 = mx*mx + my*my
    s = 1.0 - (2.0 * alpha - 1.0) * r2
    mz = (1.0 - alpha*alpha * r2) / (alpha * np.sqrt(s) + (1.0 - alpha))   # the closed form
    print(f"mx={mx:.4f}  my={my:.4f}  r2={r2:.4f}  mz={mz:.4f}")

    # full round trip over every real detected corner: project(unproject(u)) should
    # return u to float64 machine precision -- the closed form is an exact inverse.
    rays, valid_u = ds_unproject(pixels, fx, fy, cx, cy, xi, alpha)
    back_u, back_v, valid_p = ds_project(rays, fx, fy, cx, cy, xi, alpha)
    back = np.stack([back_u, back_v], axis=-1)
    good = valid_u & valid_p
    err = np.linalg.norm(back[good] - pixels[good], axis=1)
    print(f"pixels tested: {int(good.sum())} / {len(pixels)}")
    print(f"project(unproject(u)) round-trip: mean={err.mean():.2e}px  max={err.max():.2e}px")


if __name__ == "__main__":
    main()
