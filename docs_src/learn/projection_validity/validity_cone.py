"""Chapter 3 -- the Double Sphere valid-ray half-space (Usenko et al. 2018, Eq. 43-45).

For a unit-length ray at incidence angle theta, the half-space test z > -w2*d1
collapses to cos(theta) > -w2, so theta_max = arccos(-w2) is the widest ray the
model accepts. This checks that analytic value against a brute-force sweep.
"""

import json

import numpy as np

from ds_msp import DoubleSphereCamera


def main() -> None:
    intr = json.load(open("test_config.json"))["intrinsics"]  # the bundled real calibration
    cam = DoubleSphereCamera(intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                             intr["xi"], intr["alpha"],
                             width=intr["width"], height=intr["height"])
    xi, alpha = cam.xi, cam.alpha
    print(f"xi={xi:.4f}, alpha={alpha:.4f}")

    # analytic: for a unit ray (d1=1) the half-space test z > -w2*d1 becomes cos(theta) > -w2
    w1 = (1 - alpha) / alpha if alpha > 0.5 else alpha / (1 - alpha)
    w2 = (w1 + xi) / np.sqrt(2 * w1 * xi + xi * xi + 1.0)
    theta_max = np.degrees(np.arccos(-w2))
    print(f"w2 = {w2:.4f}, theta_max = {theta_max:.1f} deg")

    # numeric check: sweep rays from 0 to 180 deg, ask the model which ones it accepts
    thetas = np.linspace(0, np.pi, 4000)
    rays = np.stack([np.sin(thetas), np.zeros_like(thetas), np.cos(thetas)], axis=1)
    _, valid = cam.project(rays)
    numeric_theta_max = np.degrees(thetas[valid].max())
    print(f"numeric check: {numeric_theta_max:.1f} deg")
    print(f"total field of view: {2 * theta_max:.0f} deg")


if __name__ == "__main__":
    main()
