"""Confirm the Double Sphere paraxial-focal derivation on the bundled fixture.

Section 2 of the companion page derives, by taking theta -> 0 in `ds_project`, that
Double Sphere's *paraxial* (near-axis) focal length is not `fx` but `fx / (1 + xi)`.
This finite-differences the real `DoubleSphereModel.project` at a tiny angle off the
optical axis and checks it against that closed form -- using the bundled DS
calibration (`test_config.json`), not the TUM-VI numbers quoted in the prose (those
need the external dataset; see `examples/05_model_equivalence.py`).
"""
import json

import numpy as np

from ds_msp.models import DoubleSphereModel


def radius(model: DoubleSphereModel, theta: float) -> float:
    """Image radius (px from the principal point) of a ray at angle theta off-axis."""
    d = np.array([[np.sin(theta), 0.0, np.cos(theta)]])
    uv, _ = model.project(d)
    return float(np.hypot(uv[0, 0] - model.cx, uv[0, 1] - model.cy))


def main() -> None:
    intr = json.load(open("test_config.json"))["intrinsics"]  # the bundled real calibration
    model = DoubleSphereModel(intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                               intr["xi"], intr["alpha"])
    print(f"fx={model.fx:.4f}  xi={model.xi:.4f}")

    # closed form derived in section 2:  dr/dtheta|0 = fx / (1 + xi)
    f_formula = model.fx / (1.0 + model.xi)

    # finite-difference the real project() at a tiny angle off the axis
    h = 1e-5  # radians
    f_numeric = radius(model, h) / h

    print(f"closed form   fx/(1+xi)        = {f_formula:.6f}")
    print(f"finite diff   radius(h)/h      = {f_numeric:.6f}   (h={h:.0e} rad)")
    print(f"difference                     = {abs(f_formula - f_numeric):.2e} px")


if __name__ == "__main__":
    main()
