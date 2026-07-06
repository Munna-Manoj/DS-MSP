"""Confirm a save-then-load round-trip through the Kalibr YAML format is exact.

Writes the bundled Double Sphere sample calibration out, reads it back, and
compares ``params`` -- the flat parameter vector every DS-MSP model exposes.
No external file is needed: everything round-trips through a temp file.
"""
import os
import tempfile

import numpy as np

from ds_msp.io.kalibr import load_kalibr, save_kalibr
from ds_msp.models.double_sphere import DoubleSphereModel


def main() -> None:
    ds = DoubleSphereModel.sample()
    out = os.path.join(tempfile.gettempdir(), "rt.yaml")

    save_kalibr(ds, out, width=1920, height=1080, cam="cam0")
    back = load_kalibr(out, cam="cam0")

    print(type(back) is type(ds))                          # same DS-MSP class
    print(np.allclose(back.params, ds.params, atol=1e-9))   # same parameters
    print(np.max(np.abs(back.params - ds.params)))          # exact max diff


if __name__ == "__main__":
    main()
