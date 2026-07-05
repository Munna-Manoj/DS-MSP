"""Write a DS-MSP model out as a Kalibr camchain YAML.

``save_kalibr`` reads the model's type and emits the matching ``camera_model`` /
``distortion_model`` / ``intrinsics`` fields (see ``ds_msp/io/kalibr.py``'s module
docstring for the full per-model table). Uses the bundled Double Sphere sample
calibration (``DoubleSphereModel.sample()``) so this runs with no external file.
"""
import os
import tempfile

from ds_msp.io.kalibr import save_kalibr
from ds_msp.models.double_sphere import DoubleSphereModel


def main() -> None:
    ds = DoubleSphereModel.sample()  # bundled DS calibration, no file needed
    out = os.path.join(tempfile.gettempdir(), "camchain.yaml")

    save_kalibr(ds, out, width=1920, height=1080, cam="cam0")
    print(open(out).read())


if __name__ == "__main__":
    main()
