"""How-to: convert a calibration between camera models, in one call.

`convert(source, target_class, width=..., height=...)` fits `target_class` to
reproduce `source`'s geometry -- no images, no recalibration. Runs on the bundled
Double Sphere sample calibration (`DoubleSphereModel.sample()`, no file needed),
converting it to Kannala-Brandt (OpenCV `cv2.fisheye`'s model).
"""
from ds_msp import DoubleSphereModel, KannalaBrandtModel, convert


def main() -> None:
    ds = DoubleSphereModel.sample()        # the bundled DS calibration (no file needed)

    kb, report = convert(ds, KannalaBrandtModel, width=1920, height=1080)
    print(f"rms_px={report['rms_px']:.5f}")
    print(f"converged={report['converged']}")


if __name__ == "__main__":
    main()
