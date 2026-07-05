"""How-to: convert to a different target model -- same call, new class.

Any model in the library is a valid `target_class`. This converts the same
bundled Double Sphere calibration to EUCM instead of Kannala-Brandt.
"""
from ds_msp import DoubleSphereModel, EUCMModel, convert


def main() -> None:
    ds = DoubleSphereModel.sample()

    eucm, report = convert(ds, EUCMModel, width=1920, height=1080)
    print(f"rms_px={report['rms_px']:.3f}")


if __name__ == "__main__":
    main()
