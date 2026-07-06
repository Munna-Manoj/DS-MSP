"""How-to: restrict the FOV before converting to a narrower target model.

Pinhole-style models (RadTan) cannot represent a >180 deg fisheye. Converting
one without limiting the field of view lets rays the target can never
reproduce drag down the fit. `max_fov_deg` fits and reports only the
representable cone instead.
"""
from ds_msp import DoubleSphereModel, RadTanModel, convert


def main() -> None:
    ds = DoubleSphereModel.sample()

    rt, report = convert(ds, RadTanModel, width=1920, height=1080, max_fov_deg=120.0)
    print(f"rms_px={report['rms_px']:.3f}")
    print(f"fov_covered_deg={report['fov_covered_deg']:.1f}")


if __name__ == "__main__":
    main()
