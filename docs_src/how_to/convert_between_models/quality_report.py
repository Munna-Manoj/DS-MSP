"""How-to: read every field of convert()'s quality report.

Repeats the same Double Sphere -> Kannala-Brandt conversion as
`convert_in_one_call.py`, then reads the rest of the report. Always check it --
some conversions are lossy, and the report is how you catch that; a conversion
never fails silently.
"""
from ds_msp import DoubleSphereModel, KannalaBrandtModel, convert


def main() -> None:
    ds = DoubleSphereModel.sample()
    kb, report = convert(ds, KannalaBrandtModel, width=1920, height=1080)

    print(f"median_px={report['median_px']:.5f}")   # typical pixel error
    print(f"max_px={report['max_px']:.5f}")          # worst-case pixel error
    print(f"fov_covered_deg={report['fov_covered_deg']:.1f}")
    print(f"source_model={report['source_model']!r}")
    print(f"target_model={report['target_model']!r}")


if __name__ == "__main__":
    main()
