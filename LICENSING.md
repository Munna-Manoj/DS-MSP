# Licensing

DS-MSP is **MIT-licensed** — the entire library, including the DS+ camera model and the
robust from-scratch calibration/conversion engine, is free to use, modify, and redistribute
for any purpose, commercial or not. See [`LICENSE`](LICENSE) for the full text.

## Prior work

DS-MSP's design follows several external projects, cited in full in the
[README's Credits section](README.md#credits): **Fisheye-Calib-Adapter** (Sangjun Lee,
arXiv:2407.12405), **MC-Calib** (Rameau, Park, Bailo, Kweon, CVIU 2022), and the camera-model
papers each model implements (Double Sphere, UCM, EUCM, Kannala-Brandt, OCam/Scaramuzza — see
their docstrings for exact citations). DS-MSP does not vendor code from these projects; it
follows their published designs and cites them accordingly. Their own repositories remain
under their own licenses.

## History

Earlier releases (0.7.0–0.9.x) dual-licensed DS+ and the robust calibration engine under
PolyForm Noncommercial 1.0.0. That restriction has been removed — see
[ADR-0010](docs/process/architecture/decisions/ADR-0010-mit-relicense-and-eucmplus-removal.md)
for the rationale. Everything built so far is MIT; there is no noncommercial tier.
