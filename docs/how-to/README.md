# How-to guides

Task-oriented recipes: each page solves one concrete problem with the library, start to finish,
no theory. For the why, follow the links to [Explanation](../explain/README.md); for API facts,
to [Reference](../reference/index.md).

Six recipes are available now:

- [Calibrate any model](calibrate_any_model.md) — recover intrinsics from a calibration board using any camera model.
- [Convert between models](convert_between_models.md) — translate a calibration from one model to another without images.
- [Undistort images](undistort_images.md) — rectify a fisheye frame to a pinhole view, controlling the FOV-vs-border trade-off.
- [Solve PnP on fisheye](solve_pnp_on_fisheye.md) — recover camera pose from 3D-to-2D correspondences on wide-FOV data.
- [Export a TI LDC mesh](export_ldc_mesh.md) — generate the displacement LUT for on-chip hardware undistortion on TI Jacinto SoCs.
- [Read/write Kalibr YAML](read_write_kalibr.md) — load a Kalibr camchain into DS-MSP and write a model back out as Kalibr YAML.

Export to COLMAP / nerfstudio — coming soon.
