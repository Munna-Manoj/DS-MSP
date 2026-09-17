# `ds_msp.ldc`

Lens-distortion-correction mesh export — for embedded/ISP undistortion pipelines that need a
precomputed remap mesh rather than a per-frame closed-form call. Works with **any**
`CameraModel`: the generator uses only `project()` and `K`, so every registered model and any
model added later exports without changes here. See
[Export an LDC mesh](../how-to/export_ldc_mesh.md); for the renderer-side counterpart see
[Isaac Sim LUT export](isaac_sim.md).

::: ds_msp.ldc
