# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.5.0](https://github.com/Munna-Manoj/DS-MSP/compare/v0.4.0...v0.5.0) (2026-06-21)


### Features

* **io:** C9 ecosystem interop — COLMAP + nerfstudio export/read ([#20](https://github.com/Munna-Manoj/DS-MSP/issues/20)) ([f46f080](https://github.com/Munna-Manoj/DS-MSP/commit/f46f08020f2f10d0c839e46283fbe57a7629c2f3))
* **vo:** Tier 2 — monocular VO core + ATE/RPE evaluation toolkit ([#22](https://github.com/Munna-Manoj/DS-MSP/issues/22)) ([eb6448c](https://github.com/Munna-Manoj/DS-MSP/commit/eb6448c5260751d764bbe009974b514468ac189b))


### Documentation

* SLAM/VIO roadmap planning notes ([#23](https://github.com/Munna-Manoj/DS-MSP/issues/23)) ([ba1d95d](https://github.com/Munna-Manoj/DS-MSP/commit/ba1d95dbd7e90c721fb8ba70bb93ec7a1d568500))
* roadmap Tiers 2–4 (VO · VIO · external 3D-reconstruction) + finish Tier 1 with C9 ([#19](https://github.com/Munna-Manoj/DS-MSP/issues/19)) ([e638082](https://github.com/Munna-Manoj/DS-MSP/commit/e638082211b3cefc45fef6551b0ce696bdf9b29b))

## [0.4.0](https://github.com/Munna-Manoj/DS-MSP/compare/v0.3.0...v0.4.0) (2026-06-21)

**Tier-1 — from one calibrated camera to 3D structure.** This release adds the full multi-view
geometry stack: two-view pose on bearing vectors with robust RANSAC, end-to-end relative-pose
estimation, angular bundle adjustment, sphere-sweep stereo depth, and spherical rectification —
all on a new in-house manifold (SO(3)/SE(3)) Levenberg–Marquardt solver with Schur-complement
sparse BA. It also lands stereo-extrinsic calibration validated against TUM-VI's published rig
to ~0.06°, and a verified, figure-rich learning chapter for it.

### Features

* **calib:** stereo extrinsic calibration on TUM-VI, validated vs published (Tier 1) ([5f8163d](https://github.com/Munna-Manoj/DS-MSP/commit/5f8163d3f1828b11e01a6505480243b3481147c6))
* **core,calib:** Schur-complement sparse BA for calibration ([2a833e9](https://github.com/Munna-Manoj/DS-MSP/commit/2a833e99edb4c211c4db7e89455ccea897bd9bcf))
* **core:** Phase 2 — in-house manifold LM solver (fast + robust Lie) ([491e90d](https://github.com/Munna-Manoj/DS-MSP/commit/491e90d5041a8dab73a889ccc9204f80ec568347))
* **lie:** Phase 1 — manifold-correct pose optimization (SO(3)/SE(3) on the manifold) ([85dc7b3](https://github.com/Munna-Manoj/DS-MSP/commit/85dc7b35fc2fd02d7a55fdae9562d3473b296025))
* **mvg,stereo:** Tier-1 C5 angular two-view BA + C6 spherical rectification ([28b3306](https://github.com/Munna-Manoj/DS-MSP/commit/28b33065dfd5d4e456083366b849d1f2d164e702))
* **mvg:** estimate_relative_pose — end-to-end robust two-view pose ([23987ef](https://github.com/Munna-Manoj/DS-MSP/commit/23987ef366f4abde47a96f31c6568550abff25ce))
* **mvg:** Tier-1 C1 — two-view geometry on bearing vectors ([27aba96](https://github.com/Munna-Manoj/DS-MSP/commit/27aba96a8f02c50ac050f1e7187abf61187b5406))
* **mvg:** Tier-1 C2 — robust relative pose (RANSAC + spherical whitening) ([02fb0c5](https://github.com/Munna-Manoj/DS-MSP/commit/02fb0c5741dfa0d4c74d7ee8c21c82c633fca150))
* **ops:** Tier-1 C3 — chart reprojection library (sphere/cylinder/pinhole/cubemap/tangent) ([a8ea7ad](https://github.com/Munna-Manoj/DS-MSP/commit/a8ea7adf594d976d5ae91bf51c64b52b3aaae54c))
* **stereo:** Tier-1 C4 — sphere-sweep stereo (depth on raw fisheye, no rectification) ([5243693](https://github.com/Munna-Manoj/DS-MSP/commit/5243693ac5cbd70ca45f4e45b96d9f7a3adff427))


### Bug Fixes

* **detect:** multi-scale + board-guided AprilGrid detection for the fisheye periphery ([8ad0369](https://github.com/Munna-Manoj/DS-MSP/commit/8ad03695a62d5f85be18dd4270f529225369c97c))


### Documentation

* add 3D pipeline render (colourful world → fisheye), verified exact ([17f61d8](https://github.com/Munna-Manoj/DS-MSP/commit/17f61d82d7d640875fe7def5ddde4886e0dfde79))
* capture Tier-1 representation research as implementation-ready spec + roadmap ([65e5e3c](https://github.com/Munna-Manoj/DS-MSP/commit/65e5e3c9d23440b32f3e2a4b0b189b9370432d89))
* **learn:** Chapter 3 — projection validity & the &gt;180° cone (rescues original assets) ([a635829](https://github.com/Munna-Manoj/DS-MSP/commit/a635829c90bcafecb81607c0856c3bf3d05e2a2f))
* **learn:** clarity pass + visuals (GIFs from TUM-VI data, Mermaid) to standard ([bc27a5f](https://github.com/Munna-Manoj/DS-MSP/commit/bc27a5f512c68fb6eaa5bae78e80df28dac4f647))
* **learn:** stereo extrinsics chapter + invariance figure ([b5ceade](https://github.com/Munna-Manoj/DS-MSP/commit/b5ceadeb1240a24cc79efbf994c369a7faef6113))
* point-by-point Double Sphere pipeline render ([5b53f63](https://github.com/Munna-Manoj/DS-MSP/commit/5b53f639281d841eaeb3380ee312167f9abaad1e))
* prove conversion math with checkerboard corners across all 4 representations ([9ee9659](https://github.com/Munna-Manoj/DS-MSP/commit/9ee9659dde1139c6b64d5add0f46b4b92e82effd))
* **readme:** pip install ds-msp + PyPI badge (v0.3.0 published) ([e3590a5](https://github.com/Munna-Manoj/DS-MSP/commit/e3590a5733ff96af2b13fb9b4e5337bf249ff7cb))
* **readme:** surface fisheye image-formation + sphere/cylinder/pinhole visuals ([b211ac1](https://github.com/Munna-Manoj/DS-MSP/commit/b211ac15207b49e65a5bb4473d61ef392c942eaf))
* redesign DS render as a verified 2D cross-section (clearer + provably exact) ([c0adbcf](https://github.com/Munna-Manoj/DS-MSP/commit/c0adbcf4eb1090b3fb02e83e692ac2f980554c60))
* **research:** DS-MSP &lt;-&gt; diffpnp symbiosis — survey, gap analysis, phased plan ([6913fd6](https://github.com/Munna-Manoj/DS-MSP/commit/6913fd62020a42710ada539d5a05cdaede10f1dc))
* simulation-studio workflow + first WebGL render (Double Sphere projection) ([80bfab0](https://github.com/Munna-Manoj/DS-MSP/commit/80bfab00961ce49c22df77134a38ac135548d24a))
* simulation-studio workflow + first WebGL render (Double Sphere projection) ([e7148d0](https://github.com/Munna-Manoj/DS-MSP/commit/e7148d043b203ed3b17aac204557d6eaab0a49c2))
* sphere/cylinder/pinhole reprojection deep-dive + verified pixel maps ([300ce30](https://github.com/Munna-Manoj/DS-MSP/commit/300ce309878ef0e2f489e8761bb41553e7c2fabe))
* surface Tier-1 in curriculum nav + learning-docs audit ([138f0d8](https://github.com/Munna-Manoj/DS-MSP/commit/138f0d8d6f06e3a898f53812bf8fdd4bccf5c300))
* turn AprilGrid detection findings into learning material; refresh calib numbers ([620b066](https://github.com/Munna-Manoj/DS-MSP/commit/620b066a64b41bee787003651f39301f0acfe09f))

## [0.3.0] — 2026-06-20

First public, CI-tested, PyPI-ready release.

### Added
- **Calibration from real images.** `ds_msp.calib.detect_aprilgrid` (AprilGrid detection
  adapter, optional `[calib]` extra) + `AprilGridTarget` (board geometry); a robust loss
  (`loss=` / `f_scale=`) for `ds_msp.calib.calibrate`.
- **Learning curriculum** (`docs/learn/`) with five runnable examples on real TUM-VI data,
  including the calibration capstone (detect → bundle-adjust → match the published reference).
- **Continuous integration** — GitHub Actions running `ruff` + `import-linter` + `mypy` +
  `pytest` on Python 3.10–3.12, plus README badges.
- **Benchmarks** — `benchmarks/benchmark.py`: accuracy vs OpenCV (~1e-13 px) and the
  analytic-vs-finite-difference Jacobian speedup.
- **Dataset guide** (`datasets/README.md`) mapping each roadmap tier to its data.
- **Packaging metadata** — license (MIT), classifiers, project URLs, keywords; this CHANGELOG.

### Changed
- Minimum Python is now **3.10** (the NumPy/SciPy stack requires it).
- README refactored into a structured, guided page.

### Fixed
- `ds_msp.model` now correctly re-exports `ds_project_jacobian` (was referenced by
  `calibrate.py` and the README but missing from the re-export list).
- `import-linter` contract for service-layer independence (`ops`/`adapt`/`calib`).
- §7.2 README math now renders correctly on GitHub.

## [0.2.0]

- Multi-model camera library (UCM, EUCM, Kannala-Brandt, RadTan, OCamCalib) behind one
  `CameraModel` contract, with model conversion and Kalibr YAML I/O.
- `pip install -e .` packaging fix (setuptools flat-layout discovery).
- Double Sphere core with the correct `> 180°` half-space projection validity, analytic
  Jacobians, OpenCV-compatible API, and TI Jacinto LDC export.

[0.3.0]: https://github.com/Munna-Manoj/DS-MSP/releases/tag/v0.3.0
