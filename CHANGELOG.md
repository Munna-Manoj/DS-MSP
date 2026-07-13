# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.12.0](https://github.com/Munna-Manoj/DS-MSP/compare/v0.11.1...v0.12.0) (2026-07-13)


### Features

* **rig:** robust reporting + robust live-view depth for down-weighted outliers ([cfe3385](https://github.com/Munna-Manoj/DS-MSP/commit/cfe3385c8646344dbc4d8b3fcffaa09c304da05d))


### Bug Fixes

* **rig:** reject gross-outlier board detections via ransac_threshold gate ([22e3b02](https://github.com/Munna-Manoj/DS-MSP/commit/22e3b0210660d37abd170dd9589bbab2a8599481))
* **rig:** robust gross-outlier handling for board detections (FR-RIG-018) ([070531c](https://github.com/Munna-Manoj/DS-MSP/commit/070531c171b3e40f41df00b1d62034cbbf708f26))

## [0.11.1](https://github.com/Munna-Manoj/DS-MSP/compare/v0.11.0...v0.11.1) (2026-07-11)


### Bug Fixes

* **docs:** remove stray file from docs/; add CI top-level allowlist g… ([e689980](https://github.com/Munna-Manoj/DS-MSP/commit/e689980a4cec35fc2c366c036bb21c65375df224))
* **docs:** remove stray file from docs/; add CI top-level allowlist gate (NFR-DOCS-003) ([1ba925b](https://github.com/Munna-Manoj/DS-MSP/commit/1ba925b455feaa4e0711b63918c68ab43d222f1d))

## [0.11.0](https://github.com/Munna-Manoj/DS-MSP/compare/v0.10.0...v0.11.0) (2026-07-08)


### Features

* **rig:** multi-object merge — calibrate non-overlapping rigs via hand-eye ([c373c32](https://github.com/Munna-Manoj/DS-MSP/commit/c373c32b6ba4f354108966d6f2e212cdeb5f7a90))


### Bug Fixes

* **rig:** multi-object merge for non-overlapping rigs (FR-RIG-017, ADR-0011) ([a04f6cc](https://github.com/Munna-Manoj/DS-MSP/commit/a04f6cccd02dd971d3d47a3068570fcbf5255e97))
* **rig:** rebind live-view scene after object merge; close SEMS gate for FR-RIG-017 ([47effd2](https://github.com/Munna-Manoj/DS-MSP/commit/47effd219b934a7ab7f4b08c0c29c645f6af7a27))

## [0.10.0](https://github.com/Munna-Manoj/DS-MSP/compare/v0.9.1...v0.10.0) (2026-07-06)


### ⚠ BREAKING CHANGES

* **license:** the project is no longer available under PolyForm-Noncommercial for any part of the library — everything, including DS+ and the robust engine, is plain MIT.

### Features

* **rig:** embed a real pond-replay of rig calibration in the guide ([a6caa9e](https://github.com/Munna-Manoj/DS-MSP/commit/a6caa9e0f7fc012a6114996d8ac2c05c239467a7))


### Bug Fixes

* **docs:** point ADR-0009/0010 references at GitHub, not excluded paths ([51094a5](https://github.com/Munna-Manoj/DS-MSP/commit/51094a5fccd378902177f3a06c34ee0dabffeeb1))
* **rig:** stop _try_load_object silently reusing a stray save_path object ([90592ac](https://github.com/Munna-Manoj/DS-MSP/commit/90592ace25c06469ed542c2b02610e5c870aa279))
* **tests:** make docs_src float-noise assertions platform-robust ([41b282d](https://github.com/Munna-Manoj/DS-MSP/commit/41b282d4661fc15795a85d8756238674036e266c))
* **tests:** make remaining docs_src float-noise assertions platform-robust ([ef0e89a](https://github.com/Munna-Manoj/DS-MSP/commit/ef0e89ad784b593390206eb1358d69d3c7fcd55a))
* **tests:** make remaining docs_src float-noise assertions platform-robust ([18a93d3](https://github.com/Munna-Manoj/DS-MSP/commit/18a93d3471d5c3c49bb6b4c3cc7896212c943a02))


### Documentation

* docs_src/ scaffolding + governance, math-formatting house rule ([abc7f36](https://github.com/Munna-Manoj/DS-MSP/commit/abc7f3690402b0952595247b813e240751ab5112))
* **style:** adopt Typer's exact palette, typography, and admonition syntax ([1ec66b0](https://github.com/Munna-Manoj/DS-MSP/commit/1ec66b0060e7d04edde2fc49130de86478d5fa7c))
* **style:** fix inline math and long paragraphs in two-view geometry chapter ([97ac2ee](https://github.com/Munna-Manoj/DS-MSP/commit/97ac2ee491452da0a38b51836c0da60c721a1847))
* **style:** Phase C — restyle all 10 Learn-track pages to Typer format ([460a140](https://github.com/Munna-Manoj/DS-MSP/commit/460a14050bc7768e5cfdffb88abe6116a09f0953))
* **style:** Phase D — restyle all How-to (7) + Explain (6) pages to Typer format ([a7dcb70](https://github.com/Munna-Manoj/DS-MSP/commit/a7dcb70a17df3f6cde31109dc33cceca7d336956))
* **style:** Phase E — restyle root guides, docs/index.md, and README.md ([0f32e52](https://github.com/Munna-Manoj/DS-MSP/commit/0f32e525ceaffe034e461c712095b970639f624d))
* **style:** split paragraphs over the 40-word house limit ([c31fc22](https://github.com/Munna-Manoj/DS-MSP/commit/c31fc2286d488d323bf0578736e2b97bfc1759e0))
* **style:** wire up Termynal animated-terminal blocks ([bdb7483](https://github.com/Munna-Manoj/DS-MSP/commit/bdb74835e0ee2aa6702322e2aba974c20f3a0423))
* Typer-style restyle + rig calibration fixes and live pond-replay demo ([c6364a3](https://github.com/Munna-Manoj/DS-MSP/commit/c6364a3c59c3d129ea4581e7e70ccc435f3dfe64))
* update for EUCM+ removal and MIT relicense ([21afc65](https://github.com/Munna-Manoj/DS-MSP/commit/21afc653e14eba61a6ca826adddf74cadf2f34cc))


### Chores

* **license:** relicense DS-MSP to plain MIT ([5f1b24a](https://github.com/Munna-Manoj/DS-MSP/commit/5f1b24a814ef9c7937c0cf270581e9ae9debd36d))

## [Unreleased]

### ⚠ BREAKING CHANGES

* **models:** remove the EUCM+ camera model — `EUCMPlusModel`, the `"eucmplus"`/`"eucm+"` registry aliases, and Kalibr `eucm_plus` I/O are gone (ADR-0010).
* **license:** relicense the whole project to plain MIT — drops the PolyForm Noncommercial 1.0.0 tier that previously covered DS+ and the robust calibrate/convert engine (ADR-0010).

## [0.9.1](https://github.com/Munna-Manoj/DS-MSP/compare/v0.9.0...v0.9.1) (2026-07-05)


### Bug Fixes

* **ci:** add .nojekyll to the merged Pages artifact; retrigger a stuck deploy ([#44](https://github.com/Munna-Manoj/DS-MSP/issues/44)) ([7421759](https://github.com/Munna-Manoj/DS-MSP/commit/7421759696e5e54de3157f2e5ccfe63acb03135e))

## [0.9.0](https://github.com/Munna-Manoj/DS-MSP/compare/v0.8.0...v0.9.0) (2026-07-01)


### Features

* ship the multi-camera rig (robust-by-default), RANSAC PnP, task-oriented README ([#41](https://github.com/Munna-Manoj/DS-MSP/issues/41)) ([b5f4cae](https://github.com/Munna-Manoj/DS-MSP/commit/b5f4caea585746478b1870b363169b1fe8b9ee4c))

## [0.8.0](https://github.com/Munna-Manoj/DS-MSP/compare/v0.7.1...v0.8.0) (2026-06-29)


### Features

* **rig:** MC-Calib-compatible intrinsics handling, keypoints reuse, docs + SEMS ([#38](https://github.com/Munna-Manoj/DS-MSP/issues/38)) ([4dc33b0](https://github.com/Munna-Manoj/DS-MSP/commit/4dc33b02557d82961fc45ca9f9c0aa24c8ef0cd8))

## [0.7.1](https://github.com/Munna-Manoj/DS-MSP/compare/v0.7.0...v0.7.1) (2026-06-29)


### Documentation

* **learn:** measurable camera-model evaluation framework + EUCM⁺/DS⁺/KB case study ([#35](https://github.com/Munna-Manoj/DS-MSP/issues/35)) ([04b8026](https://github.com/Munna-Manoj/DS-MSP/commit/04b80266628fde2735a4405ce27114a2b9676fc8))

## [0.7.0](https://github.com/Munna-Manoj/DS-MSP/compare/v0.6.0...v0.7.0) (2026-06-28)


### ⚠ BREAKING CHANGES

* **release:** DS+ and EUCM+ are no longer MIT-licensed — from 0.7.0 they are PolyForm Noncommercial 1.0.0 (noncommercial use only, with attribution to Munna-Manoj). Commercial use of DS+/EUCM+ requires a separate license. The rest of the library stays MIT.

### Features

* **calib:** model-aware multi-start auto-init + robustness test suite (NFR-NUM-006) ([8db0f46](https://github.com/Munna-Manoj/DS-MSP/commit/8db0f46a2dc86d85cd277b6b769a0834273b7873))
* **calib:** robust-by-default calibrate() — two-fold seeding, auto-scale, honest stats ([fb4d263](https://github.com/Munna-Manoj/DS-MSP/commit/fb4d2637a0a5db10705c5599a4ce052b67e8d045))


### Bug Fixes

* **adapt:** deterministic shape-parameter sweep in convert() (ADR-0007, NFR-NUM-007) ([e99af32](https://github.com/Munna-Manoj/DS-MSP/commit/e99af32f3b57e90b0e182fdb0a587dcda3c9ac30))
* **calib:** pose seeding no longer crashes on sparse/degenerate views ([1a54ceb](https://github.com/Munna-Manoj/DS-MSP/commit/1a54cebb17ffa71e48e48b9a90466b6fa520a26e))
* **core:** type annotations in robust.py for the typed-core mypy gate ([afc3529](https://github.com/Munna-Manoj/DS-MSP/commit/afc352963c58358d66c28e6d981a0c78b9e78cc9))
* **docs:** correct license attribution and test counts for 0.7.0 publication ([bbd7933](https://github.com/Munna-Manoj/DS-MSP/commit/bbd79333eae1f624fc349622ad312560db90a564))
* **lint:** clear pre-existing ruff debt in rig/io/scripts ([758ac62](https://github.com/Munna-Manoj/DS-MSP/commit/758ac628337a64f6f8ebbf64f2b56f0fac8ffd24))
* **process:** address leak-guard audit of the SEMS docs ([863e246](https://github.com/Munna-Manoj/DS-MSP/commit/863e2466c620cf19691e27c12b7b6988ddd24333))
* **rig:** remove pre-existing publication leaks from rig content ([075af65](https://github.com/Munna-Manoj/DS-MSP/commit/075af6540550c3882b4c07cdb48e9ea9e21b8ee1))
* **tests:** put repo root on sys.path for cross-package test helpers ([8c8db2f](https://github.com/Munna-Manoj/DS-MSP/commit/8c8db2f3db6dac71af79742a417a38299ad2425c))


### Documentation

* **process:** SEMS P2–P3 — architecture description, ADRs, SRS, interfaces ([a1b0f36](https://github.com/Munna-Manoj/DS-MSP/commit/a1b0f36dabb78e1780dc3a5d7a03bf2e91cb3140))
* **process:** SEMS P4–P5 — QA/V&V, DoD, CI/CD, management process, playbooks ([c49a1f4](https://github.com/Munna-Manoj/DS-MSP/commit/c49a1f4df41b45400e4388de3fc713bea419a9c5))
* **process:** SEMS P6 — contribution wiring, governance CI job, handbook ([a0c4131](https://github.com/Munna-Manoj/DS-MSP/commit/a0c4131a803439026b1e197e8f0e1c27a080cba3))


### Chores

* **release:** prepare 0.7.0 — dual-license DS+/EUCM+, descope rig to 0.8.0 ([c3c72b5](https://github.com/Munna-Manoj/DS-MSP/commit/c3c72b5ecd033b38a0e2144bfaa5a82fcdb4d933))

## [0.6.0](https://github.com/Munna-Manoj/DS-MSP/compare/v0.5.0...v0.6.0) (2026-06-27)


### Features

* **models:** add DS+ and EUCM+ closed-form-invertible camera models ([#28](https://github.com/Munna-Manoj/DS-MSP/issues/28)) ([8c9d519](https://github.com/Munna-Manoj/DS-MSP/commit/8c9d51918abfe8d07b59ecbf1239f292c4a8d21e))
* **web:** interactive multi-model camera studio + Pages deploy ([4940cac](https://github.com/Munna-Manoj/DS-MSP/commit/4940cac540ef1b3837e0547e95f3e1e2063827e9))
* **web:** interactive multi-model camera studio + Pages deploy ([771d3da](https://github.com/Munna-Manoj/DS-MSP/commit/771d3daa624d12dd85c44903dfa4db737a22698c))


### Bug Fixes

* **web:** add DS+/EUCM+ projection formulas to the play stepper ([#30](https://github.com/Munna-Manoj/DS-MSP/issues/30)) ([3e23216](https://github.com/Munna-Manoj/DS-MSP/commit/3e2321603afeb8be826e861ac04edb5b26fc02d0))
* **web:** upright synthesized image + clearer projection-step animation ([#29](https://github.com/Munna-Manoj/DS-MSP/issues/29)) ([fcd733e](https://github.com/Munna-Manoj/DS-MSP/commit/fcd733ee5fd6347e31ede2729c460fa6b7ec76ab))


### Documentation

* calibration how-to + reframe DS-MSP as a spherical-camera platform ([#32](https://github.com/Munna-Manoj/DS-MSP/issues/32)) ([d44ba49](https://github.com/Munna-Manoj/DS-MSP/commit/d44ba49b03befa928de8e4ee7bcdd5733bcfabde))
* Double Sphere GIFs show both image planes, slower & smoother ([#31](https://github.com/Munna-Manoj/DS-MSP/issues/31)) ([997b390](https://github.com/Munna-Manoj/DS-MSP/commit/997b39012124f6a49b5e109f01ea6e86ab1c9e6c))
* keep the public tree developer-facing; move internal notes local-only ([#33](https://github.com/Munna-Manoj/DS-MSP/issues/33)) ([de1517b](https://github.com/Munna-Manoj/DS-MSP/commit/de1517b14ce9a4c97d087391b420f3d1b993613f))
* remove internal/conversational content from public docs ([#24](https://github.com/Munna-Manoj/DS-MSP/issues/24)) ([f00f413](https://github.com/Munna-Manoj/DS-MSP/commit/f00f4134da795a965d680e1dbd599481646c3965))

## [0.5.0](https://github.com/Munna-Manoj/DS-MSP/compare/v0.4.0...v0.5.0) (2026-06-21)


### Features

* **io:** ecosystem interop — COLMAP + nerfstudio export/read ([#20](https://github.com/Munna-Manoj/DS-MSP/issues/20)) ([f46f080](https://github.com/Munna-Manoj/DS-MSP/commit/f46f08020f2f10d0c839e46283fbe57a7629c2f3))
* **vo:** Tier 2 — monocular VO core + ATE/RPE evaluation toolkit ([#22](https://github.com/Munna-Manoj/DS-MSP/issues/22)) ([eb6448c](https://github.com/Munna-Manoj/DS-MSP/commit/eb6448c5260751d764bbe009974b514468ac189b))


### Documentation

* SLAM/VIO roadmap planning notes ([#23](https://github.com/Munna-Manoj/DS-MSP/issues/23)) ([ba1d95d](https://github.com/Munna-Manoj/DS-MSP/commit/ba1d95dbd7e90c721fb8ba70bb93ec7a1d568500))
* roadmap Tiers 2–4 (VO · VIO · external 3D-reconstruction) + finish Tier 1 ([#19](https://github.com/Munna-Manoj/DS-MSP/issues/19)) ([e638082](https://github.com/Munna-Manoj/DS-MSP/commit/e638082211b3cefc45fef6551b0ce696bdf9b29b))

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
* **core:** in-house manifold LM solver (fast + robust Lie) ([491e90d](https://github.com/Munna-Manoj/DS-MSP/commit/491e90d5041a8dab73a889ccc9204f80ec568347))
* **lie:** manifold-correct pose optimization (SO(3)/SE(3) on the manifold) ([85dc7b3](https://github.com/Munna-Manoj/DS-MSP/commit/85dc7b35fc2fd02d7a55fdae9562d3473b296025))
* **mvg,stereo:** angular two-view BA + spherical rectification ([28b3306](https://github.com/Munna-Manoj/DS-MSP/commit/28b33065dfd5d4e456083366b849d1f2d164e702))
* **mvg:** estimate_relative_pose — end-to-end robust two-view pose ([23987ef](https://github.com/Munna-Manoj/DS-MSP/commit/23987ef366f4abde47a96f31c6568550abff25ce))
* **mvg:** two-view geometry on bearing vectors ([27aba96](https://github.com/Munna-Manoj/DS-MSP/commit/27aba96a8f02c50ac050f1e7187abf61187b5406))
* **mvg:** robust relative pose (RANSAC + spherical whitening) ([02fb0c5](https://github.com/Munna-Manoj/DS-MSP/commit/02fb0c5741dfa0d4c74d7ee8c21c82c633fca150))
* **ops:** chart reprojection library (sphere/cylinder/pinhole/cubemap/tangent) ([a8ea7ad](https://github.com/Munna-Manoj/DS-MSP/commit/a8ea7adf594d976d5ae91bf51c64b52b3aaae54c))
* **stereo:** sphere-sweep stereo (depth on raw fisheye, no rectification) ([5243693](https://github.com/Munna-Manoj/DS-MSP/commit/5243693ac5cbd70ca45f4e45b96d9f7a3adff427))


### Bug Fixes

* **detect:** multi-scale + board-guided AprilGrid detection for the fisheye periphery ([8ad0369](https://github.com/Munna-Manoj/DS-MSP/commit/8ad03695a62d5f85be18dd4270f529225369c97c))


### Documentation

* add 3D pipeline render (colourful world → fisheye), verified exact ([17f61d8](https://github.com/Munna-Manoj/DS-MSP/commit/17f61d82d7d640875fe7def5ddde4886e0dfde79))
* add the Tier-1 geometry roadmap ([65e5e3c](https://github.com/Munna-Manoj/DS-MSP/commit/65e5e3c9d23440b32f3e2a4b0b189b9370432d89))
* **learn:** Chapter 3 — projection validity & the &gt;180° cone (rescues original assets) ([a635829](https://github.com/Munna-Manoj/DS-MSP/commit/a635829c90bcafecb81607c0856c3bf3d05e2a2f))
* **learn:** clarity pass + visuals (GIFs from TUM-VI data, Mermaid) to standard ([bc27a5f](https://github.com/Munna-Manoj/DS-MSP/commit/bc27a5f512c68fb6eaa5bae78e80df28dac4f647))
* **learn:** stereo extrinsics chapter + invariance figure ([b5ceade](https://github.com/Munna-Manoj/DS-MSP/commit/b5ceadeb1240a24cc79efbf994c369a7faef6113))
* point-by-point Double Sphere pipeline render ([5b53f63](https://github.com/Munna-Manoj/DS-MSP/commit/5b53f639281d841eaeb3380ee312167f9abaad1e))
* prove conversion math with checkerboard corners across all 4 representations ([9ee9659](https://github.com/Munna-Manoj/DS-MSP/commit/9ee9659dde1139c6b64d5add0f46b4b92e82effd))
* **readme:** pip install ds-msp + PyPI badge (v0.3.0 published) ([e3590a5](https://github.com/Munna-Manoj/DS-MSP/commit/e3590a5733ff96af2b13fb9b4e5337bf249ff7cb))
* **readme:** surface fisheye image-formation + sphere/cylinder/pinhole visuals ([b211ac1](https://github.com/Munna-Manoj/DS-MSP/commit/b211ac15207b49e65a5bb4473d61ef392c942eaf))
* redesign DS render as a verified 2D cross-section (clearer + provably exact) ([c0adbcf](https://github.com/Munna-Manoj/DS-MSP/commit/c0adbcf4eb1090b3fb02e83e692ac2f980554c60))
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
