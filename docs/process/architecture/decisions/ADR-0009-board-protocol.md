# ADR-0009 — One `Board` protocol unifies checkerboard / ChArUco / AprilGrid for single-camera calibration

- **Status:** Accepted (recorded 2026-07-04)
- **Deciders:** maintainer
- **Relates to:** ARC-CALIB, ARC-DETECT, ARC-MODELS, ARC-RIG
- **Supersedes:** —

## Context

Single-camera intrinsics calibration had no unified front end: ChArUco detection
(`ds_msp/detect/charuco.py`) is function-based and returns `(board_id, corner_ids, pts_2d)`
tuples; AprilGrid (`ds_msp.calib.targets.AprilGridTarget`) is class-based and returns
`(X_world_list, keypoints_list, visibility_list)` via `build_correspondences`; plain
checkerboard detection didn't exist at all (no `cv2.findChessboardCorners`/`SB` call anywhere
in the repo). The only single-camera calibration entry point was a hardcoded, DS-only script
reading hand-labeled COCO-JSON corners — a one-off workaround for a fisheye camera where the
classic checkerboard detector failed under distortion — with its own duplicate least-squares
loop that didn't even use the already-existing, already-tested, model-agnostic backend
(`ds_msp.calib.bundle.calibrate`, already reused by `ds_msp.rig` for per-camera intrinsics).

The goal: one config-driven single-camera CLI where board type, board geometry, and camera
model are just configuration, and detection is a swappable front end feeding the one shared
backend — "front end differs, backend and signature are the same."

## Decision

1. **A single-method structural protocol**, `ds_msp.calib.board.Board`:

   ```python
   @runtime_checkable
   class Board(Protocol):
       def detect(self, image_paths: Sequence[str]) -> List[Observation]: ...
   ```

   reusing `ds_msp.data.observations.Observation` (`points_3d`, `pixels`, `visibility`,
   `cam_id`, `frame_id`) — already present, already documented as "the atomic unit both
   single-camera and rig calibration build on" — rather than inventing a new correspondence
   type. `to_correspondences(obs) -> (X_world_list, keypoints_list, visibility_list)` is the
   one, trivial, board-agnostic function converting a `List[Observation]` into
   `bundle.calibrate`'s three-list input.

   A structural `Protocol` (not an ABC/base class) follows the precedent `ADR-0002` already
   set for `CameraModel`: looser coupling, no forced inheritance hierarchy, satisfied by any
   class with the right method shape.

2. **Every board type natively implements `Board`** — no adapter wrapping a pre-existing
   function's output shape after the fact. `CheckerboardBoard`, `CharucoBoard`, and
   `AprilGridBoard` (all in `ds_msp/calib/board.py`) call the *low-level*, per-image detection
   primitives directly (`detect_corners`, `detect_image`, `detect_aprilgrid`) and build
   `Observation`s inline. This is a deliberate choice over the lower-risk "thin adapter"
   alternative: it makes all three board types structurally identical end to end, not just at
   the adapter seam, at the cost of touching more surface than a pure wrapper would.

   Consequence, verified rather than assumed: **zero changes to any existing, tested,
   rig-facing detection code.** `ds_msp/detect/charuco.py`'s `detect_folder`/`detect_rig`/
   `single_board_object` (used by `ds_msp.rig`) and `AprilGridTarget.build_correspondences`
   (used by `scripts/make_learn_gifs.py`) are untouched — the native `Board` implementations
   are new, additive code sitting alongside them, reusing only the already-public, per-image
   primitives (`detect_image`, `board_object_points`, `make_detectors`, `detect_aprilgrid`,
   `AprilGridTarget.object_points`) those existing functions were themselves already built on.

3. **The `Board` implementations live in `ds_msp/calib/board.py`, not `ds_msp/detect/`.** The
   import-linter contract already forbids `ds_msp.detect` from importing `ds_msp.calib` (the
   reverse — `calib → detect` — is legal and already an existing edge). `AprilGridBoard` needs
   both `AprilGridTarget` (`ds_msp.calib.targets`) and `detect_aprilgrid`
   (`ds_msp.detect.detect`); only `calib` may legally import both. Keeping all board-type
   decisions in one file also avoids scattering "which boards exist" logic across `detect/`.

4. **Single-camera calibration needs no rigid board fusion.** `ds_msp.rig`'s multi-board
   support (`Object3D`/`ObjectObs`, board-group reconstruction) solves a genuinely harder
   problem: many boards rigidly fixed to each other, seen by many cameras, sharing one 3D point
   cloud. Single-camera calibration only needs many independent *views* of a *known* planar
   pattern — the boards seen across views need no rigid relationship to each other at all. This
   is why `CharucoBoard` supports multiple simultaneous board definitions
   (`make_detectors`/`detect_image` are already multi-board-aware) with zero fusion machinery:
   each board actually detected in an image becomes its own independent `Observation`, whether
   it appears alone or alongside other boards in the same frame. `AprilGridBoard` slots into
   the identical pattern. **`ds_msp.rig` is unmodified and stays ChArUco-only** — this
   unification is scoped entirely to single-camera calibration.

5. **`seed_from_K` moves from `ds_msp.rig.calibrate` (private `_seed_from_K`) to
   `ds_msp.models.registry.seed_from_K`, not `ds_msp.geometry.resection`.** Both `ds_msp.calib`
   (new) and `ds_msp.rig` (existing) need "instantiate a from-scratch seed of `model_cls` from
   a pinhole `K`," and `ds_msp.calib` cannot import `ds_msp.rig` (capability → pipeline is an
   illegal edge). `ds_msp.geometry.resection` — where `intrinsics_seed`/`ransac_pnp_normalized`
   already live — is the file-level "obvious" choice, but `ADR-0008` enumerates it as one of
   exactly three files under `PolyForm-Noncommercial-1.0.0` (`geometry/resection.py`,
   `calib/bundle.py`, `adapt/convert.py`). Moving a trivial, mechanical
   `model_cls.from_params(...)`-with-neutral-defaults helper there would silently convert it
   from MIT to noncommercial purely by file placement — the exact scope creep `ADR-0008`
   argues against in the opposite direction (it names the *robust engine* as the moat, not
   incidental plumbing). `ds_msp.models` is explicitly enumerated as staying MIT in that same
   ADR, the function touches no cv2/scipy (consistent with `ADR-0004`), and `model_class(name)`
   already lives in `registry.py` — natural co-location.

## Consequences

**Positive**
- One board-agnostic driver, `ds_msp.calib.single_camera.calibrate_camera(board, image_paths,
  model_name, width, height, **calib_kwargs)`, is the literal realization of "front end
  differs, backend is the same" — three lines regardless of board type.
- `ds_msp.rig`'s real-data release gate is unaffected by construction (nothing it depends on
  changed), not by careful avoidance during review.
- Adding a fourth board type (e.g. circles-grid) is a documented, mechanical extension: new
  detection in `ds_msp/detect/<board>.py`, a native `Board`-satisfying class in
  `ds_msp/calib/board.py`, wired into config — see
  `docs/process/playbooks/add-a-detection-board.md`.

**Negative / costs**
- More upfront surface than the thin-adapter alternative would have needed: three real
  `Board` implementations, each independently tested, rather than three thin wrappers around
  already-tested convenience functions.
- `CharucoBoard`'s "every image is decoded independently" flow duplicates a small amount of
  per-image OpenCV plumbing (`cv2.imread`, grayscale loop) that also exists in
  `detect_folder`/`_detect_one_image` — accepted, since those two functions serve genuinely
  different callers (multi-board-fused rig geometry vs. independent single-camera views) with
  different enough contracts that sharing the top-level loop would couple them unnecessarily.

## Scope explicitly deferred (not accidental omissions)

- **Checkerboard is single-board only** in v1 — no marker-ID system to disambiguate which
  physical board is which (unlike ChArUco/AprilGrid), so multi-checkerboard would need new
  design work, not a mechanical extension.
- **AprilGrid is single-target only** in v1, matching `AprilGridTarget` itself (inherently one
  grid).
- **No "undistort first, detect, map back" checkerboard strategy.** A real, documented
  technique for pushing checkerboard detection further into the fisheye periphery, but it
  needs a prior distortion/FOV estimate before any calibration exists — a genuine
  chicken-and-egg constraint. Not built now: `Board.detect()` takes only `image_paths`, so
  adding a future `CheckerboardBoard(strategy="undistort_first", prior_model=...)` constructor
  path is a non-breaking addition later, not an interface redesign.
- **Plain checkerboard is not claimed to be fisheye-robust.** `cv2.findChessboardCornersSB`
  (Duda & Frese, "Accurate Detection and Localization of Checkerboard Corners for
  Calibration," BMVC 2018) is a real, documented improvement in noise/blur robustness and
  sub-pixel accuracy over the classic detector, but is not documented or benchmarked as
  distortion-specific. Verified directly against both reference tools this project already
  studies: MC-Calib (`ds_msp.rig`'s own upstream reference) has no plain-checkerboard board
  type at all — only ChArUco/AprilTag; Kalibr does support checkerboard but its own docs
  recommend AprilGrid instead specifically for this failure mode. `ds-msp-calibrate`'s
  checkerboard help text carries the same guidance.

## Alternatives considered

- *Thin adapters wrapping existing high-level functions (`detect_folder`,
  `build_correspondences`) instead of native implementations.* Lower risk (even less existing
  code touched) and was the initial recommendation; rejected by the maintainer in favor of full
  native implementation for structural consistency across all three board types.
- *Put `Board` implementations in `ds_msp/detect/`.* Rejected — would require `ds_msp.detect`
  to import `ds_msp.calib.targets` (`AprilGridTarget`), violating the existing import-linter
  direction and creating a circular capability dependency.
- *Move `seed_from_K` into `ds_msp/geometry/resection.py`.* Rejected — would silently
  relicense a trivial MIT helper as noncommercial purely by file placement; see Decision §5.
- *Extend `ds_msp.rig` to also support AprilGrid/checkerboard, for full platform symmetry.*
  Rejected (maintainer's explicit scope call) — rig's multi-board need is rigid fusion across
  many cameras, a fundamentally harder and different problem than single-camera's independent
  views; extending it now would be substantial new scope disconnected from this unification's
  actual goal.
