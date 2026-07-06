# Deep-dive — detecting every AprilGrid tag, even at the fisheye periphery

> **Run alongside this:** the capstone (`python examples/03_calibrate_tumvi_aprilgrid.py`)
> uses the robust detector by default; the snippets below run on the same TUM-VI calib frames
> (`pip install -e .[calib]`, then `bash scripts/download_datasets.sh tumvi`).

A calibration is only as good as the corners you feed it. On a fisheye, the corners that
matter most are the hardest to detect.

The wide-angle tags near the image edge are exactly the ones that pin down the distortion,
and a naïve detector drops them.

This page is the story of finding that out on real data and fixing it, with a number at
every step. The detection logic lives in
[`ds_msp/detect/detect.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/detect/detect.py)
(`ds_msp/calib/detect.py` re-exports it for backward compatibility).

/// note
These snippets need the downloaded TUM-VI dataset (see Prerequisites) — they are
hand-verified excerpts, not standalone `docs_src/` files. Paste one into a
`python3 -c "…"` session, or save it as a script, to reproduce these numbers yourself.
///

**You'll learn**
- Why a fisheye detector can miss most of a fully-visible, sharp AprilGrid board (down to
  4/36 tags) — peripheral tags shrink below the detector's minimum-size gate — and prove it
  by upscaling alone (4 → 26 of 36 tags).
- Apply multi-scale detection (`detect_aprilgrid(..., scales=(1, 2, 3))`) and watch recall
  jump **36% → 94%** across a 12-frame spread.
- Avoid the two pitfalls that make multi-scale detection *hurt* a calibration if done
  carelessly: refining at the wrong scale (0.105 px vs. 0.87 px median reprojection) and
  mismapping upscaled corners back without the pixel-centre convention (0.03 px vs. 0.83 px
  principal-point error).
- See the payoff on the full capstone calibration: corners 5,180 → 14,472 and
  focal-length agreement 0.7% → ~0.02%.

**Prerequisites**
- Finish the [capstone](capstone_calibrating_a_real_camera.md) — this deep-dive puts its
  detection step under a microscope; the [learn README](README.md) lists it as a companion
  to read after the capstone.
- Same `[calib]` install and TUM-VI dataset as the capstone.

## 1. The symptom: most of the board goes undetected

Point the detector at TUM-VI's calib footage and count tags per frame (the board has 36):

```python
import glob
from aprilgrid import Detector
from ds_msp.detect.detect import _load_gray_u8

paths = sorted(glob.glob("datasets/tumvi/dataset-calib-cam1_512_16/mav0/cam0/data/*.png"))
det = Detector("t36h11")
for i in (158, 316, 276):                      # frames where the board is off-centre
    g = _load_gray_u8(paths[i])
    print(f"frame {i}: {len(det.detect(g))}/36 tags")
```

```
frame 158: 22/36 tags
frame 316: 4/36 tags
frame 276: 0/36 tags
```

Frame 316 is the damning one: the **entire board is in the image** — fully visible,
well-lit, no blur — sitting in the top-right corner, and the detector finds 4 of 36 tags.

This is not a data problem. You can see the tags; the detector can't.

## 2. Why it happens — and why it's *not* curved edges

The intuitive guess is "fisheye bends the tag edges, so the square-quad detector fails."
That's real, but secondary.

The dominant cause is size. A fisheye *compresses* the periphery, so a peripheral tag
shrinks to a handful of pixels across, and the detector's **minimum-cluster-size gate**
throws it out before it ever tries to decode.

The proof: simply **upscaling the image recovers the tags** — same pixels, same edges, just
bigger:

Continuing the `paths` and `det` from §1's setup:

```python
from ds_msp.detect.detect import _detect_union          # detect at several scales, union by id

g = _load_gray_u8(paths[316])
print("native     :", len(_detect_union(det, g, (1,))), "/36")        # 4
print("multi-scale:", len(_detect_union(det, g, (1, 2, 3))), "/36")   # 26
```

```
native     : 4 /36
multi-scale: 26 /36
```

4 → 26 from upscaling alone. Across a 12-frame spread the recall goes from **155/432 (36%)
to 408/432 (94%)**.

## 3. What the "robust" libraries actually do (myth vs. reality)

It's tempting to assume Kalibr and Basalt have a distortion-aware detector. They don't:

- **Kalibr** (`GridCalibrationTargetAprilgrid`) runs a per-tag C++ detector on the raw image
  and keeps only what it finds — **no recovery of missed tags**. It wins by a mature backend
  at full resolution and the fact that each tag is small enough to be *locally* near-planar.
- **Basalt** wraps AprilTag-3 (C) and aggregates **many frames** — a tag missed in one frame
  is caught in another.
- The libraries that genuinely recover hard tags are **ChArUco** (`refineDetectedMarkers` +
  corner interpolation from the board model) and **TartanCalib** (reproject expected corners
  via a partial model and re-detect locally). Both do **board-guided recovery**.

So the answer isn't a magic detector — it's feed the detector better, and use the board to
recover what's still missing. Two techniques, below.

## 4. Fix #1 — multi-scale detection (the big, cheap win)

Detect on up-scaled copies of the image and union the results by tag id. The default
`scales=(1, 2, 3)` is what took recall from 36% to 94% above:

```python
from ds_msp.calib import detect_aprilgrid
dets = detect_aprilgrid(paths, scales=(1, 2, 3))     # default; pass (1,) for the old behaviour
```

/// tip
Adding corners the wrong way makes the calibration **worse**, not better. Two subtleties
decide whether multi-scale detection helps or hurts.
///

### Subtlety A: refine at the *detection* scale, not at native resolution

`cv2.cornerSubPix` needs a search window. A window that suits a big central tag (say 5 px)
spans several corners of a tiny peripheral tag and drags it onto the wrong one.

If you detect at 3× but then refine at native resolution, you corrupt exactly the corners
you just gained:

| | median reproj | focal error |
|---|---|---|
| single-scale | 0.126 px | 0.84% |
| multi-scale, refine at **native** res | **0.87 px** ❌ | 1.7% |
| multi-scale, refine at **detection** scale | **0.105 px** ✅ | 0.7% |

The fix is to refine each tag *in the up-scaled image where it's big*, with a window sized
to the tag, and only then map the corner back
([`_subpix`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/detect/detect.py#L73-L83)
clamps the window to ~¼ the tag's shortest edge).

### Subtlety B: map corners back with the pixel-centre convention

`cv2.resize` follows a pixel-*centre* convention: a feature at native coordinate `N` lands
at `(N + 0.5)·s − 0.5` in the up-scaled image.

Inverting that naïvely as `c / s` over-shifts every corner by a constant `0.5·(1 − 1/s)`
(0.25 px at 2×, 0.33 px at 3×). A *constant* shift in all corners doesn't blur the fit — it
moves the **principal point**:

| | principal-point error |
|---|---|
| map back with `c / s` | ~0.83 px ❌ |
| map back with `(c + 0.5)/s − 0.5` | ~0.03 px ✅ |

/// tip
This one cost real debugging time: the focal length and reprojection error looked great, but
`cx, cy` were biased ~0.8 px *in the same direction* — the fingerprint of a constant offset.

A *systematic* (not random) corner error hides in the *principal point*, not the
<abbr title="Root Mean Square">RMS</abbr>.
///

## 5. Fix #2 — board-guided recovery (opt-in, for the last mile)

For tags multi-scale still misses, use the board's known geometry. Fit a *local* homography
from the nearest detected tags, predict where the missing tag must be, and re-detect inside
a small up-scaled <abbr title="Region of Interest">ROI</abbr>.

Accept only tags whose id re-detects, so it adds no false corners. This is the AprilGrid
analogue of ChArUco's `refineDetectedMarkers` and TartanCalib's `cornerpredictor`:

```python
from ds_msp.calib import AprilGridTarget
target = AprilGridTarget(6, 6, 0.088, 0.3)           # TUM-VI board
dets = detect_aprilgrid(paths, scales=(1, 2, 3), target=target, recover=True)
```

On this dataset multi-scale already grabs everything recoverable, so recovery is mostly
redundant here.

But it works from sparse seeds (6 detected tags → 25), and it's the right tool when you can
afford fewer scales or the board is partly occluded.

Its honest limit: a homography *extrapolates* poorly to tags right on the image border —
which is where the next idea comes in.

## 6. The frontier — detect on a reprojected view

The extreme image-edge tags (curved *and* truncated) defeat both fixes.

The production answer (TartanCalib's first method) is to **reproject the fisheye into a
virtual pinhole pointed at the board**, where the tags become near-frontal with straight
edges, detect there, and map the corners back through the camera model.

On frame 316 this recovers 6 more tags (26 → 32). That is exactly the library's
sphere/cylinder/pinhole reprojection machinery applied to the calibration front-end — the
same `project`/`unproject` maps, used to make detection robust.

/// note
Experimental; see [`experiments/aprilgrid_virtual_pinhole.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/experiments/aprilgrid_virtual_pinhole.py).
///

## 7. The payoff — a tighter calibration, for free

Multi-scale detection (with both subtleties fixed) doesn't just find more tags. It makes the
[capstone](capstone_calibrating_a_real_camera.md) **strictly better on every metric**,
because the recovered tags are the wide-<abbr title="Field of View">FOV</abbr> corners that
constrain focal length and distortion:

| | corners | median reproj | focal agreement | principal point |
|---|---|---|---|---|
| single-scale | 5,180 | 0.110 px | ~0.7% | ~0.11 px |
| **multi-scale (default)** | **14,472** | **0.080 px** | **~0.02%** | **~0.04 px** |

<div class="termy">

```console
$ python examples/03_calibrate_tumvi_aprilgrid.py --stride 4
Detecting AprilGrid in 109 frames ...
  109 usable frames, 14472 corners (5.6s)

Calibrating Kannala-Brandt from a generic seed (fx=fy=180), Cauchy loss ...
  all 14472 corners kept (none dropped); median 0.080 px, inlier RMS 0.160 px (96% within 1 px)

            fx        fy        cx        cy
published  190.978   190.973   254.932   256.897
mine       191.020   191.007   254.951   256.859
|Δ|          0.042     0.034     0.019     0.039
```

</div>

`|Δfx| = 0.042 px` on an `fx ≈ 191 px` focal — a **0.02%** agreement with TUM-VI's published
Kannala-Brandt calibration, tightened from ~0.7% with single-scale detection.

## Try it yourself
1. Re-run the snippet in §1 on `cam1` instead of `cam0`. Same dropout?
2. In the capstone, force single-scale (`detect_aprilgrid(..., scales=(1,))`) and watch the
   corner count *halve* and the focal agreement loosen — the wide-FOV corners earn their place.
3. Set `scales=(1, 2)` and watch recall sit between `(1,)` and `(1, 2, 3)`. More scales = more
   recall, at a (small) time cost.

## References
- TartanCalib — Duisterhof et al., arXiv:2210.02511 (virtual-pinhole + `cornerpredictor`).
- OpenCV ArUco/ChArUco — `refineDetectedMarkers`, ChArUco corner interpolation (local homography).
- Kalibr `GridCalibrationTargetAprilgrid`; Basalt AprilTag-3 wrapper.
- The 2-cell vs 1-cell tag border (why stock AprilTag-3/aruco decode *nothing* on Kalibr boards)
  is the [capstone's](capstone_calibrating_a_real_camera.md) detection war-story.
