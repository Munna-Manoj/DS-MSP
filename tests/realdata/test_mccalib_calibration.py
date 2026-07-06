"""Real-data validation of robust ``calibrate()`` + ``convert()`` (FR-CALIB-001,
FR-ADAPT-001, NFR-NUM-004).

The synthetic suites pin the robustness *contract*; this is the real-data half of the
release gate. It calibrates **three** fisheye cameras from a multi-camera capture (each
~60 views of a 7x7 ChArUco board, ~2k real corner detections at 2592x1800) entirely from a
**generic** init (focal = W/pi, centre = image centre; no reference intrinsics supplied)
and asserts:

  * **Per camera, robust auto-init is sub-pixel** and recovers the reference camera matrix
    K to a tight tolerance (``test_intrinsics_recovered_per_camera``), and
  * **Self-conversion is an exact fixed point of ``convert()``**
    (``test_convert_self_is_exact_per_camera``): converting a from-scratch fit back into its
    own class reproduces its *projection* to machine precision, for both DS+ and KB
    (independent real fits from the same detected corners).

The broader "DS+ is a faithful conversion target for any real fisheye fit" claim is verified
*synthetically* (``tests/adapt/test_convert_robustness.py::test_dsplus_is_faithful_universal_target``,
sub-pixel across UCM/EUCM/DS/KB ground truth). On this dataset's real ~158-170 deg lenses, a
from-scratch KB fit converted into DS+ was measured to diverge past sub-pixel at the periphery
(up to ~1.5px RMS at 50-70 deg ray angle) — a genuine cross-family gap between DS+'s sphere
parameterization and KB's polynomial one on this data, not asserted here pending a dedicated
follow-up experiment.

Camera 0 here is the same physical lens used as the reference checkerboard elsewhere in the
project; cameras 2 and 4 are distinct cameras (different K and reference reprojection).

Corners are detected on the fly with ``ds_msp.detect.charuco`` (verified to reproduce the
dataset's own published cam0 corner set: 1954 vs 1943 corners). Dataset-gated: the imagery
is local-only (gitignored) so this test SKIPS when absent (PR CI), and is required-green in
the pre-release job that closes the gate. Point it at the data with
``DSMSP_MCCALIB_DIR=/path/to/2026_06_26_MC-Calib``.
"""

import glob
import os
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest

from ds_msp.adapt import convert
from ds_msp.adapt.evaluate import reprojection_report
from ds_msp.calib import calibrate
from ds_msp.detect.charuco import BoardSpec, detect_image, make_detectors
from ds_msp.models.dsplus import DSPlusModel
from ds_msp.models.kb import KannalaBrandtModel

CAMERAS = (0, 2, 4)
# ChArUco board for this rig: 7x7, DICT_6X6_1000, 0.1 m squares (legacy pattern).
_SPEC = BoardSpec(n_x=7, n_y=7, length_square=0.1, length_marker=0.075, square_size=0.1)
_NCX = _SPEC.n_x - 1
_SQ = _SPEC.square_size


def _dataset_root() -> Path | None:
    env = os.environ.get("DSMSP_MCCALIB_DIR")
    root = Path(env) if env else Path(__file__).resolve().parents[2] / "2026_06_26_MC-Calib"
    return root if (root / "calibration_images_0").is_dir() else None


def _K4(model) -> np.ndarray:
    K = model.K
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])


def _reference_K(cam_dir: Path):
    """The dataset's published (fx, fy, cx, cy), reprojection error, and image size."""
    r = ET.parse(cam_dir / "calibration.xml").getroot()
    cm = r.find("camera_matrix")
    K = np.array([float(cm.find(t).text) for t in ("fx", "fy", "ppx", "ppy")])
    W = int(r.find("image_size/width").text)
    H = int(r.find("image_size/height").text)
    return K, float(r.find("reprojection_error").text), W, H


def _detect_cam(cam_dir: Path):
    """Detect board-0 ChArUco corners across a camera's images -> calibration inputs.

    Corner id ``k`` maps to board point ``((k % ncx)*sq, (k // ncx)*sq, 0)`` — the row-major
    interior-corner convention of ``ds_msp.detect.charuco``.
    """
    detectors = make_detectors([_SPEC], legacy=True)
    Xs, kps, vis, all_px = [], [], [], []
    for f in sorted(glob.glob(str(cam_dir / "*.png"))):
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if gray is None:                       # a few frames in this set are truncated PNGs
            continue
        for board_id, ids, pts in detect_image(detectors, gray, min_corners=6):
            if board_id != 0:
                continue
            X = np.column_stack([(ids % _NCX) * _SQ, (ids // _NCX) * _SQ, np.zeros(len(ids))])
            Xs.append(X.astype(float))
            kps.append(pts.astype(float))
            vis.append(np.ones(len(ids), bool))
            all_px.append(pts)
    return Xs, kps, vis, np.vstack(all_px)


_CACHE: dict[int, dict] = {}


def _fit_camera(cam: int) -> dict:
    """Detect + calibrate DS+ and KB from a generic init, plus the two conversions.

    Cached per camera so every test/parametrization reuses one fit.
    """
    if cam in _CACHE:
        return _CACHE[cam]
    root = _dataset_root()
    cam_dir = root / f"calibration_images_{cam}"
    refK, ref_reproj, W, H = _reference_K(cam_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # benign invalid-divide on non-projectable rays
        Xs, kps, vis, _ = _detect_cam(cam_dir)
        assert len(Xs) >= 30, f"cam{cam}: only {len(Xs)} usable views detected"
        f0 = W / np.pi
        dsp = calibrate(DSPlusModel(f0, f0, W / 2, H / 2, 0.5, 0, 0, 0, 0),
                        Xs, kps, vis, max_nfev=200)
        kb = calibrate(KannalaBrandtModel(f0, f0, W / 2, H / 2, 0, 0, 0, 0),
                       Xs, kps, vis, max_nfev=200)
        # convert(): self-conversions, must be an exact fixed point for both models.
        dsp_self, _ = convert(dsp["model"], DSPlusModel, width=W, height=H)
        kb_self, _ = convert(kb["model"], KannalaBrandtModel, width=W, height=H)
    out = dict(dsp=dsp, kb=kb, dsp_self=dsp_self, kb_self=kb_self,
               refK=refK, ref_reproj=ref_reproj, W=W, H=H,
               n_views=len(Xs), n_corners=sum(len(x) for x in Xs))
    _CACHE[cam] = out
    return out


@pytest.fixture(scope="session")
def fitted():
    if _dataset_root() is None:
        pytest.skip("MC-Calib imagery not present (set DSMSP_MCCALIB_DIR)")
    return _fit_camera


pytestmark = [pytest.mark.realdata,
              pytest.mark.req("FR-CALIB-001", "FR-ADAPT-001", "NFR-NUM-004")]


@pytest.mark.parametrize("cam", CAMERAS)
def test_intrinsics_recovered_per_camera(fitted, cam):
    """Each real fisheye calibrates sub-pixel from a generic init and recovers the
    reference camera matrix K."""
    r = fitted(cam)
    dsp, refK = r["dsp"], r["refK"]
    assert dsp["success"], (cam, dsp)
    assert dsp["median_px"] < 0.40, (cam, dsp["median_px"])
    assert dsp["mean_px"] < 0.45, (cam, dsp["mean_px"])
    assert dsp["p95_px"] < 1.0, (cam, dsp["p95_px"])
    got = _K4(dsp["model"])
    assert np.allclose(got[:2], refK[:2], rtol=0.01), (cam, got, refK)   # focal within 1%
    assert np.allclose(got[2:], refK[2:], atol=8.0), (cam, got, refK)    # centre within 8 px


@pytest.mark.parametrize("cam", CAMERAS)
def test_convert_self_is_exact_per_camera(fitted, cam):
    """A from-scratch calibration is a fixed point of convert(): a model -> its own class
    reproduces the source *projection* to machine precision. Checked for BOTH DS+ and KB.
    Exactness is measured by reprojection RMS, not parameter distance: when alpha sits at its
    bound the parameterization is degenerate (equivalent param sets project identically), so
    projection equivalence is the meaningful contract."""
    r = fitted(cam)
    rms_ds = reprojection_report(r["dsp"]["model"], r["dsp_self"], r["W"], r["H"])["rms_px"]
    rms_kb = reprojection_report(r["kb"]["model"], r["kb_self"], r["W"], r["H"])["rms_px"]
    assert rms_ds < 1e-3, (cam, "dsplus", rms_ds)
    assert rms_kb < 1e-3, (cam, "kb", rms_kb)
