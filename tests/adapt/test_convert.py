"""
Converter tests. Includes a real DS -> EUCM / UCM conversion with accuracy
assertions, and a Fake -> Fake identity (RE ~ 0) proving the converter runs with
no fisheye model present.
"""

import pytest
import numpy as np

from ds_msp.adapt import convert, sample_image_grid
from ds_msp.testing import FakeModel
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.ucm import UCMModel
from ds_msp.models.eucm import EUCMModel
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.radtan import RadTanModel

W, H = 1920, 1080


def test_sample_grid_count_and_bounds():
    pts = sample_image_grid(W, H, 500)
    assert 300 <= len(pts) <= 700
    assert (pts[:, 0] >= 0).all() and (pts[:, 0] <= W).all()
    assert (pts[:, 1] >= 0).all() and (pts[:, 1] <= H).all()


def test_fake_to_fake_identity():
    # Pinhole -> pinhole: the converter must recover the source exactly.
    src = FakeModel(600.0, 605.0, 320.0, 240.0)
    tgt, report = convert(src, FakeModel, width=640, height=480, n_samples=400)
    assert report["rms_px"] < 1e-6
    assert np.allclose(tgt.params, src.params, atol=1e-6)


def test_ds_to_eucm_is_near_exact():
    # EUCM is expressive enough to represent DS very closely.
    ds = DoubleSphereModel.sample()
    eucm, report = convert(ds, EUCMModel, width=W, height=H, n_samples=600)
    assert report["converged"]
    assert report["rms_px"] < 0.5, report


def test_ds_to_kb_is_accurate_and_opencv_ready():
    # KB (equidistant) represents DS well; result must be cv2.fisheye-ready.
    ds = DoubleSphereModel.sample()
    kb, report = convert(ds, KannalaBrandtModel, width=W, height=H, n_samples=600)
    assert report["converged"]
    assert report["rms_px"] < 1.0, report
    assert kb.K.shape == (3, 3) and kb.distortion.shape == (4,)


def test_ds_to_ucm_runs_and_reports():
    # UCM is less expressive than DS; conversion is lossy but must run and report.
    ds = DoubleSphereModel.sample()
    ucm, report = convert(ds, UCMModel, width=W, height=H, n_samples=600)
    assert report["converged"]
    assert np.isfinite(report["rms_px"])
    assert report["fov_covered_deg"] > 90.0


def test_ds_to_radtan_is_lossy_but_reported():
    # Pinhole/RadTan cannot hold a wide fisheye FOV; restrict the FOV and check
    # the converter still runs and reports coverage (it WILL be lossy).
    ds = DoubleSphereModel.sample()
    rt, report = convert(ds, RadTanModel, width=W, height=H, n_samples=600,
                         max_fov_deg=120.0)
    assert report["converged"]
    assert np.isfinite(report["rms_px"])
    assert report["fov_covered_deg"] <= 121.0


def test_wide_fov_to_radtan_without_max_fov_deg_is_not_falsely_converged():
    # Regression: converting a wide-FOV model to RadTan with NO max_fov_deg used to report
    # converged=True at rms in the thousands-to-quintillions of px (scipy's optimizer-
    # termination flag says nothing about fit quality). It must now be caught and reported
    # honestly instead.
    ds = DoubleSphereModel.sample()
    rt, report = convert(ds, RadTanModel, width=W, height=H, n_samples=600)
    diag = float(np.hypot(W, H))
    assert report["rms_px"] > 0.25 * diag  # confirms the scenario is genuinely diverged
    assert not report["converged"]
    assert "convergence_warning" in report
    assert "max_fov_deg" in report["convergence_warning"]


def test_fov_restricted_radtan_conversion_still_reports_converged():
    # The existing max_fov_deg-restricted lossy conversion (a real, usable fit) must not be
    # caught by the new sanity gate -- confirms the fix doesn't regress the legitimate case.
    ds = DoubleSphereModel.sample()
    rt, report = convert(ds, RadTanModel, width=W, height=H, n_samples=600, max_fov_deg=120.0)
    assert report["converged"]
    assert "convergence_warning" not in report


def test_converter_is_decoupled_from_concrete_models():
    # convert() receives the target class by injection; it imports no model.
    import ast
    import importlib
    import pathlib
    mod = importlib.import_module("ds_msp.adapt.convert")
    src = pathlib.Path(mod.__file__).read_text()
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("models" in m for m in imported), imported

# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-ADAPT-001")
