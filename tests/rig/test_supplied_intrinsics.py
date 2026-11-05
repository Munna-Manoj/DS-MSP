"""Independent lens calibration must survive initialization and fixed rig BA."""

import numpy as np
import pytest

from ds_msp.io.mccalib import Scenario
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.rig.pipeline import calibrate_scenario, make_initialized_front_end

from ._synth import make_rig


def _scene(k4):
    def factory(cam, rng):
        return KannalaBrandtModel(650, 660, 640, 480, .03, -.006, .0007, k4)

    obj, obs, img, truth, models = make_rig(
        n_cam=2, n_frame=12, noise_px=.05, seed=19, model_factory=factory)
    scn = Scenario(name="supplied", object=obj, object_obs=obs, cam_ids=sorted(img),
                   img_size=img, gt={}, mccalib={}, mccalib_rms={})
    return scn, truth, models


@pytest.mark.integration
@pytest.mark.req("FR-RIG-003")
@pytest.mark.parametrize("k4", [-.00015, .00015])
def test_full_kb_intrinsics_frozen_through_rigid_board_refinement(k4):
    scn, truth, models = _scene(k4)
    original = {c: m.params.copy() for c, m in models.items()}
    result = calibrate_scenario(scn, "kb", init_cameras=models, fix_intrinsics=True,
                                audit_gate="off", uncertainty_gate="off")
    rig = result["rig"]
    assert any(s["stage"] == "d_rigid_boards" for s in rig.solver_report["stages"])
    for c in models:
        np.testing.assert_array_equal(rig.cameras[c].params, original[c])
        np.testing.assert_array_equal(models[c].params, original[c])
        np.testing.assert_allclose(rig.T_c_g[c], truth[c], atol=.002)


@pytest.mark.integration
@pytest.mark.req("FR-RIG-003")
def test_partial_intrinsics_calibrate_missing_camera_and_preserve_supplied_lens():
    scn, _, models = _scene(.00015)
    supplied = models[0]
    front = make_initialized_front_end({0: supplied}, "kb", init_K={1: models[1].K})
    observations = {c: [o for o in scn.object_obs if o.cam_id == c] for c in (0, 1)}
    result = front(scn.object, observations, scn.img_size)
    assert set(result) == {0, 1}
    assert result[0] is supplied
    assert result[1].name == "kb"
    assert front.report[0]["source"] == "supplied"
    for obs in observations.values():
        assert all(o.T_c_o is not None and np.isfinite(o.T_c_o).all() for o in obs)


@pytest.mark.integration
@pytest.mark.req("FR-RIG-003")
def test_partial_intrinsics_cannot_be_frozen():
    scn, _, models = _scene(.00015)
    with pytest.raises(ValueError, match=r"missing: \[1\]"):
        calibrate_scenario(scn, "kb", init_cameras={0: models[0]}, fix_intrinsics=True)
