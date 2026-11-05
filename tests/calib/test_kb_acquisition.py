"""Equal-budget acquisition regression; complete KB parameters and unknown poses."""
import numpy as np
import pytest

from benchmarks.kb_coverage_study import study


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.req("FR-CALIB-001", "NFR-REPRO-001")
def test_peripheral_acquisition_constrains_unseen_rays_for_both_k4_signs(tmp_path):
    result = study(3, tmp_path / "coverage.json")
    assert len(result["trials"]) == 12
    for k4 in (-.00015, .00015):
        trials = [r for r in result["trials"] if r["k4"] == k4]
        central = [r for r in trials if r["acquisition"] == "central"]
        peripheral = [r for r in trials if r["acquisition"] == "peripheral"]
        assert all(r["training_rms_px"] < .2 for r in trials)
        # Both fits can explain training pixels. Only the varied acquisition must
        # satisfy this independent-ray regression tolerance; invalid rays fail it.
        assert all(r["cone95"]["invalid"] == 0 for r in peripheral)
        assert max(r["cone95"]["p95_px"] for r in peripheral) < 1.0
        central_error = [r["cone95"]["p95_px"] or np.inf for r in central]
        peripheral_error = [r["cone95"]["p95_px"] for r in peripheral]
        assert np.median(peripheral_error) < np.median(central_error)
