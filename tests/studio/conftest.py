"""A real result folder, written by the rig pipeline and the CLI report writer, for the
Studio tests. Three cameras, 30 frames, about 10 s."""
import contextlib
import io
import warnings

import pytest

from ds_msp.io.mccalib import Scenario
from ds_msp.rig import report as rpt
from ds_msp.rig.cli import _report_and_exit_code
from ds_msp.rig.pipeline import calibrate_scenario
from tests.rig._synth import make_rig


@pytest.fixture(scope="session")
def results_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("results")
    obj, obs, img_size, _ext, models = make_rig(n_cam=3, n_frame=30, noise_px=0.05, seed=4)
    scn = Scenario(name="studio", object=obj, object_obs=obs, cam_ids=sorted(models),
                   img_size=img_size, gt={}, mccalib={}, mccalib_rms={})
    save = root / "lab" / "run_a"
    save.mkdir(parents=True)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        res = calibrate_scenario(scn, {c: m.name for c, m in models.items()}, noise_bound=1.0,
                                 save_dir=str(save), crossval_gate="warn")
        _report_and_exit_code(res["rig"], scn, res["models"], str(save),
                              pass_px=rpt.DEFAULT_PASS_PX, warn_px=rpt.DEFAULT_WARN_PX,
                              html_path=None, audit=res.get("audit"),
                              output_paths=res.get("paths"), lens=res.get("lens"),
                              crossval=res.get("crossval"))
    (root / "empty_dir").mkdir()
    (root / ".hidden").mkdir()
    return root
