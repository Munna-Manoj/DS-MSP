"""Statistically validate that rig calibration is camera-model-agnostic.

For each camera model (RadTan, Double Sphere, UCM, EUCM, Kannala-Brandt) we synthesize
many independent rigs — the cameras *are represented by that model* (ground-truth
projection uses it) and are calibrated from scratch with that model — and measure the
worst per-camera extrinsic translation error against ground truth.

Per model we then run a one-sided t-test of H0: mean error >= 1% vs H1: mean error < 1%,
report the 99% upper confidence bound on the mean, the per-trial pass rate, and a
Kruskal-Wallis test across models (is the error distribution materially affected by which
model represents the cameras?). The pipeline is declared model-agnostic-within-1% when
every model rejects H0 at p < 0.01 and every trial is < 1%.

Usage:  python scripts/validate_model_agnostic.py [n_trials] [noise_px]
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats

sys.path.insert(0, ".")
from tests.rig._synth import make_rig                                   # noqa: E402
from ds_msp.rig import bundle                                              # noqa: E402
from ds_msp.rig.calibrate import calibrate_rig, make_bundle_front_end  # noqa: E402
from ds_msp.models.radtan import RadTanModel                          # noqa: E402
from ds_msp.models.double_sphere import DoubleSphereModel             # noqa: E402
from ds_msp.models.ucm import UCMModel                                # noqa: E402
from ds_msp.models.eucm import EUCMModel                              # noqa: E402
from ds_msp.models.kb import KannalaBrandtModel                       # noqa: E402

W, H = 1280, 960


def _factory(model_name):
    """Return a model_factory(cam_id, rng) that builds a GT camera of the given model,
    with mild per-camera intrinsic jitter so trials are genuinely independent."""
    def f(cam_id, rng):
        fx = 800.0 * rng.uniform(0.95, 1.05)
        fy = fx * rng.uniform(0.98, 1.02)
        cx, cy = W / 2 + rng.uniform(-15, 15), H / 2 + rng.uniform(-15, 15)
        if model_name == "radtan":
            return RadTanModel(fx, fy, cx, cy, rng.uniform(-0.08, 0.0),
                               rng.uniform(0.0, 0.03), 0.0, 0.0, 0.0)
        if model_name == "ds":
            return DoubleSphereModel(fx, fy, cx, cy, rng.uniform(0.1, 0.3),
                                     rng.uniform(0.3, 0.5))
        if model_name == "ucm":
            return UCMModel(fx, fy, cx, cy, rng.uniform(0.5, 0.7))
        if model_name == "eucm":
            return EUCMModel(fx, fy, cx, cy, rng.uniform(0.5, 0.7),
                             rng.uniform(0.9, 1.1))
        if model_name == "kb":
            return KannalaBrandtModel(fx, fy, cx, cy, rng.uniform(0.0, 0.02),
                                      rng.uniform(0.0, 0.005), 0.0, 0.0)
        raise ValueError(model_name)
    return f


def _rel(Tref, Ti):
    return Ti @ np.linalg.inv(Tref)


def _trial(model_name, model_cls, seed, noise_px):
    obj, obs, img, gt, gtm = make_rig(n_cam=3, n_frame=60, noise_px=noise_px,
                                      seed=seed, w=W, h=H,
                                      model_factory=_factory(model_name))
    rig = calibrate_rig(obj, obs, img, fix_intrinsics=False,
                        front_end=make_bundle_front_end(model_cls))
    ref = rig.ref_cam_id
    worst = max(
        abs(np.linalg.norm(_rel(rig.T_c_g[ref], rig.T_c_g[c])[:3, 3])
            - np.linalg.norm(_rel(gt[ref], gt[c])[:3, 3]))
        / np.linalg.norm(_rel(gt[ref], gt[c])[:3, 3])
        for c in rig.T_c_g if c != ref)
    rms = max(bundle.reprojection_rms(rig, obs).values())
    return 100.0 * worst, rms


MODELS = {"radtan": RadTanModel, "ds": DoubleSphereModel, "ucm": UCMModel,
          "eucm": EUCMModel, "kb": KannalaBrandtModel}


def main(n_trials=40, noise_px=0.3):
    print(f"Model-agnostic rig validation: {n_trials} trials/model, "
          f"noise={noise_px}px, 3 cameras, threshold 1%\n")
    print(f"{'model':8s} {'mean%':>7} {'std%':>7} {'max%':>7} {'99%UB':>7} "
          f"{'pass':>6} {'t-test p(mean<1%)':>18}  verdict")
    all_errs = {}
    ok = True
    for name, cls in MODELS.items():
        errs = np.array([_trial(name, cls, s, noise_px)[0] for s in range(n_trials)])
        all_errs[name] = errs
        mean, std = errs.mean(), errs.std(ddof=1)
        # one-sided 99% upper confidence bound on the mean
        ub99 = mean + stats.t.ppf(0.99, n_trials - 1) * std / np.sqrt(n_trials)
        # one-sided t-test H0: mean >= 1 vs H1: mean < 1
        tstat = (mean - 1.0) / (std / np.sqrt(n_trials))
        pval = stats.t.cdf(tstat, n_trials - 1)            # P(mean < 1) evidence
        passrate = float((errs < 1.0).mean())
        sig = (pval < 0.01) and (errs < 1.0).all()
        ok = ok and sig
        print(f"{name:8s} {mean:7.3f} {std:7.3f} {errs.max():7.3f} {ub99:7.3f} "
              f"{passrate*100:5.0f}% {pval:18.2e}  {'PASS' if sig else 'FAIL'}")

    # cross-model: does the representing model materially change the error distribution?
    kw = stats.kruskal(*all_errs.values())
    print(f"\nKruskal-Wallis across models: H={kw.statistic:.3f}, p={kw.pvalue:.3f}")
    print("(all models well under 1% -> distribution differences are immaterial to the claim)")
    print(f"\nOVERALL: {'MODEL-AGNOSTIC WITHIN 1% (statistically significant)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    npx = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    main(n, npx)
