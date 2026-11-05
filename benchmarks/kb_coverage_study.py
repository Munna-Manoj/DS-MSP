"""Self-contained equal-budget KB acquisition study; no external assets required.

Run with one BLAS thread, e.g. PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python
benchmarks/kb_coverage_study.py --output reports/kb-coverage.json --seeds 3.
Truth generates noisy measurements and scores unseen rays; the public calibrator
receives a rough common lens seed and estimates every board pose itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import scipy

from ds_msp.calib.bundle import calibrate
from ds_msp.models.kb import KannalaBrandtModel as KB


def acquisition(seed, peripheral):
    rng = np.random.default_rng(seed)
    views = []
    for i in range(30):
        central = i < 18 or not peripheral
        theta = np.radians(rng.uniform(5, 42) if central else rng.uniform(72, 83))
        phi = rng.uniform(-np.pi, np.pi)
        axis = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                         np.cos(theta)])
        ex = np.array([-np.sin(phi), np.cos(phi), 0.])
        rotation = np.column_stack([ex, np.cross(axis, ex), axis]) @ cv2.Rodrigues(
            np.radians(rng.uniform([-25, -25, -180], [25, 25, 180])))[0]
        views.append((rotation, axis * rng.uniform(.85, 1.2)))
    return views


def metric(errors):
    def finite(value):
        return float(value) if np.isfinite(value) else None
    return {"p95_px": finite(np.quantile(errors, .95, method="higher")),
            "max_px": finite(np.max(errors)), "invalid": int((~np.isfinite(errors)).sum()),
            "n_queries": len(errors)}


def study(seeds, output):
    import ds_msp
    cv2.setNumThreads(1)
    xx, yy = np.meshgrid(np.linspace(-.24, .24, 9), np.linspace(-.18, .18, 7))
    board = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    theta, phi = np.meshgrid(np.radians(np.linspace(.2, 95, 96)),
                             np.linspace(-np.pi, np.pi, 120, endpoint=False))
    rays = np.column_stack([(np.sin(theta) * np.cos(phi)).ravel(),
                            (np.sin(theta) * np.sin(phi)).ravel(), np.cos(theta).ravel()])
    result = {"protocol": {"noise_sigma_px": .1, "views": 30, "corners_per_view": 63,
                            "seed_count": seeds, "fit": "full KB8 plus unknown board poses",
                            "truth_used_to_initialize": False},
              "runtime": {"python": sys.version, "numpy": np.__version__,
                          "scipy": scipy.__version__, "opencv": cv2.__version__},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_sha256": {
                  str(p.relative_to(Path(ds_msp.__file__).parent)):
                  hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(Path(ds_msp.__file__).parent.rglob("*.py"))},
              "trials": []}
    for k4 in (-.00015, .00015):
        truth = KB(400, 408, 803, 796, .03, -.006, .0007, k4)
        expected, valid = truth.project(rays)
        assert valid.all()
        for seed in range(seeds):
            for peripheral in (False, True):
                rng = np.random.default_rng(314159 + seed)
                objects, pixels, masks = [], [], []
                for rotation, translation in acquisition(20261105 + seed, peripheral):
                    uv, valid = truth.project(board @ rotation.T + translation)
                    assert valid.all() and ((uv > 0) & (uv < 1600)).all()
                    objects.append(board.copy())
                    pixels.append(uv + rng.normal(0, .1, uv.shape))
                    masks.append(np.ones(len(board), bool))
                start = time.monotonic()
                fit = calibrate(KB(380, 380, 800, 800, 0, 0, 0, 0), objects, pixels, masks,
                                robust="none", max_nfev=400, multi_start=True,
                                n_restarts=2, seed=1024)
                predicted, valid = fit["model"].project(rays)
                error = np.linalg.norm(predicted - expected, axis=1)
                error[~valid | ~np.isfinite(predicted).all(axis=1)] = np.inf
                row = {"k4": k4, "seed": seed,
                       "acquisition": "peripheral" if peripheral else "central",
                       "success": bool(fit["success"]), "training_rms_px": fit["rms_px"],
                       "parameters": fit["model"].params.tolist(),
                       "cone95": metric(error),
                       "periphery75to95": metric(error[theta.ravel() >= np.radians(75)]),
                       "seconds": time.monotonic() - start}
                result["trials"].append(row)
                output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    study(args.seeds, args.output)


if __name__ == "__main__":
    main()
