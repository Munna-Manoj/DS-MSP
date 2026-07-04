#!/usr/bin/env python3
"""Repo-root convenience wrapper for a git-clone checkout.

The real CLI logic lives in :mod:`ds_msp.calib.cli` so it ships inside the ``ds_msp`` package
and is available as the ``ds-msp-calibrate`` console command straight after
``pip install ds-msp`` (only ``ds_msp*`` is included in the wheel/sdist -- this ``scripts/``
directory is not). This file exists purely so ``python scripts/calibrate.py ...`` works for
anyone who cloned the repo instead of installing from PyPI.
"""
import sys

sys.path.insert(0, ".")
from ds_msp.calib.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
