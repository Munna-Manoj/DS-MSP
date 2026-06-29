"""Mark every rig test as ``slow``.

The rig integration tests run full multi-camera bundle adjustment across several
camera models and outlier regimes; individually they take tens to hundreds of
seconds (``test_model_agnostic`` is 200-415 s per model — see ``pytest
--durations``). They are the authoritative rig validation and run in CI's
dedicated slow-test job, but they must NOT run on every local push or in the
fast PR matrix. Marking the whole package ``slow`` lets ``-m "not slow"`` give
fast local feedback while ``-m slow`` runs the full rig validation.
"""
from pathlib import Path

import pytest

_RIG_DIR = Path(__file__).parent


def pytest_collection_modifyitems(config, items):
    for item in items:
        try:
            item.path.relative_to(_RIG_DIR)
        except ValueError:
            continue  # not a rig test
        item.add_marker(pytest.mark.slow)
