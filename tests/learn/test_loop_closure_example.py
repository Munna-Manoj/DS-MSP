"""Deterministic unit gates for the loop-closure learning reference."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_example():
    path = Path(__file__).resolve().parents[2] / "examples/12_loop_closure_tumvi.py"
    spec = importlib.util.spec_from_file_location("loop_closure_tumvi", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


loop = _load_example()


def test_binary_majority_is_not_byte_mean():
    rows = np.array([[0], [1], [1]], dtype=np.uint8)
    assert int(loop._majority(rows)[0]) == 1
    assert int(rows.mean()) == 0


def test_vocabulary_is_deterministic_and_has_bounded_capacity():
    rng = np.random.default_rng(10)
    descriptors = rng.integers(0, 256, size=(240, 32), dtype=np.uint8)
    first = loop.BinaryVocabularyTree(branching=4, levels=3).fit(descriptors)
    second = loop.BinaryVocabularyTree(branching=4, levels=3).fit(descriptors)
    np.testing.assert_array_equal(first.transform(descriptors), second.transform(descriptors))
    assert 1 <= first.words <= 4**3


def test_inverted_file_matches_dense_histogram_intersection():
    rng = np.random.default_rng(11)
    words = 80
    database = loop.InvertedFile(words)
    dense = []
    for _ in range(25):
        vector = rng.random(words)
        vector[rng.random(words) < 0.85] = 0.0
        vector /= vector.sum()
        dense.append(vector)
        database.add(vector)
    query = rng.random(words)
    query[rng.random(words) < 0.85] = 0.0
    query /= query.sum()
    image_id, score = database.query(query, max_id=len(dense))
    expected = np.minimum(np.stack(dense), query).sum(axis=1)
    assert image_id == int(np.argmax(expected))
    np.testing.assert_allclose(score, expected[image_id], atol=1e-15)


def test_geometric_gate_cannot_improve_a_false_candidate():
    detections = [
        loop.Detection(0.9, True, True, True),
        loop.Detection(0.8, False, False, False),
        loop.Detection(0.7, True, True, True),
    ]
    raw_zero_fp, raw_f1 = loop._metrics(detections, require_geometry=False)
    verified_zero_fp, verified_f1 = loop._metrics(detections, require_geometry=True)
    assert verified_zero_fp >= raw_zero_fp
    assert verified_f1 >= raw_f1
