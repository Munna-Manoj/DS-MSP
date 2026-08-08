#!/usr/bin/env python3
"""Chapter 10 companion: deterministic loop closure on TUM-VI fisheye data.

This is a readable reference pipeline, not a replacement for DBoW3:

  ORB -> hierarchical binary vocabulary -> TF-IDF -> inverted file
      -> temporal exclusion -> bearing-ray geometric verification -> PR metrics

The vocabulary images and evaluation images are disjoint. Every random-looking
choice is deterministic, and geometry uses the calibrated fisheye model rather
than pretending the raw image is pinhole.

Run from the repository root:
  python examples/12_loop_closure_tumvi.py
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from ds_msp.io import load_kalibr
from ds_msp.mvg import ransac_relative_pose

ROOM = Path("datasets/tumvi/dataset-room1_512_16")


def _hamming(rows: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between uint8 descriptor rows and centers."""
    xor = np.bitwise_xor(rows[:, None, :], centers[None, :, :])
    return np.unpackbits(xor, axis=2).sum(axis=2)


def _majority(rows: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(rows, axis=1)
    return np.packbits((2 * bits.sum(axis=0) >= len(rows)).astype(np.uint8))


@dataclass
class _TreeNode:
    centers: np.ndarray | None = None
    children: list["_TreeNode"] | None = None
    word: int = -1


class BinaryVocabularyTree:
    """Small deterministic hierarchical k-means tree for binary descriptors."""

    def __init__(self, branching: int = 6, levels: int = 3, iterations: int = 8):
        if branching < 2 or levels < 1:
            raise ValueError("branching >= 2 and levels >= 1 required")
        self.branching = branching
        self.levels = levels
        self.iterations = iterations
        self.root = _TreeNode()
        self.words = 0

    def fit(self, descriptors: np.ndarray) -> "BinaryVocabularyTree":
        if descriptors.ndim != 2 or descriptors.dtype != np.uint8 or len(descriptors) == 0:
            raise ValueError("expected a non-empty (N,D) uint8 descriptor matrix")
        self.words = 0
        self.root = self._fit_node(descriptors, depth=0)
        return self

    def _initial_centers(self, rows: np.ndarray, count: int) -> np.ndarray:
        # Lexicographic first center + farthest-first expansion. Ties resolve to
        # the lowest row index, making the result independent of RNG state.
        order = np.lexsort(rows[:, ::-1].T)
        centers = [rows[int(order[0])]]
        nearest = _hamming(rows, np.stack(centers))[:, 0]
        for _ in range(1, count):
            index = int(np.argmax(nearest))
            centers.append(rows[index])
            nearest = np.minimum(nearest, _hamming(rows, rows[index : index + 1])[:, 0])
        return np.stack(centers)

    def _fit_node(self, rows: np.ndarray, depth: int) -> _TreeNode:
        if depth == self.levels or len(rows) < self.branching:
            node = _TreeNode(word=self.words)
            self.words += 1
            return node

        count = min(self.branching, len(rows))
        centers = self._initial_centers(rows, count)
        assignment = np.zeros(len(rows), dtype=np.int32)
        for _ in range(self.iterations):
            assignment = np.argmin(_hamming(rows, centers), axis=1)
            updated = centers.copy()
            for cluster in range(count):
                members = rows[assignment == cluster]
                if len(members):
                    updated[cluster] = _majority(members)
            if np.array_equal(updated, centers):
                break
            centers = updated
        assignment = np.argmin(_hamming(rows, centers), axis=1)
        children = [
            self._fit_node(rows[assignment == cluster], depth + 1)
            if np.any(assignment == cluster)
            else self._fit_node(centers[cluster : cluster + 1], self.levels)
            for cluster in range(count)
        ]
        return _TreeNode(centers=centers, children=children)

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        words = np.empty(len(descriptors), dtype=np.int32)
        for index, descriptor in enumerate(descriptors):
            node = self.root
            while node.word < 0:
                assert node.centers is not None and node.children is not None
                child = int(np.argmin(_hamming(descriptor[None, :], node.centers)[0]))
                node = node.children[child]
            words[index] = node.word
        return words


class InvertedFile:
    """Exact sparse histogram-intersection query over posting lists."""

    def __init__(self, words: int):
        self.postings: list[list[tuple[int, float]]] = [[] for _ in range(words)]
        self.size = 0

    def add(self, bow: np.ndarray) -> int:
        image_id = self.size
        for word in np.flatnonzero(bow):
            self.postings[int(word)].append((image_id, float(bow[word])))
        self.size += 1
        return image_id

    def query(self, bow: np.ndarray, max_id: int) -> tuple[int, float] | None:
        scores: dict[int, float] = {}
        for word in np.flatnonzero(bow):
            for image_id, value in self.postings[int(word)]:
                if image_id >= max_id:
                    break
                scores[image_id] = scores.get(image_id, 0.0) + min(float(bow[word]), value)
        if not scores:
            return None
        # Stable tie-break: earlier image ID wins.
        return max(scores.items(), key=lambda item: (item[1], -item[0]))


@dataclass
class Frame:
    source_index: int
    timestamp: int
    keypoints: np.ndarray
    descriptors: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray
    bow: np.ndarray | None = None


@dataclass
class Detection:
    score: float
    pair_is_loop: bool
    query_has_loop: bool
    geometry_pass: bool


def _image_rows() -> list[tuple[int, str]]:
    path = ROOM / "mav0/cam0/data.csv"
    rows = []
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            timestamp, filename = line.split(",")
            rows.append((int(timestamp), filename))
    return rows


def _nearest_mocap(timestamps: np.ndarray, mocap: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(mocap[:, 0].astype(np.int64), timestamps)
    positions = np.clip(positions, 1, len(mocap) - 1)
    before = positions - 1
    choose_before = (
        timestamps - mocap[before, 0].astype(np.int64)
        <= mocap[positions, 0].astype(np.int64) - timestamps
    )
    return np.where(choose_before, before, positions)


def _extract(index: int, row: tuple[int, str], orb: cv2.ORB, mocap_row: np.ndarray) -> Frame:
    timestamp, filename = row
    raw = cv2.imread(str(ROOM / "mav0/cam0/data" / filename), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(filename)
    # TUM-VI's 16-bit camera samples need an explicit deterministic conversion.
    image = cv2.convertScaleAbs(raw, alpha=1.0 / 256.0) if raw.dtype == np.uint16 else raw
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is None:
        raise RuntimeError(f"ORB found no descriptors in frame {index}")
    return Frame(
        source_index=index,
        timestamp=timestamp,
        keypoints=np.array([keypoint.pt for keypoint in keypoints], dtype=np.float64),
        descriptors=descriptors,
        position=mocap_row[1:4].copy(),
        quaternion=mocap_row[4:8].copy(),
    )


def _is_loop(a: Frame, b: Frame, distance_m: float = 0.75, angle_deg: float = 45.0) -> bool:
    distance = float(np.linalg.norm(a.position - b.position))
    dot = float(np.clip(abs(np.dot(a.quaternion, b.quaternion)), 0.0, 1.0))
    angle = float(np.degrees(2.0 * np.arccos(dot)))
    return distance <= distance_m and angle <= angle_deg


def _bow(word_ids: np.ndarray, idf: np.ndarray) -> np.ndarray:
    vector = np.bincount(word_ids, minlength=len(idf)).astype(np.float64) * idf
    norm = float(vector.sum())
    return vector / norm if norm else vector


def _verify(query: Frame, candidate: Frame, model, seed: int) -> tuple[bool, int, int]:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(query.descriptors, candidate.descriptors, k=2)
    accepted = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(accepted) < 8:
        return False, 0, len(accepted)
    query_uv = query.keypoints[[match.queryIdx for match in accepted]]
    candidate_uv = candidate.keypoints[[match.trainIdx for match in accepted]]
    query_rays, query_valid = model.unproject(query_uv)
    candidate_rays, candidate_valid = model.unproject(candidate_uv)
    valid = query_valid & candidate_valid
    if int(valid.sum()) < 8:
        return False, 0, int(valid.sum())
    try:
        _, _, inliers = ransac_relative_pose(
            query_rays[valid],
            candidate_rays[valid],
            threshold=0.006,
            max_iters=500,
            seed=seed,
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return False, 0, int(valid.sum())
    count = int(inliers.sum())
    return count >= 20 and count / len(inliers) >= 0.25, count, len(inliers)


def _metrics(detections: list[Detection], require_geometry: bool) -> tuple[float, float]:
    thresholds = sorted({d.score for d in detections}, reverse=True) + [np.inf, -np.inf]
    best_f1 = 0.0
    zero_fp_recall = 0.0
    for threshold in thresholds:
        tp = fp = fn = 0
        for detection in detections:
            accepted = detection.score >= threshold and (
                detection.geometry_pass or not require_geometry
            )
            tp += int(accepted and detection.pair_is_loop)
            fp += int(accepted and not detection.pair_is_loop)
            fn += int(detection.query_has_loop and not (accepted and detection.pair_is_loop))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if fp == 0:
            zero_fp_recall = max(zero_fp_recall, recall)
        best_f1 = max(best_f1, f1)
    return zero_fp_recall, best_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=20, help="evaluation stride (20 Hz source)")
    parser.add_argument("--train-stride", type=int, default=43)
    parser.add_argument("--exclude-seconds", type=float, default=15.0)
    parser.add_argument("--max-features", type=int, default=500)
    args = parser.parse_args()
    if not ROOM.is_dir():
        raise SystemExit("Fetch TUM-VI first: bash scripts/download_datasets.sh tumvi")

    cv2.setNumThreads(1)
    cv2.setRNGSeed(10)
    started = perf_counter()
    model = load_kalibr(ROOM / "dso/camchain.yaml", "cam0")
    rows = _image_rows()
    mocap = np.loadtxt(ROOM / "mav0/mocap0/data.csv", delimiter=",", comments="#")
    all_timestamps = np.array([timestamp for timestamp, _ in rows], dtype=np.int64)
    mocap_indices = _nearest_mocap(all_timestamps, mocap)
    orb = cv2.ORB_create(args.max_features)

    train_ids = list(range(0, len(rows), args.train_stride))
    train_set = set(train_ids)
    eval_ids = [index for index in range(1, len(rows), args.stride) if index not in train_set]
    train_frames = [_extract(i, rows[i], orb, mocap[mocap_indices[i]]) for i in train_ids]
    eval_frames = [_extract(i, rows[i], orb, mocap[mocap_indices[i]]) for i in eval_ids]
    assert train_set.isdisjoint(eval_ids)

    vocabulary = BinaryVocabularyTree(branching=6, levels=3).fit(
        np.vstack([frame.descriptors for frame in train_frames])
    )
    document_frequency = np.zeros(vocabulary.words)
    train_word_ids = []
    for frame in train_frames:
        word_ids = vocabulary.transform(frame.descriptors)
        train_word_ids.append(word_ids)
        document_frequency[np.unique(word_ids)] += 1
    idf = np.zeros(vocabulary.words)
    present = document_frequency > 0
    idf[present] = np.log(len(train_frames) / document_frequency[present])
    for frame, word_ids in zip(train_frames, train_word_ids):
        frame.bow = _bow(word_ids, idf)
    for frame in eval_frames:
        frame.bow = _bow(vocabulary.transform(frame.descriptors), idf)

    database = InvertedFile(vocabulary.words)
    detections: list[Detection] = []
    verified_pairs = 0
    exclusion_ns = int(args.exclude_seconds * 1e9)
    for query_id, query in enumerate(eval_frames):
        eligible = 0
        while (
            eligible < query_id
            and eval_frames[eligible].timestamp <= query.timestamp - exclusion_ns
        ):
            eligible += 1
        candidate = database.query(query.bow, max_id=eligible) if query.bow is not None else None
        if candidate is not None:
            candidate_id, score = candidate
            prior = eval_frames[candidate_id]
            query_has_loop = any(_is_loop(query, frame) for frame in eval_frames[:eligible])
            geometry_pass, _, _ = _verify(query, prior, model, seed=query.source_index)
            verified_pairs += int(geometry_pass)
            detections.append(
                Detection(
                    score=score,
                    pair_is_loop=_is_loop(query, prior),
                    query_has_loop=query_has_loop,
                    geometry_pass=geometry_pass,
                )
            )
        database.add(query.bow)

    raw_zero_fp, raw_f1 = _metrics(detections, require_geometry=False)
    verified_zero_fp, verified_f1 = _metrics(detections, require_geometry=True)
    positives = sum(d.query_has_loop for d in detections)
    print(f"frames: train={len(train_frames)} eval={len(eval_frames)} overlap=0")
    print(
        f"vocabulary: k=6 L=3 words={vocabulary.words} "
        f"(capacity={6**3}, quantization <= {6*3} Hamming distances/descriptor)"
    )
    print(
        f"queries={len(detections)} positive_queries={positives} "
        f"geometry_pass={verified_pairs}"
    )
    print(
        f"appearance: zero-FP recall={raw_zero_fp:.3f} best-F1={raw_f1:.3f}\n"
        f"verified:   zero-FP recall={verified_zero_fp:.3f} best-F1={verified_f1:.3f}"
    )
    print(f"elapsed={perf_counter() - started:.2f}s deterministic_seed=10")


if __name__ == "__main__":
    main()
