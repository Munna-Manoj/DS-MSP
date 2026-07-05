"""Covisibility graph utilities — the same skeleton MC-Calib applies at three levels
(boards, cameras, camera-groups) via Boost Graph (``Graph.cpp``).

Zero external deps (no networkx): a tiny union-find for connected components and a
plain Dijkstra for shortest paths. Edge weight is ``1 / co-observation-count`` so the
shortest path between two elements prefers the route with the *most* shared
observations (McCalib.cpp:873, 1113).
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Tuple

Edge = Tuple[int, int]


def covis_weights(pair_counts: Dict[Edge, int]) -> Dict[Edge, float]:
    """``pair_counts[(i, j)] = #frames i and j were co-observed`` -> edge weight ``1/N``.

    Keys are normalized to ``i < j`` (undirected). Symmetric duplicates are summed.
    """
    w: Dict[Edge, int] = {}
    for (i, j), n in pair_counts.items():
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        w[key] = w.get(key, 0) + n
    return {e: 1.0 / n for e, n in w.items() if n > 0}


def connected_components(nodes: List[int], weights: Dict[Edge, float]) -> List[List[int]]:
    """Union-find connected components. Each component is sorted so ``comp[0]`` is the
    minimum id — MC-Calib's reference choice (``min_element``)."""
    parent = {n: n for n in nodes}

    def find(x: int) -> int:
        """Union-find root of ``x``, with path halving."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j) in weights:
        if i in parent and j in parent:
            parent[find(i)] = find(j)

    comps: Dict[int, List[int]] = {}
    for n in nodes:
        comps.setdefault(find(n), []).append(n)
    return [sorted(c) for c in comps.values()]


def _adjacency(weights: Dict[Edge, float]) -> Dict[int, List[Tuple[int, float]]]:
    adj: Dict[int, List[Tuple[int, float]]] = {}
    for (i, j), w in weights.items():
        adj.setdefault(i, []).append((j, w))
        adj.setdefault(j, []).append((i, w))
    return adj


def shortest_path(src: int, dst: int, weights: Dict[Edge, float]) -> List[int]:
    """Dijkstra shortest path ``src -> dst`` by edge weight. Returns the node list
    ``[src, ..., dst]`` (``[src]`` if equal; raises if disconnected)."""
    if src == dst:
        return [src]
    adj = _adjacency(weights)
    dist = {src: 0.0}
    prev: Dict[int, int] = {}
    pq: List[Tuple[float, int]] = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dst not in prev and dst != src:
        raise ValueError(f"no path from {src} to {dst}")
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]
