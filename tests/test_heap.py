"""release_free_heap returns freed heap pages to the OS where glibc offers malloc_trim and is a
harmless no-op elsewhere."""

import sys

import numpy as np
import pytest

from ds_msp.heap import release_free_heap

pytestmark = pytest.mark.req("NFR-PERF-008")


def _rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2 ** 20


def test_release_free_heap_is_safe_to_call():
    assert release_free_heap() in (True, False)
    assert release_free_heap() in (True, False)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc malloc_trim only")
def test_release_free_heap_returns_fragmented_small_blocks():
    """Many small freed arrays stay in malloc's free lists (RSS unchanged) until trimmed."""
    import ctypes.util
    if "glibc" not in (ctypes.util.find_library("c") or "libc.so.6") and not release_free_heap():
        pytest.skip("libc without malloc_trim")
    keep = []
    junk = [np.ones(2000) for _ in range(40000)]           # 16 kB each, below the mmap threshold
    keep.extend(junk[::1000])                              # pin a few so the heap top cannot shrink
    del junk
    before = _rss_mb()
    release_free_heap()
    after = _rss_mb()
    assert after < before - 100, (before, after)
    del keep
