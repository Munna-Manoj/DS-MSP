"""Give freed heap memory back to the operating system after a memory-heavy stage.

glibc's ``malloc`` keeps memory a program has freed, so it can reuse it, and gives every thread
its own arena. After a stage that ran many threads at once (image detection uses one thread per
core) or held large temporaries (the observability audit), the process keeps that memory for
the rest of the run although nothing uses it: on a 48-core machine, detecting 72 images of
2448 x 2048 left the process at 1.14 GB, and ``malloc_trim(0)`` brought it back to 0.21 GB.

:func:`release_free_heap` calls ``malloc_trim(0)`` where glibc provides it and does nothing
elsewhere (macOS, Windows, musl). It frees only memory that is already unused, so it cannot
change any result.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Callable, Optional

_TRIM: Optional[Callable[[int], int]] = None
_LOADED = False


def _load() -> Optional[Callable[[int], int]]:
    global _TRIM, _LOADED
    if not _LOADED:
        _LOADED = True
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            fn = libc.malloc_trim
            fn.argtypes = [ctypes.c_size_t]
            fn.restype = ctypes.c_int
            _TRIM = fn
        except (OSError, AttributeError):
            _TRIM = None
    return _TRIM


def release_free_heap() -> bool:
    """Return the process's free heap pages to the OS (glibc ``malloc_trim(0)``).

    Returns ``True`` when memory was released, ``False`` when there was none to release or
    the platform has no ``malloc_trim``."""
    fn = _load()
    if fn is None:
        return False
    try:
        return bool(fn(0))
    except OSError:                                   # pragma: no cover - defensive
        return False
