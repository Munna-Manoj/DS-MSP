"""Minimal, dependency-free OpenEXR scanline I/O for float32 RGB textures.

Isaac Sim's ``OmniLensDistortionLutAPI`` needs 32-bit float EXR textures. The
project's OpenCV dependency range includes wheels built *without* OpenEXR (for
example the opencv-python 5.0 wheels report ``OpenEXR: NO``), so the exporter
cannot rely on ``cv2.imwrite``. This module writes and reads the small subset of
the OpenEXR 2.0 format the LUT exporter needs:

* single-part scanline images, ``INCREASING_Y`` line order;
* ``FLOAT`` (32-bit) channels named ``R``, ``G``, ``B``;
* ``NONE`` or ``ZIP`` (16-scanline, zlib + predictor) compression on write, and
  ``NONE`` / ``ZIPS`` / ``ZIP`` on read.

The byte layout follows the OpenEXR file-format specification
(https://openexr.com/en/latest/OpenEXRFileLayout.html) and the ``ZIP`` predictor
+ byte-interleave scheme from ``ImfZip.cpp``; output is verified against the
reference OpenEXR library in ``tests/isaac_sim/test_exr.py`` whenever a
reference reader is installed.
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

MAGIC = 20000630  # 0x762f3101 as a little-endian int32
VERSION = 2  # single-part scanline, no long names, no deep data
_PIXEL_FLOAT = 2  # chlist pixel type: UINT=0, HALF=1, FLOAT=2
_COMPRESSION_NONE = 0
_COMPRESSION_ZIPS = 2  # one scanline per chunk
_COMPRESSION_ZIP = 3  # sixteen scanlines per chunk
_LINES_PER_CHUNK = {_COMPRESSION_NONE: 1, _COMPRESSION_ZIPS: 1, _COMPRESSION_ZIP: 16}
_COMPRESSION_BY_NAME = {"none": _COMPRESSION_NONE, "zip": _COMPRESSION_ZIP}

# Channels are stored alphabetically inside the file (the reference library sorts them).
_CHANNELS: Tuple[str, ...] = ("B", "G", "R")


# --------------------------------------------------------------------------- ZIP codec


def _zip_encode(raw: bytes) -> bytes:
    """OpenEXR ``ZIP``/``ZIPS`` payload: byte interleave, delta predictor, zlib."""
    data = np.frombuffer(raw, dtype=np.uint8)
    n = data.size
    half = (n + 1) // 2
    interleaved = np.empty(n, dtype=np.uint8)
    interleaved[:half] = data[0::2]
    interleaved[half:] = data[1::2]
    predicted = interleaved.astype(np.int32)
    predicted[1:] = (predicted[1:] - predicted[:-1] + 128) & 0xFF
    return zlib.compress(predicted.astype(np.uint8).tobytes())


def _zip_decode(payload: bytes, expected_size: int) -> bytes:
    """Inverse of :func:`_zip_encode`."""
    predicted = np.frombuffer(zlib.decompress(payload), dtype=np.uint8)
    if predicted.size != expected_size:
        raise ValueError(
            f"EXR chunk decompressed to {predicted.size} bytes, expected {expected_size}"
        )
    restored = np.empty(predicted.size, dtype=np.int64)
    restored[0] = predicted[0]
    restored[1:] = predicted[1:].astype(np.int64) - 128
    restored = (np.cumsum(restored) & 0xFF).astype(np.uint8)
    half = (predicted.size + 1) // 2
    out = np.empty(predicted.size, dtype=np.uint8)
    out[0::2] = restored[:half]
    out[1::2] = restored[half:]
    return out.tobytes()


# --------------------------------------------------------------------------- header


def _attribute(name: str, type_name: str, value: bytes) -> bytes:
    return name.encode("ascii") + b"\0" + type_name.encode("ascii") + b"\0" + struct.pack(
        "<i", len(value)
    ) + value


def _channel_list(channels: Tuple[str, ...]) -> bytes:
    out = b""
    for name in channels:
        # name, pixel type, pLinear + 3 reserved bytes, xSampling, ySampling
        out += name.encode("ascii") + b"\0" + struct.pack("<iBBBBii", _PIXEL_FLOAT, 0, 0, 0, 0, 1, 1)
    return out + b"\0"


def _header(width: int, height: int, compression: int) -> bytes:
    box = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    return b"".join(
        [
            struct.pack("<ii", MAGIC, VERSION),
            _attribute("channels", "chlist", _channel_list(_CHANNELS)),
            _attribute("compression", "compression", struct.pack("<B", compression)),
            _attribute("dataWindow", "box2i", box),
            _attribute("displayWindow", "box2i", box),
            _attribute("lineOrder", "lineOrder", struct.pack("<B", 0)),
            _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
            _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
            _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
            b"\0",
        ]
    )


# --------------------------------------------------------------------------- write


def write_rgb32_exr(path: str | Path, rgb: np.ndarray, *, compression: str = "zip") -> None:
    """Write an ``HxWx3`` float32 array as an ``R``/``G``/``B`` scanline EXR.

    The file is written to a temporary sibling and atomically renamed into place.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB data, got {rgb.shape}")
    try:
        method = _COMPRESSION_BY_NAME[compression.lower()]
    except KeyError as exc:
        raise ValueError(f"compression must be one of {sorted(_COMPRESSION_BY_NAME)}") from exc
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if width <= 0 or height <= 0:
        raise ValueError("EXR dimensions must be positive")

    # (H, C, W) float32 little-endian: each scanline is stored channel by channel.
    planes = np.ascontiguousarray(
        np.stack([rgb[..., "RGB".index(name)] for name in _CHANNELS], axis=1), dtype="<f4"
    )
    lines_per_chunk = _LINES_PER_CHUNK[method]
    chunk_count = (height + lines_per_chunk - 1) // lines_per_chunk
    header = _header(width, height, method)
    offset_table_size = 8 * chunk_count

    chunks: List[bytes] = []
    for start in range(0, height, lines_per_chunk):
        stop = min(start + lines_per_chunk, height)
        raw = planes[start:stop].tobytes()
        payload = raw
        if method == _COMPRESSION_ZIP:
            encoded = _zip_encode(raw)
            # The reference writer keeps the raw bytes when compression does not help.
            if len(encoded) < len(raw):
                payload = encoded
        chunks.append(struct.pack("<ii", start, len(payload)) + payload)

    offsets = []
    position = len(header) + offset_table_size
    for chunk in chunks:
        offsets.append(position)
        position += len(chunk)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".tmp" + target.suffix)
    with temporary.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<%dQ" % chunk_count, *offsets))
        for chunk in chunks:
            stream.write(chunk)
    os.replace(temporary, target)


# --------------------------------------------------------------------------- read


def _read_cstring(buf: bytes, pos: int) -> Tuple[str, int]:
    end = buf.index(b"\0", pos)
    return buf[pos:end].decode("ascii"), end + 1


def _parse_header(buf: bytes) -> Tuple[Dict[str, Tuple[str, bytes]], int]:
    magic, version = struct.unpack_from("<ii", buf, 0)
    if magic != MAGIC:
        raise ValueError("not an OpenEXR file (bad magic number)")
    if version & 0xFF != 2 or version & ~0xFF:
        raise ValueError(
            f"unsupported OpenEXR version/flags {version:#x}; only single-part scanline "
            "files are supported"
        )
    pos = 8
    attributes: Dict[str, Tuple[str, bytes]] = {}
    while buf[pos] != 0:
        name, pos = _read_cstring(buf, pos)
        type_name, pos = _read_cstring(buf, pos)
        (size,) = struct.unpack_from("<i", buf, pos)
        pos += 4
        attributes[name] = (type_name, buf[pos:pos + size])
        pos += size
    return attributes, pos + 1


def _parse_channels(value: bytes) -> List[Tuple[str, int]]:
    channels = []
    pos = 0
    while value[pos] != 0:
        name, pos = _read_cstring(value, pos)
        pixel_type, = struct.unpack_from("<i", value, pos)
        x_sampling, y_sampling = struct.unpack_from("<ii", value, pos + 8)
        if (x_sampling, y_sampling) != (1, 1):
            raise ValueError(f"subsampled EXR channel {name!r} is not supported")
        channels.append((name, pixel_type))
        pos += 16
    return channels


def read_exr(path: str | Path) -> Dict[str, np.ndarray]:
    """Read a single-part scanline EXR into ``{channel_name: HxW float32}``."""
    buf = Path(path).read_bytes()
    attributes, pos = _parse_header(buf)
    try:
        channels = _parse_channels(attributes["channels"][1])
        (compression,) = struct.unpack("<B", attributes["compression"][1])
        x_min, y_min, x_max, y_max = struct.unpack("<iiii", attributes["dataWindow"][1])
    except KeyError as exc:
        raise ValueError(f"EXR header is missing required attribute {exc}") from exc
    if compression not in _LINES_PER_CHUNK:
        raise ValueError(f"unsupported EXR compression {compression}; expected NONE/ZIPS/ZIP")
    for name, pixel_type in channels:
        if pixel_type != _PIXEL_FLOAT:
            raise ValueError(f"EXR channel {name!r} is not 32-bit FLOAT")
    (line_order,) = struct.unpack("<B", attributes["lineOrder"][1])
    if line_order != 0:
        raise ValueError("only INCREASING_Y EXR line order is supported")

    width, height = x_max - x_min + 1, y_max - y_min + 1
    if width <= 0 or height <= 0:
        raise ValueError("EXR dataWindow is empty")
    lines_per_chunk = _LINES_PER_CHUNK[compression]
    chunk_count = (height + lines_per_chunk - 1) // lines_per_chunk
    offsets = struct.unpack_from("<%dQ" % chunk_count, buf, pos)

    planes = np.empty((height, len(channels), width), dtype="<f4")
    line_bytes = 4 * width * len(channels)
    for offset in offsets:
        y, size = struct.unpack_from("<ii", buf, offset)
        start = y - y_min
        stop = min(start + lines_per_chunk, height)
        expected = line_bytes * (stop - start)
        payload = buf[offset + 8:offset + 8 + size]
        if len(payload) != size:
            raise ValueError("truncated EXR chunk")
        raw = payload if size == expected else _zip_decode(payload, expected)
        planes[start:stop] = np.frombuffer(raw, dtype="<f4").reshape(
            stop - start, len(channels), width
        )
    return {name: np.ascontiguousarray(planes[:, index]) for index, (name, _) in enumerate(channels)}


def read_rgb32_exr(path: str | Path) -> np.ndarray:
    """Read an EXR with ``R``/``G``/``B`` float32 channels as an ``HxWx3`` array."""
    planes = read_exr(path)
    missing = [name for name in "RGB" if name not in planes]
    if missing:
        raise ValueError(f"EXR at {path} lacks channel(s) {missing}; found {sorted(planes)}")
    return np.stack([planes["R"], planes["G"], planes["B"]], axis=-1).astype(np.float32)


__all__ = ["MAGIC", "VERSION", "read_exr", "read_rgb32_exr", "write_rgb32_exr"]
