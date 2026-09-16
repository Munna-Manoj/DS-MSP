"""DS-MSP's dependency-free OpenEXR writer/reader (ds_msp.isaac_sim.exr).

The round-trip tests always run. The cross-checks against a reference OpenEXR
implementation run only where one is installed (the ``OpenEXR`` bindings, or an
OpenCV build with the OpenEXR codec) and are skipped otherwise, because the
opencv-python wheels inside the project's dependency range ship without it.
"""

import os
import struct

import numpy as np
import pytest

from ds_msp.isaac_sim.exr import MAGIC, VERSION, read_exr, read_rgb32_exr, write_rgb32_exr

SIZES = [(1, 1), (7, 5), (47, 35), (64, 17), (33, 16), (40, 48)]


def _sample(height, width, seed=0):
    rng = np.random.default_rng(seed)
    rgb = rng.uniform(-3.0, 3.0, size=(height, width, 3)).astype(np.float32)
    # Include exact zeros and repeated rows so the ZIP predictor sees compressible data.
    rgb[:, : max(1, width // 3), 2] = 0.0
    if height > 2:
        rgb[1] = rgb[0]
    return rgb


@pytest.mark.parametrize("compression", ["zip", "none"])
@pytest.mark.parametrize("size", SIZES, ids=lambda size: f"{size[1]}x{size[0]}")
def test_round_trip_is_bit_exact(tmp_path, size, compression):
    height, width = size
    rgb = _sample(height, width)
    path = tmp_path / f"{compression}.exr"
    write_rgb32_exr(path, rgb, compression=compression)
    assert np.array_equal(read_rgb32_exr(path), rgb)
    assert not list(tmp_path.glob("*.tmp*")), "temporary file must be renamed away"


def test_header_is_a_single_part_scanline_exr(tmp_path):
    path = tmp_path / "header.exr"
    write_rgb32_exr(path, _sample(3, 4))
    head = path.read_bytes()[:8]
    assert struct.unpack("<ii", head) == (MAGIC, VERSION)
    planes = read_exr(path)
    assert sorted(planes) == ["B", "G", "R"]
    assert all(plane.shape == (3, 4) and plane.dtype == np.float32 for plane in planes.values())


def test_zip_compresses_and_falls_back_to_raw_chunks(tmp_path):
    compressible = np.zeros((32, 64, 3), np.float32)
    rng = np.random.default_rng(1)
    incompressible = rng.uniform(-1e6, 1e6, size=(32, 64, 3)).astype(np.float32)
    write_rgb32_exr(tmp_path / "flat.exr", compressible)
    write_rgb32_exr(tmp_path / "noise.exr", incompressible)
    write_rgb32_exr(tmp_path / "raw.exr", incompressible, compression="none")
    assert (tmp_path / "flat.exr").stat().st_size < 32 * 64 * 3 * 4 // 10
    assert np.array_equal(read_rgb32_exr(tmp_path / "noise.exr"), incompressible)
    assert np.array_equal(read_rgb32_exr(tmp_path / "raw.exr"), incompressible)


def test_rejects_wrong_shapes_and_unknown_compression(tmp_path):
    with pytest.raises(ValueError, match="HxWx3"):
        write_rgb32_exr(tmp_path / "bad.exr", np.zeros((4, 4), np.float32))
    with pytest.raises(ValueError, match="compression"):
        write_rgb32_exr(tmp_path / "bad.exr", np.zeros((4, 4, 3), np.float32), compression="rle")
    (tmp_path / "junk.exr").write_bytes(b"\0" * 64)
    with pytest.raises(ValueError, match="magic"):
        read_exr(tmp_path / "junk.exr")


def _reference_openexr_reader():
    try:
        import OpenEXR  # noqa: F401
    except ImportError:
        return None

    def read(path):
        with OpenEXR.File(str(path), separate_channels=True) as handle:
            channels = handle.channels()
            return np.stack([channels[name].pixels for name in "RGB"], axis=-1)

    return read


def _reference_cv2_reader():
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    if "OpenEXR:" not in cv2.getBuildInformation() or "OpenEXR:                     NO" in (
        cv2.getBuildInformation()
    ):
        return None

    def read(path):
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return None if bgr is None else bgr[..., ::-1]

    return read


@pytest.mark.parametrize("compression", ["zip", "none"])
def test_reference_openexr_library_reads_our_files(tmp_path, compression):
    read = _reference_openexr_reader()
    if read is None:
        pytest.skip("reference OpenEXR bindings not installed")
    rgb = _sample(47, 35, seed=2)
    path = tmp_path / "ref.exr"
    write_rgb32_exr(path, rgb, compression=compression)
    assert np.array_equal(read(path), rgb)


@pytest.mark.parametrize("compression", ["zip", "none"])
def test_reference_opencv_openexr_codec_reads_our_files(tmp_path, compression):
    read = _reference_cv2_reader()
    if read is None:
        pytest.skip("this OpenCV build has no OpenEXR codec")
    rgb = _sample(47, 35, seed=3)
    path = tmp_path / "ref.exr"
    write_rgb32_exr(path, rgb, compression=compression)
    decoded = read(path)
    if decoded is None:
        pytest.skip("this OpenCV build could not decode EXR")
    assert np.array_equal(decoded, rgb)


def test_reads_reference_library_output(tmp_path):
    OpenEXR = pytest.importorskip("OpenEXR")
    rgb = _sample(21, 19, seed=4)
    path = tmp_path / "from_reference.exr"
    for compression in (OpenEXR.ZIP_COMPRESSION, OpenEXR.NO_COMPRESSION, OpenEXR.ZIPS_COMPRESSION):
        header = {"compression": compression, "type": OpenEXR.scanlineimage}
        # The binding's writer expects contiguous planes, not strided views.
        channels = {name: np.ascontiguousarray(rgb[..., index]) for index, name in enumerate("RGB")}
        with OpenEXR.File(header, channels) as handle:
            handle.write(str(path))
        assert np.array_equal(read_rgb32_exr(path), rgb), compression


pytestmark = pytest.mark.req("FR-INTEROP-003")
