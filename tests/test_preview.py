import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from microflow.reporting.preview import first_plane, write_before_after_png, write_png_grey8


def _decode_png(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    position = 8
    width = height = 0
    idat = b""
    while position < len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        tag = data[position + 4 : position + 8]
        payload = data[position + 8 : position + 8 + length]
        checksum = data[position + 8 + length : position + 12 + length]
        assert struct.unpack(">I", checksum)[0] == zlib.crc32(tag + payload) & 0xFFFFFFFF
        if tag == b"IHDR":
            width, height, bit_depth, colour_type = struct.unpack(">IIBB", payload[:10])
            assert (bit_depth, colour_type) == (8, 0)
        if tag == b"IDAT":
            idat += payload
        position += 12 + length
    return width, height, zlib.decompress(idat)


def test_png_writer_produces_decodable_grey_image(tmp_path: Path) -> None:
    pixels = np.array([[0, 128, 255], [10, 20, 30]], dtype=np.uint8)
    path = tmp_path / "grey.png"
    write_png_grey8(path, pixels)
    width, height, raw = _decode_png(path)
    assert (width, height) == (3, 2)
    assert raw == b"\x00\x00\x80\xff" + b"\x00\x0a\x14\x1e"


def test_png_writer_rejects_wrong_dtype(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2-D uint8"):
        write_png_grey8(tmp_path / "bad.png", np.zeros((2, 2), dtype=np.float64))


def test_before_after_panel_has_expected_dimensions(tmp_path: Path) -> None:
    before = np.linspace(0, 1000, 20 * 30).reshape(20, 30)
    after = before / 2
    path = tmp_path / "panel.png"
    write_before_after_png(path, before, after)
    width, height, _ = _decode_png(path)
    assert (width, height) == (30 + 4 + 30, 20)


def test_before_and_after_share_the_display_range(tmp_path: Path) -> None:
    before = np.full((4, 4), 100.0)
    before[0, 0] = 0.0
    before[-1, -1] = 200.0
    after = np.full((4, 4), 50.0)
    path = tmp_path / "shared.png"
    write_before_after_png(path, before, after)
    _, _, raw = _decode_png(path)
    # Each scanline is one filter byte, then 4 before pixels, 4 gap pixels, 4 after pixels.
    row = raw[13:26]
    # Display range from `before` is 7.5..192.5, so 100 maps to ~128 and 50 to ~59.
    assert abs(row[2] - 128) <= 1
    assert abs(row[10] - 59) <= 1


def test_preview_planes_must_match_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="differ in shape"):
        write_before_after_png(tmp_path / "x.png", np.zeros((4, 4)), np.zeros((4, 5)))


def test_nan_pixels_render_black(tmp_path: Path) -> None:
    before = np.array([[np.nan, 1.0], [2.0, 3.0]])
    path = tmp_path / "nan.png"
    write_before_after_png(path, before, before)
    _, _, raw = _decode_png(path)
    assert raw[1] == 0


def test_first_plane_selects_yx_plane_in_row_column_order() -> None:
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    plane = first_plane(data, "CYX")
    assert plane.shape == (3, 4)
    np.testing.assert_array_equal(plane, data[0])
    transposed = first_plane(data.transpose(0, 2, 1), "CXY")
    np.testing.assert_array_equal(transposed, data[0])
