"""Before/after preview images for visual inspection.

Previews use a display-only stretch and never change the data they are made from.
"""

import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_GAP_WIDTH = 4


def first_plane(data: npt.NDArray[Any], axes: str) -> npt.NDArray[Any]:
    """Return the first Y-X plane, with Y as rows and X as columns."""
    index = tuple(slice(None) if axis in "YX" else 0 for axis in axes)
    plane = np.asarray(data[index])
    return plane if axes.index("Y") < axes.index("X") else plane.T


def write_before_after_png(path: Path, before: npt.NDArray[Any], after: npt.NDArray[Any]) -> None:
    """Write two planes side by side as 8-bit grey.

    Both planes use the display range of ``before`` (its 0.5th to 99.5th percentile), so changes
    in background and intensity remain visible. NaN pixels are shown black.
    """
    if before.shape != after.shape:
        raise ValueError(f"preview planes differ in shape: {before.shape} and {after.shape}")
    reference = np.asarray(before, dtype=np.float64)
    low, high = (float(v) for v in np.nanpercentile(reference, [0.5, 99.5]))
    if high <= low:
        high = low + 1.0
    gap = np.full((before.shape[0], _GAP_WIDTH), 255, dtype=np.uint8)
    panel = np.hstack([_to_grey8(before, low, high), gap, _to_grey8(after, low, high)])
    write_png_grey8(path, panel)


def write_png_grey8(path: Path, pixels: npt.NDArray[np.uint8]) -> None:
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError("PNG writer expects a 2-D uint8 array")
    height, width = pixels.shape
    scanlines = b"".join(b"\x00" + row.tobytes() for row in pixels)
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(
        _PNG_SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(scanlines))
        + _chunk(b"IEND", b"")
    )


def _to_grey8(plane: npt.NDArray[Any], low: float, high: float) -> npt.NDArray[np.uint8]:
    scaled = (np.asarray(plane, dtype=np.float64) - low) / (high - low)
    scaled = np.clip(np.nan_to_num(scaled, nan=0.0), 0.0, 1.0)
    return np.round(scaled * 255).astype(np.uint8)


def _chunk(tag: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", checksum)
