"""Scientific image loading for TIFF, OME-TIFF and NumPy ``.npy`` files.

Loaders keep the native dtype and values, the axis labels the file declares, and any pixel
calibration or channel names it carries. Nothing is rescaled or reordered. Anything that cannot be
represented faithfully raises :class:`ImageLoadError` rather than being approximated.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree

import numpy as np
import numpy.typing as npt
import tifffile

from microflow.config import AXIS_LETTERS
from microflow.units import Calibration, LengthUnit

TIFF_SUFFIXES = frozenset({".tif", ".tiff"})

# TIFF ResolutionUnit codes that define a physical length, expressed in micrometres.
_RESOLUTION_UNIT_MICROMETRES = {2: 25_400.0, 3: 10_000.0}  # 2 = inch, 3 = centimetre

_OME_LENGTH_UNITS: dict[str, LengthUnit] = {
    "\u00b5m": "um",
    "\u03bcm": "um",
    "um": "um",
    "micron": "um",
    "nm": "nm",
    "mm": "mm",
    "m": "m",
}
_OME_DEFAULT_UNIT = "\u00b5m"  # OME-XML default when no PhysicalSize unit is declared


class ImageLoadError(Exception):
    """An image cannot be loaded faithfully: it is malformed, unsupported, or inconsistent."""


@dataclass(frozen=True)
class LoadedImage:
    data: npt.NDArray[Any]
    axes: str
    calibration: Calibration | None
    channels: tuple[str, ...] | None
    source: str
    storage: Literal["memory", "memmap"]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    def axis_size(self, axis: str) -> int:
        return self.shape[self.axes.index(axis)]


def load_image(
    path: Path,
    *,
    axes: str | None = None,
    calibration: Calibration | None = None,
    series: int | None = None,
    lazy: bool = False,
) -> LoadedImage:
    """Load a TIFF, OME-TIFF or ``.npy`` file.

    ``axes`` is checked against a TIFF's declared axes and is required for ``.npy`` files, which
    carry no labels. ``calibration`` overrides any calibration found in the file. ``series``
    selects one series of a multi-series TIFF and must be given when there is more than one.
    With ``lazy=True`` the pixels are memory-mapped where the storage allows it; otherwise they
    are read into memory and ``storage`` says which was used.
    """
    suffix = path.suffix.lower()
    if suffix in TIFF_SUFFIXES:
        return _load_tiff(path, axes=axes, calibration=calibration, series=series, lazy=lazy)
    if suffix == ".npy":
        return _load_npy(path, axes=axes, calibration=calibration, series=series, lazy=lazy)
    raise ImageLoadError(
        f"{path}: unsupported image format {suffix!r}; supported formats are .tif, .tiff and .npy"
    )


def load_array(
    array: npt.ArrayLike,
    *,
    axes: str,
    calibration: Calibration | None = None,
    channels: Sequence[str] | None = None,
) -> LoadedImage:
    """Wrap an in-memory array without copying it."""
    data = np.asarray(array)
    _validate(axes, data.shape, data.dtype, channels, "<array>")
    return LoadedImage(
        data=data,
        axes=axes,
        calibration=calibration,
        channels=_as_tuple(channels),
        source="<array>",
        storage="memory",
    )


def parse_ome_pixels(
    xml: str, index: int, filename: str
) -> tuple[Calibration | None, tuple[str, ...] | None]:
    """Read calibration and channel names for one OME image from its OME-XML description."""
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise ImageLoadError(f"{filename}: malformed OME-XML: {exc}") from exc
    pixels = [element for element in root.iter() if _local_name(element) == "Pixels"]
    if index >= len(pixels):
        raise ImageLoadError(
            f"{filename}: series {index} has no matching OME Pixels element ({len(pixels)} present)"
        )
    selected = pixels[index]
    for element in selected.iter():
        if _local_name(element) == "UUID" and element.get("FileName", filename) != filename:
            raise ImageLoadError(
                f"{filename}: multi-file OME-TIFF (data in {element.get('FileName')!r}) "
                "is not supported"
            )
    return _ome_calibration(selected, filename), _ome_channels(selected)


def _load_tiff(
    path: Path,
    *,
    axes: str | None,
    calibration: Calibration | None,
    series: int | None,
    lazy: bool,
) -> LoadedImage:
    try:
        with tifffile.TiffFile(path) as tif:
            index = _select_series(path, len(tif.series), series)
            selected = tif.series[index]
            file_axes = selected.axes
            shape = tuple(selected.shape)
            ome_xml = tif.ome_metadata if tif.is_ome else None
            if ome_xml is not None:
                file_calibration, channels = parse_ome_pixels(ome_xml, index, path.name)
            else:
                file_calibration = _resolution_calibration(selected.pages[0], path)
                channels = None
            if axes is not None and axes != file_axes:
                raise ImageLoadError(
                    f"{path}: requested axes {axes!r} but the file declares {file_axes!r}"
                )
            _validate(file_axes, shape, selected.dtype, channels, str(path))
            data, storage = _read_pixels(path, index, shape, lazy)
    except (OSError, EOFError, ValueError, IndexError, tifffile.TiffFileError) as exc:
        raise ImageLoadError(f"{path}: cannot read TIFF: {exc}") from exc

    return LoadedImage(
        data=data,
        axes=file_axes,
        calibration=calibration if calibration is not None else file_calibration,
        channels=_as_tuple(channels),
        source=str(path),
        storage=storage,
    )


def _load_npy(
    path: Path,
    *,
    axes: str | None,
    calibration: Calibration | None,
    series: int | None,
    lazy: bool,
) -> LoadedImage:
    if series is not None:
        raise ImageLoadError(f"{path}: series applies only to TIFF files")
    if axes is None:
        raise ImageLoadError(f"{path}: .npy files carry no axis labels; pass axes=...")
    try:
        data = np.load(path, mmap_mode="r" if lazy else None, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ImageLoadError(f"{path}: cannot read NumPy file: {exc}") from exc
    if not isinstance(data, np.ndarray):
        raise ImageLoadError(f"{path}: expected a single array in .npy file")
    _validate(axes, data.shape, data.dtype, None, str(path))
    return LoadedImage(
        data=data,
        axes=axes,
        calibration=calibration,
        channels=None,
        source=str(path),
        storage="memmap" if lazy else "memory",
    )


def _select_series(path: Path, count: int, series: int | None) -> int:
    if series is None:
        if count != 1:
            raise ImageLoadError(
                f"{path}: file contains {count} series; pass series=<index> to select one"
            )
        return 0
    if not 0 <= series < count:
        raise ImageLoadError(f"{path}: series {series} is out of range; file has {count}")
    return series


def _read_pixels(
    path: Path, index: int, shape: tuple[int, ...], lazy: bool
) -> tuple[npt.NDArray[Any], Literal["memory", "memmap"]]:
    if lazy:
        try:
            data: npt.NDArray[Any] = tifffile.memmap(path, series=index)
        except ValueError:
            # Compressed or otherwise non-contiguous data cannot be memory-mapped.
            data = tifffile.imread(path, series=index)
            storage: Literal["memory", "memmap"] = "memory"
        else:
            storage = "memmap"
    else:
        data = tifffile.imread(path, series=index)
        storage = "memory"
    if tuple(data.shape) != shape:
        raise ImageLoadError(
            f"{path}: pixel data has shape {tuple(data.shape)} but metadata declares {shape}"
        )
    return data, storage


def _validate(
    axes: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    channels: Sequence[str] | None,
    source: str,
) -> None:
    unknown = sorted(set(axes) - AXIS_LETTERS)
    if unknown:
        hint = " (RGB or other sample axes are not supported; split them into channels)"
        raise ImageLoadError(
            f"{source}: unsupported axis {unknown} in {axes!r}; supported axes are T, C, Z, Y, X"
            + (hint if "S" in unknown else "")
        )
    if not {"Y", "X"} <= set(axes) or len(set(axes)) != len(axes):
        raise ImageLoadError(
            f"{source}: axes must be unique letters including Y and X, got {axes!r}"
        )
    if len(axes) != len(shape):
        raise ImageLoadError(
            f"{source}: axes {axes!r} describe {len(axes)} dimensions but the data has "
            f"{len(shape)} (shape {shape})"
        )
    if dtype.kind not in "iuf":
        raise ImageLoadError(
            f"{source}: dtype {dtype} is not an intensity type; expected integer or floating point"
        )
    if channels is not None:
        count = shape[axes.index("C")] if "C" in axes else 1
        if len(channels) != count:
            raise ImageLoadError(
                f"{source}: {len(channels)} channel name(s) given for {count} channel(s)"
            )
        if len(set(channels)) != len(channels):
            raise ImageLoadError(f"{source}: channel names must be unique, got {list(channels)}")


def _resolution_calibration(page: Any, path: Path) -> Calibration | None:
    unit_tag = page.tags.get("ResolutionUnit")
    x_tag = page.tags.get("XResolution")
    y_tag = page.tags.get("YResolution")
    if unit_tag is None or x_tag is None or y_tag is None:
        return None
    unit_micrometres = _RESOLUTION_UNIT_MICROMETRES.get(int(unit_tag.value))
    if unit_micrometres is None:
        return None
    x_numerator, x_denominator = x_tag.value
    y_numerator, y_denominator = y_tag.value
    if x_numerator <= 0 or y_numerator <= 0:
        return None
    x_size = unit_micrometres * x_denominator / x_numerator
    y_size = unit_micrometres * y_denominator / y_numerator
    if not math.isclose(x_size, y_size, rel_tol=1e-6):
        raise ImageLoadError(
            f"{path}: anisotropic pixels (x={x_size} um, y={y_size} um) are not supported"
        )
    return Calibration(pixel_size=x_size, unit="um")


def _ome_calibration(pixels: ElementTree.Element, filename: str) -> Calibration | None:
    size_x = pixels.get("PhysicalSizeX")
    size_y = pixels.get("PhysicalSizeY")
    if size_x is None and size_y is None:
        return None
    if size_x is None or size_y is None:
        raise ImageLoadError(f"{filename}: PhysicalSizeX and PhysicalSizeY must both be set")
    unit_name = pixels.get("PhysicalSizeXUnit", _OME_DEFAULT_UNIT)
    if pixels.get("PhysicalSizeYUnit", _OME_DEFAULT_UNIT) != unit_name:
        raise ImageLoadError(f"{filename}: PhysicalSizeX and PhysicalSizeY use different units")
    unit = _OME_LENGTH_UNITS.get(unit_name)
    if unit is None:
        raise ImageLoadError(f"{filename}: unsupported physical unit {unit_name!r}")
    try:
        x_size, y_size = float(size_x), float(size_y)
        if not math.isclose(x_size, y_size, rel_tol=1e-9):
            raise ImageLoadError(
                f"{filename}: anisotropic pixels (x={x_size}, y={y_size}) are not supported"
            )
        return Calibration(pixel_size=x_size, unit=unit)
    except ValueError as exc:
        raise ImageLoadError(f"{filename}: invalid physical pixel size: {exc}") from exc


def _ome_channels(pixels: ElementTree.Element) -> tuple[str, ...] | None:
    channels = [child for child in pixels if _local_name(child) == "Channel"]
    if not channels:
        return None
    labels = [child.get("Name") or child.get("ID") for child in channels]
    if None in labels:
        return None
    return tuple(str(label) for label in labels)


def _local_name(element: ElementTree.Element) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def _as_tuple(channels: Sequence[str] | None) -> tuple[str, ...] | None:
    return tuple(channels) if channels is not None else None
