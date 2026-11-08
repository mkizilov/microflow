"""Per-object geometry and intensity measurements on one labelled Y-X plane.

Coordinates are measured from the top-left corner of the image. Pixel (row, col) covers
[row, row + 1) x [col, col + 1), so its centre is at (row + 0.5, col + 0.5). With a calibration
every length is multiplied by the pixel size and every area by its square. Without one, lengths
are in pixels and the table's length unit is ``px``. See docs/measurements.md for definitions.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from skimage import measure

from microflow.quantification.records import (
    GeometryRecord,
    IntensityRecord,
    MeasurementTable,
    ObjectRecord,
    Source,
)
from microflow.units import Calibration


class MeasurementError(ValueError):
    """Measurements cannot be made faithfully from the given inputs."""


@dataclass(frozen=True)
class IntensitySource:
    """An intensity image on the label grid, as planes shaped (channels, Y, X)."""

    name: Source
    planes: npt.NDArray[np.float64]
    channels: tuple[str, ...]


def measure_plane(
    labels: npt.ArrayLike,
    sources: Sequence[IntensitySource],
    calibration: Calibration | None,
) -> MeasurementTable:
    """Measure every object in ``labels`` (0 is background) in each intensity source."""
    label_plane = np.asarray(labels)
    if label_plane.ndim != 2:
        raise MeasurementError(
            f"expected one Y-X label plane, got shape {label_plane.shape}; "
            "measure planes one at a time"
        )
    if not np.issubdtype(label_plane.dtype, np.integer):
        raise MeasurementError(f"label image must have an integer dtype, got {label_plane.dtype}")
    if not sources:
        raise MeasurementError("at least one intensity source is required")
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise MeasurementError(f"intensity sources must be distinct, got {names}")
    channels = sources[0].channels
    for source in sources:
        if source.planes.ndim != 3 or source.planes.shape[1:] != label_plane.shape:
            raise MeasurementError(
                f"{source.name}: planes have shape {source.planes.shape}, expected "
                f"(channels, {label_plane.shape[0]}, {label_plane.shape[1]})"
            )
        if len(source.channels) != source.planes.shape[0]:
            raise MeasurementError(
                f"{source.name}: {len(source.channels)} channel name(s) for "
                f"{source.planes.shape[0]} channel(s)"
            )
        if source.channels != channels:
            raise MeasurementError("all intensity sources must share the same channel names")

    scale = calibration.pixel_size if calibration is not None else 1.0
    unit = calibration.unit if calibration is not None else "px"
    objects = tuple(
        _measure_object(region, sources, scale) for region in measure.regionprops(label_plane)
    )
    return MeasurementTable(
        length_unit=unit,
        channels=channels,
        sources=tuple(names),
        objects=objects,
    )


def _measure_object(region: Any, sources: Sequence[IntensitySource], scale: float) -> ObjectRecord:
    area_px = float(region.area)
    row_min, col_min, row_max, col_max = region.bbox
    geometry = GeometryRecord(
        label=int(region.label),
        area=area_px * scale**2,
        perimeter=_perimeter_px(region.image) * scale,
        centroid_row=(region.centroid[0] + 0.5) * scale,
        centroid_col=(region.centroid[1] + 0.5) * scale,
        bbox_min_row=row_min * scale,
        bbox_min_col=col_min * scale,
        bbox_max_row=row_max * scale,
        bbox_max_col=col_max * scale,
        eccentricity=float(region.eccentricity),
        equivalent_diameter=float(np.sqrt(4.0 * area_px / np.pi)) * scale,
    )
    intensities = tuple(
        _intensity(source, index, region.coords, scale)
        for source in sources
        for index in range(len(source.channels))
    )
    return ObjectRecord(geometry=geometry, intensities=intensities)


def _perimeter_px(mask: npt.NDArray[np.bool_]) -> float:
    """Length of the 0.5-level iso-contour of the object mask, in pixels.

    The contour runs midway between object and background pixel centres. A square of s x s
    pixels therefore has perimeter 4(s - 1) + 2*sqrt(2).
    """
    padded = np.pad(mask, 1).astype(np.float64)
    contours = measure.find_contours(padded, 0.5)
    return float(sum(np.hypot(*np.diff(contour, axis=0).T).sum() for contour in contours))


def _intensity(
    source: IntensitySource, channel: int, coords: npt.NDArray[np.intp], scale: float
) -> IntensityRecord:
    values = source.planes[channel][coords[:, 0], coords[:, 1]]
    finite = values[np.isfinite(values)]
    name = source.channels[channel]
    if finite.size == 0:
        nan = float("nan")
        return IntensityRecord(
            source=source.name,
            channel=name,
            valid_pixels=0,
            mean=nan,
            median=nan,
            std=nan,
            minimum=nan,
            maximum=nan,
            integrated=nan,
        )
    return IntensityRecord(
        source=source.name,
        channel=name,
        valid_pixels=int(finite.size),
        mean=float(finite.mean()),
        median=float(np.median(finite)),
        std=float(finite.std()),
        minimum=float(finite.min()),
        maximum=float(finite.max()),
        integrated=float(finite.sum()) * scale**2,
    )
