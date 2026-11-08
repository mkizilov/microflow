"""Typed measurement records and their tabular (CSV) form.

Geometry and intensity are separate types. Intensity records carry their source ("raw" or
"corrected") and channel, so intensities from different images never share a column.

Column names carry the length unit: ``px`` when no pixel calibration was available, otherwise
the calibration unit (for example ``um``). Areas use the square of that unit (``um2``).
"""

import csv
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Source = Literal["raw", "corrected"]

_DIMENSIONLESS = frozenset({"label", "eccentricity"})
_INTENSITY_FIELDS = ("valid_pixels", "mean", "median", "std", "minimum", "maximum", "integrated")
_INTENSITY_STEMS = {"minimum": "min", "maximum": "max"}


class GeometryRecord(BaseModel):
    """Shape of one object. Lengths are in the table's length unit; area is its square."""

    model_config = ConfigDict(frozen=True)

    label: int
    area: float
    perimeter: float
    centroid_row: float
    centroid_col: float
    bbox_min_row: float
    bbox_min_col: float
    bbox_max_row: float
    bbox_max_col: float
    eccentricity: float
    equivalent_diameter: float


class IntensityRecord(BaseModel):
    """Intensity statistics of one object, in one channel of one source, over finite pixels.

    ``integrated`` is the sum of intensities times the area of one pixel, in the table's
    length unit squared. Without calibration it is the plain sum.
    """

    model_config = ConfigDict(frozen=True)

    source: Source
    channel: str
    valid_pixels: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    integrated: float


class ObjectRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    geometry: GeometryRecord
    intensities: tuple[IntensityRecord, ...]


class MeasurementTable(BaseModel):
    """One row per object. ``length_unit`` is ``px`` or a calibration unit."""

    model_config = ConfigDict(frozen=True)

    length_unit: str
    channels: tuple[str, ...]
    sources: tuple[Source, ...]
    objects: tuple[ObjectRecord, ...]

    def columns(self) -> list[str]:
        names = [_geometry_column(field, self.length_unit) for field in GeometryRecord.model_fields]
        for source in self.sources:
            for channel in self.channels:
                names += [
                    _intensity_column(source, channel, field, self.length_unit)
                    for field in _INTENSITY_FIELDS
                ]
        return names

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for measured in self.objects:
            row: dict[str, Any] = {}
            for field, value in measured.geometry.model_dump().items():
                row[_geometry_column(field, self.length_unit)] = value
            for intensity in measured.intensities:
                for field in _INTENSITY_FIELDS:
                    column = _intensity_column(
                        intensity.source, intensity.channel, field, self.length_unit
                    )
                    row[column] = getattr(intensity, field)
            rows.append(row)
        return rows

    def write_csv(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns())
            writer.writeheader()
            writer.writerows(self.rows())


def _geometry_column(field: str, unit: str) -> str:
    if field in _DIMENSIONLESS:
        return field
    if field == "area":
        return f"area_{unit}2"
    return f"{field}_{unit}"


def _intensity_column(source: str, channel: str, field: str, unit: str) -> str:
    if field == "integrated":
        stem = f"integrated_intensity_{unit}2"
    else:
        stem = _INTENSITY_STEMS.get(field, field)
    return f"{source}_{channel}_{stem}"
