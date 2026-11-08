"""Quality-control checks on segmentation results.

This module only reads results: the images, the label image, and the settings. The numerical
modules never import it, so the reports can change without affecting analysis.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from microflow.config import QualityControlConfig
from microflow.loading import LoadedImage
from microflow.units import Area, Calibration

FlagCode = Literal[
    "empty_mask",
    "excessive_coverage",
    "saturation",
    "invalid_pixels",
    "object_area_below",
    "object_area_above",
]


class QualityFlag(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: FlagCode
    plane: str
    message: str


class PlaneSummary(BaseModel):
    """Segmentation summary for one Y-X plane. Areas are in the report's area unit."""

    model_config = ConfigDict(frozen=True)

    plane: str
    total_pixels: int
    foreground_pixels: int
    foreground_fraction: float
    object_count: int
    area_min: float | None
    area_median: float | None
    area_mean: float | None
    area_max: float | None
    saturated_fraction: float | None
    invalid_fraction: float


class QualityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    source: str
    axes: str
    length_unit: str
    area_unit: str
    figure_plane: str
    area_min_limit: float | None
    area_max_limit: float | None
    planes: tuple[PlaneSummary, ...]
    flags: tuple[QualityFlag, ...]
    settings: QualityControlConfig


@dataclass(frozen=True)
class QualityInputs:
    """Images on one grid, split into planes: arrays are shaped (planes, Y, X)."""

    title: str
    source: str
    axes: str
    plane_names: tuple[str, ...]
    raw: npt.NDArray[np.float64]
    labels: npt.NDArray[Any]
    preprocessed: npt.NDArray[np.float64] | None
    raw_dtype: str
    calibration: Calibration | None
    settings: QualityControlConfig


def inputs_from_images(
    raw: LoadedImage,
    labels: LoadedImage,
    settings: QualityControlConfig,
    *,
    preprocessed: LoadedImage | None = None,
    title: str | None = None,
) -> QualityInputs:
    if labels.axes != raw.axes or labels.shape != raw.shape:
        raise ValueError(
            f"labels must share the raw image's axes and shape: labels {labels.axes!r} "
            f"{labels.shape}, raw {raw.axes!r} {raw.shape}"
        )
    if preprocessed is not None and (
        preprocessed.axes != raw.axes or preprocessed.shape != raw.shape
    ):
        raise ValueError(
            f"preprocessed image must share the raw image's axes and shape: "
            f"{preprocessed.axes!r} {preprocessed.shape}, raw {raw.axes!r} {raw.shape}"
        )
    raw_planes, names = _planes(raw)
    label_planes, _ = _planes(labels)
    preprocessed_planes = None
    if preprocessed is not None:
        preprocessed_planes = _planes(preprocessed)[0].astype(np.float64)
    return QualityInputs(
        title=title or f"Quality control: {raw.source}",
        source=raw.source,
        axes=raw.axes,
        plane_names=names,
        raw=raw_planes.astype(np.float64),
        labels=label_planes,
        preprocessed=preprocessed_planes,
        raw_dtype=str(raw.data.dtype),
        calibration=raw.calibration,
        settings=settings,
    )


def build_quality_report(inputs: QualityInputs) -> QualityReport:
    unit = inputs.calibration.unit if inputs.calibration is not None else "px"
    area_unit = f"{unit}2"
    min_limit, max_limit = _area_limits(inputs.settings, inputs.calibration)
    saturation = _saturation_level(inputs.settings, inputs.raw_dtype)
    scale_area = pixel_area(inputs.calibration)
    settings = inputs.settings

    summaries: list[PlaneSummary] = []
    flags: list[QualityFlag] = []
    object_counts: list[int] = []
    for index, name in enumerate(inputs.plane_names):
        labels = inputs.labels[index]
        raw_plane = inputs.raw[index]
        ids, areas = object_areas(labels, scale_area)
        foreground = int(np.count_nonzero(labels))
        fraction = foreground / labels.size
        saturated = None if saturation is None else float(np.mean(raw_plane >= saturation))
        checked = raw_plane if inputs.preprocessed is None else inputs.preprocessed[index]
        invalid = float(np.mean(~np.isfinite(checked)))
        summary = PlaneSummary(
            plane=name,
            total_pixels=int(labels.size),
            foreground_pixels=foreground,
            foreground_fraction=fraction,
            object_count=int(ids.size),
            area_min=_stat(areas, np.min),
            area_median=_stat(areas, np.median),
            area_mean=_stat(areas, np.mean),
            area_max=_stat(areas, np.max),
            saturated_fraction=saturated,
            invalid_fraction=invalid,
        )
        summaries.append(summary)
        object_counts.append(int(ids.size))

        if ids.size == 0:
            flags.append(
                QualityFlag(code="empty_mask", plane=name, message="no objects were segmented")
            )
        if fraction > settings.max_foreground_fraction:
            flags.append(
                QualityFlag(
                    code="excessive_coverage",
                    plane=name,
                    message=(
                        f"foreground covers {fraction:.1%} of the plane; "
                        f"the limit is {settings.max_foreground_fraction:.1%}"
                    ),
                )
            )
        if saturated is not None and saturated > settings.max_saturated_fraction:
            flags.append(
                QualityFlag(
                    code="saturation",
                    plane=name,
                    message=(
                        f"{saturated:.2%} of pixels are at or above {saturation:g}; "
                        f"the limit is {settings.max_saturated_fraction:.2%}"
                    ),
                )
            )
        if invalid > settings.max_invalid_fraction:
            flags.append(
                QualityFlag(
                    code="invalid_pixels",
                    plane=name,
                    message=(
                        f"{invalid:.2%} of pixels are NaN or infinite; "
                        f"the limit is {settings.max_invalid_fraction:.2%}"
                    ),
                )
            )
        if min_limit is not None:
            below = ids[areas < min_limit]
            if below.size:
                flags.append(
                    QualityFlag(
                        code="object_area_below",
                        plane=name,
                        message=(
                            f"{below.size} object(s) smaller than {min_limit:g} {area_unit}: "
                            f"labels {_label_list(below)}"
                        ),
                    )
                )
        if max_limit is not None:
            above = ids[areas > max_limit]
            if above.size:
                flags.append(
                    QualityFlag(
                        code="object_area_above",
                        plane=name,
                        message=(
                            f"{above.size} object(s) larger than {max_limit:g} {area_unit}: "
                            f"labels {_label_list(above)}"
                        ),
                    )
                )

    figure_plane = inputs.plane_names[int(np.argmax(object_counts))]
    return QualityReport(
        title=inputs.title,
        source=inputs.source,
        axes=inputs.axes,
        length_unit=unit,
        area_unit=area_unit,
        figure_plane=figure_plane,
        area_min_limit=min_limit,
        area_max_limit=max_limit,
        planes=tuple(summaries),
        flags=tuple(flags),
        settings=settings,
    )


def object_areas(
    labels: npt.NDArray[Any], area_per_pixel: float
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """Object label values and areas, from pixel counts. Background is excluded."""
    counts = np.bincount(np.asarray(labels, dtype=np.intp).ravel())
    counts[0] = 0
    ids = np.flatnonzero(counts)
    return ids, counts[ids] * area_per_pixel


def pixel_area(calibration: Calibration | None) -> float:
    return calibration.pixel_size**2 if calibration is not None else 1.0


def _planes(image: LoadedImage) -> tuple[npt.NDArray[Any], tuple[str, ...]]:
    lead = [index for index, axis in enumerate(image.axes) if axis not in "YX"]
    spatial = [image.axes.index("Y"), image.axes.index("X")]
    moved = np.transpose(np.asarray(image.data), lead + spatial)
    lead_axes = [image.axes[index] for index in lead]
    names = tuple(
        "".join(
            f"{axis}{position}"
            for axis, position in zip(lead_axes, position_index, strict=True)
        )
        or "image"
        for position_index in np.ndindex(moved.shape[: len(lead)])
    )
    return moved.reshape((-1,) + moved.shape[-2:]), names


def _area_limits(
    settings: QualityControlConfig, calibration: Calibration | None
) -> tuple[float | None, float | None]:
    def convert(area: Area | None) -> float | None:
        if area is None:
            return None
        if calibration is None:
            raise ValueError(
                "object area limits need pixel calibration; the image has none, "
                "so give its pixel size and unit"
            )
        return area.to(calibration.unit).value

    return convert(settings.min_object_area), convert(settings.max_object_area)


def _saturation_level(settings: QualityControlConfig, dtype: str) -> float | None:
    if settings.saturation_level is not None:
        return settings.saturation_level
    kind = np.dtype(dtype).kind
    return float(np.iinfo(dtype).max) if kind in "iu" else None


def _stat(values: npt.NDArray[np.float64], function: Any) -> float | None:
    return float(function(values)) if values.size else None


def _label_list(ids: npt.NDArray[np.intp], limit: int = 10) -> str:
    shown = ", ".join(str(value) for value in ids[:limit])
    return shown + (", ..." if ids.size > limit else "")
