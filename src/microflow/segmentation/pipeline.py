"""Classical segmentation of images into labelled objects, driven by SegmentationConfig.

Order of operations for each Y-X plane:
threshold, optional opening, optional closing, connected components, optional watershed split,
removal of objects below the minimum area, optional removal of border-touching objects, and
raster-order numbering. Pixels that are NaN are background throughout.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from microflow.config import SegmentationConfig
from microflow.plugins.contracts import call, check_labels
from microflow.plugins.registry import configure
from microflow.segmentation import algorithms as alg
from microflow.units import Calibration, area_to_pixels, to_pixels


@dataclass(frozen=True)
class SegmentationResult:
    """Label image with the same axes as the input. Label 0 is background; labels are numbered
    1..N within each Y-X plane, in raster order of their first pixel."""

    labels: npt.NDArray[np.int32]
    axes: str
    object_counts: tuple[int, ...]
    thresholds: tuple[float | None, ...]


def segment_plane(
    plane: npt.NDArray[np.float64],
    config: SegmentationConfig,
    calibration: Calibration | None,
) -> tuple[npt.NDArray[np.int32], float | None]:
    """Segment one Y-X plane. Returns the labels and the global threshold used, if any."""
    if config.method == "plugin":
        return _segment_with_plugin(plane, config, calibration), None
    block_size = 0
    if config.method == "adaptive":
        block_size = _odd_block_size(to_pixels(config.adaptive_block_size, calibration))
    mask, level = alg.foreground_mask(
        plane, config.method, config.threshold, block_size, config.adaptive_offset
    )
    if config.opening_radius is not None:
        mask = alg.open_mask(mask, to_pixels(config.opening_radius, calibration))
    if config.closing_radius is not None:
        mask = alg.close_mask(mask, to_pixels(config.closing_radius, calibration))

    labels = alg.label_objects(mask)
    if config.watershed:
        min_distance = to_pixels(config.watershed_min_distance, calibration)
        labels = alg.split_touching(mask, labels, min_distance)
    labels = alg.remove_small(labels, area_to_pixels(config.min_object_area, calibration))
    if config.exclude_border:
        labels = alg.remove_border(labels)
    return alg.relabel_in_raster_order(labels), level


def segment_image(
    data: npt.ArrayLike,
    axes: str,
    config: SegmentationConfig,
    calibration: Calibration | None,
) -> SegmentationResult:
    """Segment every Y-X plane of ``data``, whatever the other axes are."""
    array = np.asarray(data, dtype=np.float64)
    spatial = (axes.index("Y"), axes.index("X"))
    planes = np.moveaxis(array, spatial, (-2, -1))
    labels = np.zeros(planes.shape, dtype=np.int32)
    counts: list[int] = []
    levels: list[float | None] = []
    for index in np.ndindex(planes.shape[:-2]):
        plane_labels, level = segment_plane(planes[index], config, calibration)
        labels[index] = plane_labels
        counts.append(int(plane_labels.max(initial=0)))
        levels.append(level)
    return SegmentationResult(
        labels=np.moveaxis(labels, (-2, -1), spatial),
        axes=axes,
        object_counts=tuple(counts),
        thresholds=tuple(levels),
    )


def _segment_with_plugin(
    plane: npt.NDArray[np.float64],
    config: SegmentationConfig,
    calibration: Calibration | None,
) -> npt.NDArray[np.int32]:
    if config.plugin is None:
        raise ValueError("method 'plugin' needs a plugin to be named")
    configured = configure("segmenter", config.plugin)
    label = f"segmenter plugin {config.plugin.name!r}"
    output = call(
        "segmenter",
        config.plugin.name,
        "segment",
        configured.plugin.segment,
        np.array(plane, dtype=np.float64),
        calibration,
        configured.options,
    )
    labels = check_labels(output, plane.shape, label)
    labels = alg.remove_small(labels, area_to_pixels(config.min_object_area, calibration))
    if config.exclude_border:
        labels = alg.remove_border(labels)
    return alg.relabel_in_raster_order(labels)


def _odd_block_size(pixels: float) -> int:
    size = max(3, round(pixels))
    return size if size % 2 else size + 1
