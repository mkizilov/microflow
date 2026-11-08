"""Quality-control figures, rendered headless to PNG bytes.

Figures are built with the object-oriented matplotlib API on the Agg canvas. ``pyplot`` is never
imported, so no display or GUI backend is needed and batch or CI runs behave identically.
Figures show one Y-X plane, the plane with the most objects. The summary table covers all planes.
"""

import io
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries

from microflow.reporting.qc import QualityInputs, QualityReport, object_areas, pixel_area

FIGURE_TITLES = {
    "original": "Original image",
    "preprocessed": "Preprocessed image",
    "overlay": "Segmentation overlay",
    "labels": "Object labels",
    "intensity": "Intensity distribution",
    "area": "Object area distribution",
}

_HISTOGRAM_BINS = 64
_AREA_BINS = 20
_DPI = 100


def render_figures(inputs: QualityInputs, report: QualityReport) -> dict[str, bytes]:
    """Render every figure as PNG bytes, in the order of :data:`FIGURE_TITLES`."""
    index = inputs.plane_names.index(report.figure_plane)
    raw = inputs.raw[index]
    labels = inputs.labels[index]
    preprocessed = None if inputs.preprocessed is None else inputs.preprocessed[index]
    object_count = report.planes[index].object_count

    figures: dict[str, bytes] = {"original": _image_png(raw, FIGURE_TITLES["original"])}
    if preprocessed is not None:
        figures["preprocessed"] = _image_png(preprocessed, FIGURE_TITLES["preprocessed"])
    figures["overlay"] = _overlay_png(raw, labels, object_count)
    figures["labels"] = _labels_png(labels)
    figures["intensity"] = _intensity_png(raw, preprocessed, labels)
    _, areas = object_areas(labels, pixel_area(inputs.calibration))
    figures["area"] = _area_png(areas, report)
    return figures


def _figure(width: float, height: float) -> Figure:
    figure = Figure(figsize=(width, height), dpi=_DPI)
    FigureCanvasAgg(figure)
    return figure


def _png(figure: Figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    return buffer.getvalue()


def _display(plane: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Display-only stretch to [0, 1] between the 0.5th and 99.5th percentiles."""
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros(plane.shape)
    low, high = (float(value) for value in np.percentile(finite, [0.5, 99.5]))
    if high <= low:
        high = low + 1.0
    return np.clip(np.nan_to_num((plane - low) / (high - low), nan=0.0), 0.0, 1.0)


def _image_png(plane: npt.NDArray[np.float64], title: str) -> bytes:
    figure = _figure(5, 4.5)
    axes = figure.add_subplot()
    axes.imshow(_display(plane), cmap="gray", vmin=0, vmax=1)
    axes.set_title(title)
    axes.set_axis_off()
    return _png(figure)


def _overlay_png(raw: npt.NDArray[np.float64], labels: npt.NDArray[Any], count: int) -> bytes:
    rgb = np.repeat(_display(raw)[..., np.newaxis], 3, axis=2)
    rgb[find_boundaries(labels, mode="inner")] = (1.0, 0.0, 0.0)
    figure = _figure(5, 4.5)
    axes = figure.add_subplot()
    axes.imshow(rgb)
    axes.set_title(f"{FIGURE_TITLES['overlay']} ({count} objects)")
    axes.set_axis_off()
    return _png(figure)


def _labels_png(labels: npt.NDArray[Any]) -> bytes:
    figure = _figure(5, 4.5)
    axes = figure.add_subplot()
    axes.imshow(label2rgb(labels, bg_label=0))
    for value in np.unique(labels):
        if value == 0:
            continue
        rows, cols = np.nonzero(labels == value)
        axes.text(
            cols.mean(),
            rows.mean(),
            str(int(value)),
            color="white",
            fontsize=7,
            ha="center",
            va="center",
        )
    axes.set_title(FIGURE_TITLES["labels"])
    axes.set_axis_off()
    return _png(figure)


def _intensity_png(
    raw: npt.NDArray[np.float64],
    preprocessed: npt.NDArray[np.float64] | None,
    labels: npt.NDArray[Any],
) -> bytes:
    panels = [("Raw intensity", raw)]
    if preprocessed is not None:
        panels.append(("Preprocessed intensity", preprocessed))
    figure = _figure(6 * len(panels), 4)
    axes_row = figure.subplots(1, len(panels), squeeze=False)[0]
    for axes, (title, values) in zip(axes_row, panels, strict=True):
        inside = values[(labels > 0) & np.isfinite(values)]
        outside = values[(labels == 0) & np.isfinite(values)]
        axes.set_title(title)
        axes.set_xlabel("intensity")
        axes.set_ylabel("density")
        if inside.size + outside.size == 0:
            axes.text(0.5, 0.5, "no finite values", ha="center", transform=axes.transAxes)
            continue
        edges = np.histogram_bin_edges(np.concatenate([inside, outside]), bins=_HISTOGRAM_BINS)
        if outside.size:
            axes.hist(outside, bins=edges, density=True, alpha=0.5, label="background")
        if inside.size:
            axes.hist(inside, bins=edges, density=True, alpha=0.5, label="objects")
        axes.legend()
    return _png(figure)


def _area_png(areas: npt.NDArray[np.float64], report: QualityReport) -> bytes:
    figure = _figure(6, 4)
    axes = figure.add_subplot()
    axes.set_title(FIGURE_TITLES["area"])
    axes.set_xlabel(f"area ({report.area_unit})")
    axes.set_ylabel("objects")
    if areas.size:
        axes.hist(areas, bins=_AREA_BINS)
    else:
        axes.text(0.5, 0.5, "no objects", ha="center", transform=axes.transAxes)
    for limit, label in ((report.area_min_limit, "minimum"), (report.area_max_limit, "maximum")):
        if limit is not None:
            axes.axvline(limit, color="red", linestyle="--", label=f"configured {label}")
    if report.area_min_limit is not None or report.area_max_limit is not None:
        axes.legend()
    return _png(figure)
