"""Plane-level ML segmentation, independent of PyTorch.

``segment_plane`` takes a window predictor that maps a normalised window to probabilities. The
PyTorch predictor is in ``microflow.ml.predict``. Keeping the sequence here (validate, normalise,
pad, tile, threshold, postprocess, report) means it is tested without PyTorch.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from microflow.ml.fallback import Predict
from microflow.ml.postprocess import labels_from_probabilities
from microflow.ml.preparation import Normalization
from microflow.ml.quality import PlaneQuality, assess_plane, calibration_flag
from microflow.ml.tiling import TilePlan, pad_to_multiple, plan_tiles, predict_tiled


@dataclass(frozen=True)
class PlaneResult:
    probabilities: NDArray[np.float32]
    labels: NDArray[np.int32]
    quality: PlaneQuality
    plan: TilePlan
    overlap: int
    receptive_radius: int


def _no_fallback() -> bool:
    return False


def segment_plane(
    plane: NDArray[np.float64],
    *,
    normalization: Normalization,
    predict: Predict,
    multiple: int,
    tile_size: int,
    overlap: int,
    receptive_radius: int,
    threshold: float,
    min_area_px: float = 0.0,
    exclude_border: bool = False,
    calibration: tuple[float, str] | None = None,
    training_calibrations: Sequence[tuple[float, str]] = (),
    fell_back: Callable[[], bool] = _no_fallback,
) -> PlaneResult:
    if plane.ndim != 2:
        raise ValueError(f"expected one Y-X plane; got shape {plane.shape}")
    if not np.isfinite(plane).all():
        raise ValueError(
            "the plane contains non-finite values; fill or mask them before ML segmentation"
        )
    rows, cols = plane.shape
    padded = pad_to_multiple(normalization.apply(np.asarray(plane, dtype=np.float64)), multiple)
    plan = plan_tiles(padded.shape, tile_size=tile_size, overlap=overlap, multiple=multiple)
    probabilities = predict_tiled(padded, predict, plan)[:rows, :cols].copy()
    labels = labels_from_probabilities(
        probabilities,
        threshold=threshold,
        min_area_px=min_area_px,
        exclude_border=exclude_border,
    )

    extra: list[str] = []
    if len(plan.tiles) > 1 and overlap < receptive_radius:
        extra.append("overlap_below_receptive_field")
    issue = calibration_flag(calibration, tuple(training_calibrations))
    if issue is not None:
        extra.append(issue)
    if fell_back():
        extra.append("device_fallback")
    quality = assess_plane(probabilities, labels, threshold=threshold, extra_flags=extra)
    return PlaneResult(
        probabilities=probabilities,
        labels=labels,
        quality=quality,
        plan=plan,
        overlap=overlap,
        receptive_radius=receptive_radius,
    )


def segment_stack(
    data: ArrayLike,
    axes: str,
    segment: Callable[[NDArray[np.float64]], PlaneResult],
) -> tuple[NDArray[np.int32], tuple[PlaneResult, ...]]:
    """Segment every Y-X plane of ``data``, whatever the other axes are. Returns labels with the
    input's axes, and one result per plane in the order of the other axes."""
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"the image needs Y and X axes; got {axes!r}")
    array = np.asarray(data, dtype=np.float64)
    spatial = (axes.index("Y"), axes.index("X"))
    planes = np.moveaxis(array, spatial, (-2, -1))
    labels = np.zeros(planes.shape, dtype=np.int32)
    results: list[PlaneResult] = []
    for index in np.ndindex(planes.shape[:-2]):
        result = segment(planes[index])
        labels[index] = result.labels
        results.append(result)
    return np.moveaxis(labels, (-2, -1), spatial), tuple(results)
