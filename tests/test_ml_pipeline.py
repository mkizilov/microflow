from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import ndimage

from microflow.ml.pipeline import PlaneResult, segment_plane, segment_stack
from microflow.ml.postprocess import labels_from_probabilities
from microflow.ml.preparation import Normalization
from microflow.ml.quality import calibration_flag
from microflow.ml.tiling import pad_to_multiple, receptive_radius

IDENTITY = Normalization(mean=0.0, std=1.0)
SIGMA = 2.0


def _operator(window: np.ndarray) -> np.ndarray:
    return np.asarray(
        ndimage.gaussian_filter(window, sigma=SIGMA, mode="constant", truncate=4.0),
        dtype=np.float32,
    )


def _run(
    plane: NDArray[np.float64],
    predict: Callable[[NDArray[np.float64]], NDArray[np.float32]] = _operator,
    **overrides: Any,
) -> PlaneResult:
    values: dict[str, Any] = {
        "normalization": IDENTITY,
        "predict": predict,
        "multiple": 4,
        "tile_size": 40,
        "overlap": 8,
        "receptive_radius": 8,
        "threshold": 0.5,
    }
    values.update(overrides)
    return segment_plane(plane, **values)


def test_labels_are_background_zero_int32_numbered_in_raster_order() -> None:
    probabilities = np.zeros((8, 8))
    probabilities[5, 5] = 0.9
    probabilities[2:4, 2:4] = 0.9
    labels = labels_from_probabilities(probabilities, threshold=0.5)
    assert labels.dtype == np.int32
    assert labels[5, 5] == 2
    assert labels[2, 2] == 1
    assert (labels > 0).sum() == 5


def test_threshold_is_inclusive_and_shared_with_scoring() -> None:
    probabilities = np.array([[0.49, 0.5], [0.51, 0.0]])
    labels = labels_from_probabilities(probabilities, threshold=0.5)
    np.testing.assert_array_equal(labels > 0, [[False, True], [True, False]])


def test_minimum_area_and_border_rules_follow_the_classical_contract() -> None:
    probabilities = np.zeros((8, 8))
    probabilities[0, 0] = 0.9
    probabilities[2:4, 2:4] = 0.9
    probabilities[6, 6] = 0.9
    small_removed = labels_from_probabilities(probabilities, threshold=0.5, min_area_px=2)
    assert (small_removed > 0).sum() == 4
    border_removed = labels_from_probabilities(
        probabilities, threshold=0.5, min_area_px=1, exclude_border=True
    )
    assert border_removed[0, 0] == 0
    assert border_removed.max() == 2


def test_uniform_probabilities_below_threshold_give_an_empty_flagged_plane() -> None:
    result = _run(np.zeros((12, 12)), predict=lambda window: np.zeros(window.shape, np.float32))
    assert result.quality.object_count == 0
    assert "empty_prediction" in result.quality.flags
    assert result.quality.mean_object_probability is None


def test_output_dimensions_follow_the_input_even_when_padding_was_needed() -> None:
    plane = np.random.default_rng(3).uniform(0, 1, size=(13, 7))
    result = _run(plane, predict=lambda window: window.astype(np.float32))
    assert result.probabilities.shape == (13, 7)
    assert result.labels.shape == (13, 7)
    assert result.labels.dtype == np.int32
    np.testing.assert_allclose(result.probabilities, plane.astype(np.float32))


def test_segmenting_the_whole_plane_equals_segmenting_its_tiles_when_overlap_suffices() -> None:
    plane = np.random.default_rng(0).normal(size=(100, 90))
    result = _run(plane)
    padded = pad_to_multiple(IDENTITY.apply(plane), 4)
    whole = _operator(padded)[:100, :90]
    np.testing.assert_allclose(result.probabilities, whole, rtol=0, atol=1e-6)
    assert len(result.plan.tiles) > 1
    assert "overlap_below_receptive_field" not in result.quality.flags


def test_overlap_below_the_receptive_radius_is_flagged_for_multi_tile_planes() -> None:
    plane = np.random.default_rng(0).normal(size=(100, 90))
    result = _run(plane, overlap=0, receptive_radius=receptive_radius(2))
    assert "overlap_below_receptive_field" in result.quality.flags


def test_a_single_window_plane_has_no_overlap_warning() -> None:
    result = _run(np.ones((20, 20)), overlap=0, receptive_radius=26)
    assert len(result.plan.tiles) == 1
    assert "overlap_below_receptive_field" not in result.quality.flags


def test_excessive_coverage_and_uncertainty_are_flagged() -> None:
    everything = _run(
        np.zeros((12, 12)), predict=lambda window: np.full(window.shape, 0.9, np.float32)
    )
    assert "excessive_coverage" in everything.quality.flags
    assert everything.quality.object_count == 1
    unsure = _run(np.zeros((12, 12)), predict=lambda window: np.full(window.shape, 0.5, np.float32))
    assert "high_uncertainty" in unsure.quality.flags
    assert unsure.quality.uncertain_fraction == pytest.approx(1.0)


def test_border_objects_are_counted() -> None:
    # The plane is padded to 12 x 12, and the predictor sees the padded window.
    probabilities = np.zeros((12, 12), np.float32)
    probabilities[0:2, 0:2] = 0.9
    probabilities[5:7, 5:7] = 0.9
    result = _run(np.zeros((10, 10)), predict=lambda window: probabilities)
    assert result.quality.object_count == 2
    assert result.quality.border_object_count == 1


def test_a_plane_with_non_finite_values_is_refused() -> None:
    plane = np.ones((8, 8))
    plane[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _run(plane)


def test_calibration_is_compared_only_when_the_model_recorded_one() -> None:
    assert calibration_flag(None, ()) is None
    assert calibration_flag(None, ((0.5, "um"),)) == "calibration_missing"
    assert calibration_flag((0.5, "um"), ((0.5, "um"),)) is None
    assert calibration_flag((0.25, "um"), ((0.5, "um"),)) == "calibration_mismatch"
    assert calibration_flag((0.5, "nm"), ((0.5, "um"),)) == "calibration_mismatch"


def test_a_device_fallback_is_reported_on_the_plane() -> None:
    result = _run(np.ones((12, 12)), fell_back=lambda: True)
    assert "device_fallback" in result.quality.flags


def test_stacks_keep_their_axes_and_segment_each_plane_independently() -> None:
    rng = np.random.default_rng(5)
    data = rng.uniform(0, 1, size=(20, 2, 16))
    options: dict[str, Any] = {
        "predict": lambda window: window.astype(np.float32),
        "tile_size": 16,
        "overlap": 2,
        "multiple": 4,
        "receptive_radius": receptive_radius(2),
    }
    labels, results = segment_stack(data, "YZX", lambda plane: _run(plane, **options))
    assert labels.shape == data.shape
    assert len(results) == 2
    for index in range(2):
        alone = _run(data[:, index, :], **options)
        np.testing.assert_array_equal(labels[:, index, :], alone.labels)


def test_stacks_without_spatial_axes_are_refused() -> None:
    with pytest.raises(ValueError, match="needs Y and X axes"):
        segment_stack(np.zeros((2, 3)), "ZC", lambda plane: _run(plane))
