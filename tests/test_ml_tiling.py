import numpy as np
import pytest
from scipy import ndimage

from microflow.ml.tiling import pad_to_multiple, plan_tiles, predict_tiled, receptive_radius

SIGMA = 2.0
SUPPORT = int(4.0 * SIGMA + 0.5)


def _local_operator(window: np.ndarray) -> np.ndarray:
    """A Gaussian filter whose output at each pixel depends only on pixels within SUPPORT. Zero
    padding at the edge of whatever it is given, so a window agrees with the plane wherever its
    context lies inside the window."""
    filtered = ndimage.gaussian_filter(window, sigma=SIGMA, mode="constant", truncate=4.0)
    return np.asarray(filtered, dtype=np.float32)


def test_receptive_radius_follows_the_stated_bound() -> None:
    assert receptive_radius(1) == 10
    assert receptive_radius(2) == 26
    assert receptive_radius(3) == 58
    with pytest.raises(ValueError, match="at least 1"):
        receptive_radius(0)


def test_padding_adds_zeros_at_the_bottom_and_right_only() -> None:
    image = np.arange(6, dtype=np.float64).reshape(2, 3)
    padded = pad_to_multiple(image, 4)
    assert padded.shape == (4, 4)
    np.testing.assert_array_equal(padded[:2, :3], image)
    assert not padded[2:].any()
    assert not padded[:, 3:].any()
    assert pad_to_multiple(np.zeros((8, 4)), 4).shape == (8, 4)


@pytest.mark.parametrize(
    ("shape", "tile", "overlap"),
    [
        ((100, 90), 40, 8),
        ((37, 23), 16, 3),
        ((12, 9), 16, 2),
        ((64, 64), 32, 10),
        ((33, 70), 24, 0),
    ],
)
def test_every_pixel_is_a_core_pixel_of_exactly_one_window(
    shape: tuple[int, int], tile: int, overlap: int
) -> None:
    plan = plan_tiles(shape, tile_size=tile, overlap=overlap)
    coverage = np.zeros(shape, dtype=int)
    for item in plan.tiles:
        top, bottom, left, right = item.window
        assert 0 <= top < bottom <= shape[0]
        assert 0 <= left < right <= shape[1]
        assert bottom - top <= tile and right - left <= tile
        core_top, core_bottom, core_left, core_right = item.core
        assert top <= core_top < core_bottom <= bottom
        assert left <= core_left < core_right <= right
        coverage[core_top:core_bottom, core_left:core_right] += 1
    assert (coverage == 1).all()


@pytest.mark.parametrize(("shape", "tile", "overlap"), [((100, 90), 40, 8), ((37, 23), 16, 3)])
def test_each_core_keeps_the_overlap_of_context_unless_it_meets_the_plane_edge(
    shape: tuple[int, int], tile: int, overlap: int
) -> None:
    plan = plan_tiles(shape, tile_size=tile, overlap=overlap)
    for item in plan.tiles:
        top, bottom, left, right = item.window
        core_top, core_bottom, core_left, core_right = item.core
        assert top == 0 or core_top - top >= overlap
        assert bottom == shape[0] or bottom - core_bottom >= overlap
        assert left == 0 or core_left - left >= overlap
        assert right == shape[1] or right - core_right >= overlap


def test_tiled_prediction_equals_whole_plane_prediction_when_overlap_covers_the_support() -> None:
    plane = np.random.default_rng(0).normal(size=(100, 90))
    whole = _local_operator(plane)
    plan = plan_tiles(plane.shape, tile_size=40, overlap=SUPPORT)
    tiled = predict_tiled(plane, _local_operator, plan)
    np.testing.assert_allclose(tiled, whole, rtol=0, atol=1e-6)


def test_too_little_overlap_visibly_changes_the_result_near_tile_seams() -> None:
    plane = np.random.default_rng(0).normal(size=(100, 90))
    whole = _local_operator(plane)
    plan = plan_tiles(plane.shape, tile_size=40, overlap=0)
    tiled = predict_tiled(plane, _local_operator, plan)
    assert np.abs(tiled - whole).max() > 1e-3


def test_a_plane_smaller_than_the_tile_is_predicted_in_one_window() -> None:
    plane = np.random.default_rng(1).normal(size=(20, 30))
    plan = plan_tiles(plane.shape, tile_size=40, overlap=8)
    assert len(plan.tiles) == 1
    tiled = predict_tiled(plane, _local_operator, plan)
    np.testing.assert_array_equal(tiled, _local_operator(plane))


def test_tiling_is_deterministic() -> None:
    plane = np.random.default_rng(2).normal(size=(64, 80))
    plan = plan_tiles(plane.shape, tile_size=32, overlap=8)
    first = predict_tiled(plane, _local_operator, plan)
    second = predict_tiled(plane, _local_operator, plan)
    np.testing.assert_array_equal(first, second)


def test_image_and_plan_must_have_the_same_shape() -> None:
    plan = plan_tiles((6, 6), tile_size=16, overlap=2)
    with pytest.raises(ValueError, match="does not match the plan"):
        predict_tiled(np.zeros((5, 5)), _local_operator, plan)


def test_a_predictor_returning_the_wrong_shape_is_refused() -> None:
    plan = plan_tiles((6, 6), tile_size=16, overlap=2)
    with pytest.raises(ValueError, match="returned shape"):
        predict_tiled(np.zeros((6, 6)), lambda window: np.zeros((2, 2), np.float32), plan)


@pytest.mark.parametrize(
    ("tile", "overlap", "multiple", "message"),
    [
        (10, 2, 4, "positive multiple of 4"),
        (16, -1, 1, "cannot be negative"),
        (16, 8, 1, "less than half"),
    ],
)
def test_invalid_tiling_parameters_are_refused(
    tile: int, overlap: int, multiple: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_tiles((32, 32), tile_size=tile, overlap=overlap, multiple=multiple)
