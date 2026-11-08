import numpy as np
import pytest
from scipy import ndimage

from microflow.config import SegmentationConfig
from microflow.sample_data import Disc, make_phantom
from microflow.segmentation import algorithms as alg
from microflow.segmentation import score_segmentation, segment_image, segment_plane
from microflow.units import Area, Calibration, Length

PIXEL = Calibration(pixel_size=1.0, unit="um")

# Discs are (centre row, centre column) in pixels. Discs 4 and 5 overlap, so they form one
# foreground object until the watershed splits them. Disc 7 is 5 pixels across the middle.
DISCS = [
    Disc(centre=(30, 30), radius=8),
    Disc(centre=(30, 95), radius=6),
    Disc(centre=(90, 40), radius=10),
    Disc(centre=(100, 100), radius=7),
    Disc(centre=(100, 112), radius=7),
    Disc(centre=(8, 64), radius=8),
    Disc(centre=(60, 15), radius=1.2),
]
DOT = 7
SPLIT = {"watershed": True, "watershed_min_distance": Length(value=3, unit="um")}


def _config(**overrides: object) -> SegmentationConfig:
    values: dict[str, object] = {"min_object_area": Area(value=9.0, unit="um"), **overrides}
    return SegmentationConfig.model_validate(values)


def _label_of(labels: np.ndarray, truth: np.ndarray, number: int) -> int:
    values = np.unique(labels[truth == number])
    assert values.size == 1 and values[0] != 0
    return int(values[0])


def _assert_raster_ordered(labels: np.ndarray) -> None:
    first = [int(np.flatnonzero(labels == value)[0]) for value in range(1, labels.max() + 1)]
    assert first == sorted(first)


@pytest.fixture(scope="module")
def phantom() -> tuple[np.ndarray, np.ndarray]:
    image, truth = make_phantom(DISCS, seed=1)
    assert (truth == DOT).sum() == 5
    return image, truth


def test_otsu_recovers_isolated_discs_exactly(phantom: tuple[np.ndarray, np.ndarray]) -> None:
    image, truth = phantom
    labels, level = segment_plane(image, _config(), PIXEL)
    assert level is not None and 100 < level < 400
    assert labels.max() == 5
    for number in (1, 2, 3, 6):
        label = _label_of(labels, truth, number)
        assert np.array_equal(labels == label, truth == number)


def test_without_watershed_overlapping_discs_form_one_object(
    phantom: tuple[np.ndarray, np.ndarray],
) -> None:
    labels, _ = segment_plane(phantom[0], _config(), PIXEL)
    assert labels[100, 100] == labels[100, 112] != 0


def test_watershed_splits_overlapping_discs(phantom: tuple[np.ndarray, np.ndarray]) -> None:
    labels, _ = segment_plane(phantom[0], _config(**SPLIT), PIXEL)
    assert labels.max() == 6
    left, right = labels[100, 100], labels[100, 112]
    assert left != 0 and right != 0 and left != right
    for label, centre in ((left, (100, 100)), (right, (100, 112))):
        centroid = ndimage.center_of_mass(labels == label)
        assert np.hypot(centroid[0] - centre[0], centroid[1] - centre[1]) < 2.5


def test_exclude_border_removes_only_objects_touching_the_edge(
    phantom: tuple[np.ndarray, np.ndarray],
) -> None:
    image, truth = phantom
    labels, _ = segment_plane(image, _config(**SPLIT, exclude_border=True), PIXEL)
    assert labels.max() == 5
    assert not labels[0].any()
    assert not labels[truth == 6].any()


def test_exclude_border_without_watershed(phantom: tuple[np.ndarray, np.ndarray]) -> None:
    labels, _ = segment_plane(phantom[0], _config(exclude_border=True), PIXEL)
    assert labels.max() == 4


@pytest.mark.parametrize(("minimum", "dot_kept"), [(5, True), (6, False)])
def test_minimum_object_area_is_inclusive(
    phantom: tuple[np.ndarray, np.ndarray], minimum: float, dot_kept: bool
) -> None:
    labels, _ = segment_plane(
        phantom[0], _config(min_object_area=Area(value=minimum, unit="um")), PIXEL
    )
    assert (labels[60, 15] != 0) is dot_kept


def test_segmentation_scores_against_ground_truth(phantom: tuple[np.ndarray, np.ndarray]) -> None:
    image, truth = phantom
    labels, _ = segment_plane(image, _config(**SPLIT), PIXEL)
    scores = score_segmentation(labels > 0, truth > 0)
    assert scores.dice > 0.99
    assert scores.iou > 0.98
    # The only difference from the truth is the 5-pixel dot, which the minimum area removes.
    assert scores.truth_pixels - scores.predicted_pixels == 5


def test_manual_threshold_matches_otsu_on_clean_phantom(
    phantom: tuple[np.ndarray, np.ndarray],
) -> None:
    image, _ = phantom
    labels, level = segment_plane(image, _config(method="threshold", threshold=250.0), PIXEL)
    assert level == 250.0
    otsu_labels, _ = segment_plane(image, _config(), PIXEL)
    np.testing.assert_array_equal(labels, otsu_labels)


def test_adaptive_threshold_handles_illumination_gradient_where_otsu_fails() -> None:
    ramp = np.tile(np.linspace(0.0, 800.0, 128), (128, 1))
    image, truth = make_phantom(DISCS[:3], background=ramp, seed=2)
    adaptive = _config(
        method="adaptive",
        adaptive_block_size=Length(value=31, unit="um"),
        adaptive_offset=50.0,
    )
    adaptive_labels, adaptive_level = segment_plane(image, adaptive, PIXEL)
    otsu_labels, _ = segment_plane(image, _config(), PIXEL)
    assert adaptive_level is None
    assert score_segmentation(adaptive_labels > 0, truth > 0).dice > 0.9
    assert score_segmentation(otsu_labels > 0, truth > 0).dice < 0.6


def test_invalid_pixels_are_background_and_do_not_disturb_thresholds(
    phantom: tuple[np.ndarray, np.ndarray],
) -> None:
    image, _ = phantom
    clean, _ = segment_plane(image, _config(), PIXEL)
    damaged = image.copy()
    damaged[0:10, :] = np.nan
    labels, level = segment_plane(damaged, _config(), PIXEL)
    assert level is not None and np.isfinite(level)
    assert not labels[0:10].any()
    np.testing.assert_array_equal(labels[10:] > 0, clean[10:] > 0)


def test_constant_image_has_no_objects() -> None:
    labels, _ = segment_plane(np.full((16, 16), 7.0), _config(), PIXEL)
    assert labels.max() == 0


def test_segmentation_is_deterministic(phantom: tuple[np.ndarray, np.ndarray]) -> None:
    image, _ = phantom
    first, _ = segment_plane(image, _config(**SPLIT), PIXEL)
    second, _ = segment_plane(image, _config(**SPLIT), PIXEL)
    np.testing.assert_array_equal(first, second)


def test_label_image_type_and_numbering(phantom: tuple[np.ndarray, np.ndarray]) -> None:
    labels, _ = segment_plane(phantom[0], _config(**SPLIT), PIXEL)
    assert labels.dtype == np.int32
    assert np.unique(labels).tolist() == list(range(labels.max() + 1))
    _assert_raster_ordered(labels)


def test_length_settings_need_calibration() -> None:
    with pytest.raises(ValueError, match="needs pixel calibration"):
        segment_plane(np.zeros((8, 8)), SegmentationConfig(), None)


def test_segment_image_labels_each_plane_in_axis_order(
    phantom: tuple[np.ndarray, np.ndarray],
) -> None:
    image, _ = phantom
    other, _ = make_phantom(DISCS[:2], seed=9)
    stack = np.stack([image, other], axis=-1)
    result = segment_image(stack, "YXC", _config(), PIXEL)
    assert result.axes == "YXC"
    assert result.labels.shape == (128, 128, 2)
    assert result.labels.dtype == np.int32
    assert result.object_counts == (5, 2)
    assert len(result.thresholds) == 2
    for channel, plane in enumerate((image, other)):
        expected, _ = segment_plane(plane, _config(), PIXEL)
        np.testing.assert_array_equal(result.labels[..., channel], expected)


def test_opening_removes_isolated_pixels_and_keeps_objects() -> None:
    rows, cols = np.indices((40, 40))
    disc = (rows - 20) ** 2 + (cols - 20) ** 2 <= 36
    mask = disc.copy()
    salt = [(3, 3), (3, 35), (35, 3)]
    for point in salt:
        mask[point] = True
    opened = alg.open_mask(mask, 1)
    assert not any(opened[point] for point in salt)
    assert opened[20, 20]
    assert opened.sum() >= 0.9 * disc.sum()


def test_closing_fills_single_pixel_holes() -> None:
    rows, cols = np.indices((40, 40))
    mask = (rows - 20) ** 2 + (cols - 20) ** 2 <= 36
    mask[20, 20] = False
    assert alg.close_mask(mask, 1)[20, 20]


def test_opening_does_not_erode_objects_touching_the_image_edge() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[0:10, 0:10] = True
    opened = alg.open_mask(mask, 2)
    assert opened[3:7, 0:10].all()
    assert opened[0, 5]


def test_label_objects_uses_eight_connectivity() -> None:
    diagonal = np.zeros((6, 6), dtype=bool)
    diagonal[0:2, 0:2] = True
    diagonal[2:4, 2:4] = True
    assert alg.label_objects(diagonal).max() == 1
    separated = diagonal.copy()
    separated[2, 2] = False
    separated[3, 3] = False
    assert alg.label_objects(separated).max() == 2


def test_relabel_in_raster_order_numbers_objects_by_first_pixel() -> None:
    labels = np.array([[3, 0, 3], [0, 1, 0]], dtype=np.int32)
    np.testing.assert_array_equal(alg.relabel_in_raster_order(labels), [[1, 0, 1], [0, 2, 0]])


def test_remove_small_keeps_objects_at_the_minimum() -> None:
    labels = np.zeros((3, 4), dtype=np.int32)
    labels[0, 0:4] = 1
    labels[2, 0:2] = 2
    cleaned = alg.remove_small(labels, 4)
    assert (cleaned == 1).sum() == 4
    assert not (cleaned == 2).any()


def test_remove_border_drops_only_objects_on_the_edge() -> None:
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[0, 2] = 1
    labels[2, 2] = 2
    labels[4, 4] = 3
    cleaned = alg.remove_border(labels)
    assert cleaned[0, 2] == 0 and cleaned[4, 4] == 0 and cleaned[2, 2] == 2


def test_phantom_overlap_is_assigned_to_nearest_centre() -> None:
    _, truth = make_phantom([Disc((10, 10), 6), Disc((10, 16), 6)], shape=(32, 32), seed=0)
    assert truth[10, 8] == 1
    assert truth[10, 18] == 2
    assert truth[10, 13] == 1  # 3 px from centre 1, 3 px from centre 2: first disc wins the tie
