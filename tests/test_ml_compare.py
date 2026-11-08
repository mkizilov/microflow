import numpy as np
import pytest

from microflow.ml.compare import compare_label_stacks


def _labels(*objects: tuple[int, int, int, int, int]) -> np.ndarray:
    """Build a 10 x 10 label image. Each object is (label, top, bottom, left, right)."""
    labels = np.zeros((10, 10), dtype=np.int32)
    for label, top, bottom, left, right in objects:
        labels[top:bottom, left:right] = label
    return labels


def test_identical_labels_agree_perfectly() -> None:
    labels = _labels((1, 0, 2, 0, 2), (2, 5, 8, 5, 8))
    result = compare_label_stacks(labels, labels, "YX")
    assert result.dice == 1.0
    assert result.iou == 1.0
    assert result.objects_a == result.objects_b == result.matched_objects == 2


def test_disjoint_labels_share_nothing() -> None:
    first = _labels((1, 0, 2, 0, 2))
    second = _labels((1, 6, 8, 6, 8))
    result = compare_label_stacks(first, second, "YX")
    assert result.dice == 0.0
    assert result.matched_objects == 0


def test_partial_overlap_is_scored_and_matched_by_iou() -> None:
    # Two 2 x 2 objects sharing a column: intersection 2, union 6, IoU 1/3, Dice 1/2.
    first = _labels((1, 0, 2, 0, 2))
    second = _labels((1, 0, 2, 1, 3))
    strict = compare_label_stacks(first, second, "YX", match_iou=0.5)
    assert strict.dice == pytest.approx(0.5)
    assert strict.iou == pytest.approx(2 / 6)
    assert strict.matched_objects == 0
    lenient = compare_label_stacks(first, second, "YX", match_iou=0.3)
    assert lenient.matched_objects == 1


def test_matching_is_one_to_one() -> None:
    reference = _labels((1, 0, 4, 0, 4))
    ml = _labels((1, 0, 2, 0, 4), (2, 2, 4, 0, 4))
    result = compare_label_stacks(ml, reference, "YX", match_iou=0.3)
    assert result.objects_a == 2
    assert result.matched_objects == 1


def test_stacks_are_compared_plane_by_plane_so_restarted_labels_do_not_collide() -> None:
    plane_a = _labels((1, 0, 2, 0, 2))
    plane_b = _labels((1, 5, 7, 5, 7))
    ml = np.stack([plane_a, plane_b])
    reference = np.stack([plane_b, plane_a])
    result = compare_label_stacks(ml, reference, "ZYX")
    assert result.planes == 2
    assert result.matched_objects == 0
    matched = compare_label_stacks(ml, ml, "ZYX")
    assert matched.matched_objects == 2


def test_comparison_is_deterministic() -> None:
    rng = np.random.default_rng(9)
    first = rng.integers(0, 4, size=(12, 12)).astype(np.int32)
    second = rng.integers(0, 4, size=(12, 12)).astype(np.int32)
    assert compare_label_stacks(first, second, "YX") == compare_label_stacks(first, second, "YX")


def test_shapes_and_spatial_axes_are_checked() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        compare_label_stacks(np.zeros((4, 4)), np.zeros((4, 5)), "YX")
    with pytest.raises(ValueError, match="need Y and X"):
        compare_label_stacks(np.zeros((4, 4)), np.zeros((4, 4)), "ZC")
    with pytest.raises(ValueError, match="match_iou"):
        compare_label_stacks(np.zeros((4, 4)), np.zeros((4, 4)), "YX", match_iou=0.0)
