import numpy as np
import pytest

from microflow.segmentation import score_segmentation


def test_identical_masks_score_perfectly() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    scores = score_segmentation(mask, mask)
    assert scores.dice == 1.0
    assert scores.iou == 1.0


def test_disjoint_masks_score_zero() -> None:
    prediction = np.zeros(8, dtype=bool)
    prediction[0:3] = True
    truth = np.zeros(8, dtype=bool)
    truth[5:8] = True
    scores = score_segmentation(prediction, truth)
    assert scores.dice == 0.0
    assert scores.iou == 0.0


def test_partial_overlap_matches_hand_computed_values() -> None:
    prediction = np.zeros(10, dtype=int)
    prediction[0:6] = 1
    truth = np.zeros(10, dtype=int)
    truth[3:9] = 2
    scores = score_segmentation(prediction, truth)
    assert scores.dice == pytest.approx(2 * 3 / 12)
    assert scores.iou == pytest.approx(3 / 9)
    assert scores.predicted_pixels == 6
    assert scores.truth_pixels == 6


def test_dice_and_iou_satisfy_their_identity() -> None:
    rng = np.random.default_rng(0)
    prediction = rng.random((32, 32)) > 0.5
    truth = rng.random((32, 32)) > 0.4
    scores = score_segmentation(prediction, truth)
    assert scores.dice == pytest.approx(2 * scores.iou / (1 + scores.iou))


def test_label_values_are_treated_as_foreground() -> None:
    labels = np.array([[0, 7, 7], [3, 0, 0]])
    mask = np.array([[0, 1, 1], [1, 0, 0]], dtype=bool)
    assert score_segmentation(labels, mask).dice == 1.0


def test_two_empty_masks_agree_perfectly() -> None:
    scores = score_segmentation(np.zeros((3, 3)), np.zeros((3, 3)))
    assert scores.dice == 1.0
    assert scores.iou == 1.0
    assert scores.truth_pixels == 0


def test_one_empty_mask_scores_zero() -> None:
    scores = score_segmentation(np.zeros((3, 3)), np.ones((3, 3)))
    assert scores.dice == 0.0
    assert scores.iou == 0.0


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        score_segmentation(np.zeros((3, 3)), np.zeros((3, 4)))
