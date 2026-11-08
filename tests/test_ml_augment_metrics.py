import numpy as np
import pytest

from microflow.ml.augment import augment_pair, random_crop
from microflow.ml.config import AugmentConfig
from microflow.ml.metrics import binarize, score_probabilities


def _aligned_pair() -> tuple[np.ndarray, np.ndarray]:
    """An image whose pixels equal its mask, so alignment is checkable after any transform."""
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:6, 9:14] = 1
    mask[12:14, 1:3] = 1
    return mask.astype(np.float64), mask


def test_geometric_augmentation_keeps_image_and_mask_aligned() -> None:
    image, mask = _aligned_pair()
    options = AugmentConfig(flip=True, rotate90=True, intensity=0.0, noise_std=0.0)
    for seed in range(25):
        augmented_image, augmented_mask = augment_pair(
            image.copy(), mask.copy(), np.random.default_rng(seed), options
        )
        np.testing.assert_array_equal(augmented_image, augmented_mask.astype(np.float64))
        assert augmented_mask.sum() == mask.sum()


def test_intensity_and_noise_change_only_the_image() -> None:
    image, mask = _aligned_pair()
    options = AugmentConfig(flip=False, rotate90=False, intensity=0.2, noise_std=0.05)
    augmented_image, augmented_mask = augment_pair(
        image.copy(), mask.copy(), np.random.default_rng(4), options
    )
    np.testing.assert_array_equal(augmented_mask, mask)
    assert not np.allclose(augmented_image, image)


def test_augmentation_is_reproducible_for_a_generator_state() -> None:
    image, mask = _aligned_pair()
    options = AugmentConfig(intensity=0.1, noise_std=0.02)
    first = augment_pair(image.copy(), mask.copy(), np.random.default_rng([7, 1, 2]), options)
    second = augment_pair(image.copy(), mask.copy(), np.random.default_rng([7, 1, 2]), options)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_disabled_augmentation_returns_the_input_values() -> None:
    image, mask = _aligned_pair()
    options = AugmentConfig(flip=False, rotate90=False, intensity=0.0, noise_std=0.0)
    augmented_image, augmented_mask = augment_pair(
        image, mask, np.random.default_rng(0), options
    )
    np.testing.assert_array_equal(augmented_image, image)
    np.testing.assert_array_equal(augmented_mask, mask)


def test_random_crop_has_the_requested_size_and_stays_aligned() -> None:
    image, mask = _aligned_pair()
    for seed in range(10):
        crop_image, crop_mask = random_crop(image, mask, 8, np.random.default_rng(seed))
        assert crop_image.shape == crop_mask.shape == (8, 8)
        np.testing.assert_array_equal(crop_image, crop_mask.astype(np.float64))


def test_crop_larger_than_the_image_is_refused() -> None:
    image, mask = _aligned_pair()
    with pytest.raises(ValueError, match="larger than the image"):
        random_crop(image, mask, 17, np.random.default_rng(0))


def test_binarize_treats_the_threshold_as_inclusive() -> None:
    probabilities = np.array([0.49, 0.5, 0.51])
    np.testing.assert_array_equal(binarize(probabilities, 0.5), [False, True, True])
    with pytest.raises(ValueError, match="between 0 and 1"):
        binarize(probabilities, 1.0)


def test_dice_and_iou_match_their_definitions() -> None:
    # Predicted foreground: two pixels. True foreground: one pixel, which is predicted.
    # Dice = 2 * 1 / (2 + 1) = 2/3, IoU = 1 / (2 + 1 - 1) = 1/2.
    probabilities = np.array([[0.9, 0.2], [0.6, 0.4]])
    truth = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    scores = score_probabilities(probabilities, truth, 0.5)
    assert scores.dice == pytest.approx(2 / 3)
    assert scores.iou == pytest.approx(1 / 2)
    assert scores.predicted_pixels == 2
    assert scores.truth_pixels == 1


def test_two_empty_masks_agree_perfectly() -> None:
    scores = score_probabilities(np.zeros((4, 4)), np.zeros((4, 4), dtype=np.uint8), 0.5)
    assert scores.dice == 1.0
    assert scores.iou == 1.0
