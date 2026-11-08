"""Seeded augmentation. Geometric changes touch an image and its mask together; intensity and noise
change the image only. Every draw comes from the generator passed in, so the same generator state
gives the same result."""

import numpy as np
from numpy.typing import NDArray

from microflow.ml.config import AugmentConfig


def random_crop(
    image: NDArray[np.float64], mask: NDArray[np.uint8], size: int, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    rows, cols = image.shape
    if size > rows or size > cols:
        raise ValueError(f"crop size {size} is larger than the image shape {image.shape}")
    top = int(rng.integers(0, rows - size + 1))
    left = int(rng.integers(0, cols - size + 1))
    window = np.s_[top : top + size, left : left + size]
    return image[window], mask[window]


def augment_pair(
    image: NDArray[np.float64],
    mask: NDArray[np.uint8],
    rng: np.random.Generator,
    options: AugmentConfig,
) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    if options.rotate90:
        turns = int(rng.integers(0, 4))
        image = np.rot90(image, turns)
        mask = np.rot90(mask, turns)
    if options.flip and rng.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    if options.flip and rng.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if options.intensity > 0:
        scale = 1.0 + rng.uniform(-options.intensity, options.intensity)
        offset = rng.uniform(-options.intensity, options.intensity)
        image = image * scale + offset
    if options.noise_std > 0:
        image = image + rng.normal(0.0, options.noise_std, size=image.shape)
    return np.array(image, dtype=np.float64), np.array(mask, dtype=np.uint8)
