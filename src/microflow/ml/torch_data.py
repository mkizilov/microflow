"""Training patches: random crops with seeded augmentation, drawn only from the images of one split.

Each (epoch, index) pair seeds its own generator, so a patch depends only on the seed, the epoch
and its index, not on how many workers load the data or in what order.
"""

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from microflow.ml.augment import augment_pair, random_crop
from microflow.ml.config import AugmentConfig
from microflow.ml.preparation import ImagePair, Normalization


class PatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        pairs: Sequence[ImagePair],
        *,
        normalization: Normalization,
        size: int,
        per_image: int,
        augment: AugmentConfig,
        seed: int,
    ) -> None:
        if not pairs:
            raise ValueError("there are no training images to draw patches from")
        for pair in pairs:
            if min(pair.image.shape) < size:
                raise ValueError(
                    f"{pair.image_path}: image shape {pair.image.shape} is smaller than the "
                    f"patch size {size}"
                )
        self._images = [normalization.apply(pair.image) for pair in pairs]
        self._masks = [pair.mask for pair in pairs]
        self._size = size
        self._per_image = per_image
        self._augment = augment
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._images) * self._per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source = index % len(self._images)
        rng = np.random.default_rng([self._seed, self._epoch, index])
        image, mask = random_crop(self._images[source], self._masks[source], self._size, rng)
        image, mask = augment_pair(image, mask, rng, self._augment)
        return (
            torch.as_tensor(image, dtype=torch.float32)[None],
            torch.as_tensor(mask, dtype=torch.float32)[None],
        )
