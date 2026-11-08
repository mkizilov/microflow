"""Deterministic synthetic image and mask pairs, for CI and for trying the pipeline out.

Each sample is a donor with several fields of view. Every field is a different random arrangement
of discs with known truth, so no two images share content. The fields of one donor share a sample
name, which exercises the split-by-sample rule.
"""

from pathlib import Path

import numpy as np
import yaml

from microflow.image_io import write_image
from microflow.sample_data import Disc, make_phantom

MIN_SIZE = 24


def make_synthetic_dataset(
    directory: Path,
    *,
    samples: int = 6,
    images_per_sample: int = 2,
    size: int = 64,
    seed: int = 0,
) -> Path:
    """Write images, masks and a manifest under ``directory``. Returns the manifest path."""
    if samples < 3:
        raise ValueError("at least 3 samples are needed to form train, validation and test")
    if images_per_sample < 1:
        raise ValueError("images_per_sample must be at least 1")
    if size < MIN_SIZE:
        raise ValueError(f"size must be at least {MIN_SIZE} pixels")
    rng = np.random.default_rng(seed)
    images_dir = directory / "images"
    masks_dir = directory / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []
    for sample_index in range(samples):
        sample = f"donor-{sample_index:02d}"
        for field in range(images_per_sample):
            name = f"{sample}_field{field}"
            image, truth = make_phantom(
                _random_discs(rng, size),
                (size, size),
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            write_image(images_dir / f"{name}.tif", np.round(image).astype(np.uint16))
            write_image(masks_dir / f"{name}.tif", (truth > 0).astype(np.uint8))
            entries.append(
                {
                    "sample": sample,
                    "image": f"images/{name}.tif",
                    "mask": f"masks/{name}.tif",
                }
            )
    manifest = directory / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"samples": entries}, sort_keys=False), encoding="utf-8")
    return manifest


def _random_discs(rng: np.random.Generator, size: int) -> list[Disc]:
    discs: list[Disc] = []
    for _ in range(int(rng.integers(2, 5))):
        radius = float(rng.uniform(4.0, 9.0))
        centre = (
            float(rng.uniform(radius, size - radius)),
            float(rng.uniform(radius, size - radius)),
        )
        discs.append(Disc(centre=centre, radius=radius))
    return discs
