"""Splitting by original sample, and the leakage checks that go with it.

Every image belongs to one original sample and every sample to exactly one split. Patches are cut
only from images of their own split, so no patch can leak into another split through its source
image. Identical image content under two different samples is refused, because the two samples
would have to be in different splits and the same pixels would then be seen twice.

Regions of one acquisition that are cut into tiles must be listed under one sample. Overlap
between differently named files cannot be detected from their content, and is the user's
responsibility.
"""

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microflow.ml.manifest import SampleImage
from microflow.ml.runtime import LeakageError

SPLIT_NAMES = ("train", "validation", "test")
MIN_SAMPLES = 3


@dataclass(frozen=True)
class SplitPlan:
    seed: int
    assignment: tuple[tuple[str, str], ...]

    def members(self, split: str) -> tuple[str, ...]:
        return tuple(sample for sample, name in self.assignment if name == split)

    def split_of(self, sample: str) -> str:
        return dict(self.assignment)[sample]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "splits": {name: list(self.members(name)) for name in SPLIT_NAMES},
        }


def plan_split(
    sample_ids: Iterable[str], *, validation: float, test: float, seed: int
) -> SplitPlan:
    """Assign whole samples to splits. Each held-out split gets at least one sample, and training
    gets the rest. The same seed and sample names always give the same plan."""
    samples = sorted(set(sample_ids))
    if len(samples) < MIN_SAMPLES:
        raise ValueError(
            f"at least {MIN_SAMPLES} samples are needed to form three splits; got {len(samples)}"
        )
    order = list(samples)
    random.Random(seed).shuffle(order)
    total = len(order)
    n_test = max(1, round(total * test))
    n_validation = max(1, round(total * validation))
    n_train = total - n_test - n_validation
    if n_train < 1:
        raise ValueError(
            "the split fractions leave no samples for training; add samples or lower the "
            "validation or test fraction"
        )
    names = ["train"] * n_train + ["validation"] * n_validation + ["test"] * n_test
    return SplitPlan(seed=seed, assignment=tuple(sorted(zip(order, names, strict=True))))


def check_no_leakage(records: Iterable[SampleImage], digests: Mapping[Path, str]) -> None:
    """Refuse a manifest in which the same image content appears under two different samples.

    ``digests`` maps each image path to the SHA-256 of its file.
    """
    owner: dict[str, str] = {}
    for record in records:
        content = digests[record.image]
        previous = owner.setdefault(content, record.sample)
        if previous != record.sample:
            raise LeakageError(
                f"{record.image} has the same content as an image of sample {previous!r}. "
                f"Samples {previous!r} and {record.sample!r} would be split apart and share "
                "pixels; if they are one specimen, give their images one sample name"
            )
