"""Everything a training or evaluation run derives from its data, before any model is involved.

The order is fixed, and it is what keeps held-out data out of training:

1. read the manifest and hash every image and mask file;
2. refuse identical content under two samples;
3. assign whole samples to splits;
4. load each split's images and masks;
5. fit intensity normalisation on the training split only.

Validation and test pixels never reach step 5, so they cannot shift the normalisation.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from microflow import __version__
from microflow.loading import load_image
from microflow.ml.config import TrainingConfig
from microflow.ml.manifest import SampleImage, read_manifest
from microflow.ml.runtime import MLError
from microflow.ml.splitting import SPLIT_NAMES, SplitPlan, check_no_leakage, plan_split
from microflow.provenance.hashing import digest, sha256_file

RUN_FILE = "run.json"
IDENTITY_KEY = "training_id"
IDENTITY_SCHEMA = 1
_EXCLUDED_FROM_IDENTITY = ("output_directory", "num_workers", "device", "manifest")
_MIN_STD = 1e-12


@dataclass(frozen=True)
class Normalization:
    """Intensity normalisation fitted on the training images: (x - mean) / std."""

    mean: float
    std: float

    def apply(self, image: NDArray[np.float64]) -> NDArray[np.float64]:
        return (image - self.mean) / self.std

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}


def fit_normalization(images: Sequence[NDArray[np.float64]]) -> Normalization:
    if not images:
        raise MLError("there are no training images to fit the intensity normalisation on")
    values = np.concatenate([image.ravel() for image in images])
    std = float(values.std())
    if not std > _MIN_STD:
        raise MLError("the training images have no intensity variation; normalisation is undefined")
    return Normalization(mean=float(values.mean()), std=std)


@dataclass(frozen=True)
class ImagePair:
    sample: str
    image_path: Path
    image_sha256: str
    calibration: tuple[float, str] | None
    image: NDArray[np.float64]
    mask: NDArray[np.uint8]


@dataclass(frozen=True)
class PreparedRun:
    records: tuple[SampleImage, ...]
    plan: SplitPlan
    pairs: dict[str, tuple[ImagePair, ...]]
    normalization: Normalization
    identity: str


def digests_for(records: Sequence[SampleImage]) -> dict[Path, str]:
    """SHA-256 of every image and mask file, keyed by path."""
    digests: dict[Path, str] = {}
    for record in records:
        digests[record.image] = sha256_file(record.image)
        digests[record.mask] = sha256_file(record.mask)
    return digests


def plan_for(
    config: TrainingConfig, records: Sequence[SampleImage], digests: dict[Path, str]
) -> SplitPlan:
    check_no_leakage(records, digests)
    return plan_split(
        {record.sample for record in records},
        validation=config.split.validation,
        test=config.split.test,
        seed=config.seed,
    )


def load_pairs(records: Sequence[SampleImage], digests: dict[Path, str]) -> tuple[ImagePair, ...]:
    pairs: list[ImagePair] = []
    for record in records:
        image, calibration = _load_plane(record.image)
        mask, _ = _load_plane(record.mask)
        if image.shape != mask.shape:
            raise MLError(
                f"{record.mask}: mask shape {mask.shape} differs from image shape {image.shape}"
            )
        if not np.isfinite(image).all():
            raise MLError(f"{record.image}: the image contains non-finite values")
        values = np.unique(mask)
        if not np.isin(values, (0.0, 1.0)).all():
            raise MLError(
                f"{record.mask}: masks must hold 0 (background) and 1 (object); "
                f"found {values[:5].tolist()}"
            )
        pairs.append(
            ImagePair(
                sample=record.sample,
                image_path=record.image,
                image_sha256=digests[record.image],
                calibration=calibration,
                image=image,
                mask=mask.astype(np.uint8),
            )
        )
    return tuple(pairs)


def training_calibrations(pairs: Sequence[ImagePair]) -> tuple[tuple[float, str], ...]:
    """The distinct pixel sizes of the training images. Empty if none of them is calibrated."""
    return tuple(sorted({pair.calibration for pair in pairs if pair.calibration is not None}))


def _load_plane(path: Path) -> tuple[NDArray[np.float64], tuple[float, str] | None]:
    loaded = load_image(path, axes="YX")
    if loaded.data.ndim != 2:
        raise MLError(f"{path}: expected one 2-D Y-X plane, got shape {loaded.shape}")
    calibration = None
    if loaded.calibration is not None:
        calibration = (float(loaded.calibration.pixel_size), loaded.calibration.unit)
    return np.asarray(loaded.data, dtype=np.float64), calibration


def run_identity(
    config: TrainingConfig, records: Sequence[SampleImage], digests: dict[Path, str]
) -> str:
    """Identity of a training run: its settings and the content of every file it reads.

    Output location, worker count, device and manifest location are excluded, so moving the data
    or the results does not change it. Changing any setting or any pixel does.
    """
    settings = config.model_dump(mode="json", exclude=set(_EXCLUDED_FROM_IDENTITY))
    data = sorted(
        (
            {
                "sample": record.sample,
                "image": digests[record.image],
                "mask": digests[record.mask],
            }
            for record in records
        ),
        key=lambda item: (item["sample"], item["image"]),
    )
    payload: dict[str, Any] = {
        "schema": IDENTITY_SCHEMA,
        "microflow": __version__,
        "settings": settings,
        "data": data,
    }
    return digest(payload)


def prepare(config: TrainingConfig) -> PreparedRun:
    records = read_manifest(config.manifest)
    digests = digests_for(records)
    plan = plan_for(config, records, digests)
    assignment = dict(plan.assignment)
    pairs = {
        name: load_pairs(
            [record for record in records if assignment[record.sample] == name], digests
        )
        for name in SPLIT_NAMES
    }
    normalization = fit_normalization([pair.image for pair in pairs["train"]])
    identity = run_identity(config, records, digests)
    return PreparedRun(
        records=records,
        plan=plan,
        pairs=pairs,
        normalization=normalization,
        identity=identity,
    )


def same_normalization(recorded: dict[str, float], current: Normalization) -> bool:
    return all(
        math.isclose(recorded[key], value, rel_tol=1e-12, abs_tol=1e-12)
        for key, value in (("mean", current.mean), ("std", current.std))
    )
