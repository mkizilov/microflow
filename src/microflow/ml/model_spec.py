"""What a checkpoint must declare before it is used for inference. Torch-free.

Format, schema, architecture, normalisation and threshold decide what the model computes, so they
are checked strictly. The MicroFlow minor version must match, because the code around the network
(normalisation, tiling, thresholding) changes with it. A different PyTorch minor version only
produces a warning, because the same weights normally give the same numbers.
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from microflow import __version__
from microflow.ml.preparation import Normalization
from microflow.ml.runtime import ModelCompatibilityError

CHECKPOINT_FORMAT = "microflow.unet"
CHECKPOINT_SCHEMA = 2
ARCHITECTURE_LIMITS: dict[str, tuple[int, int]] = {
    "in_channels": (1, 1),
    "base_channels": (1, 128),
    "depth": (1, 5),
}
REQUIRED_KEYS = (
    "architecture",
    "normalization",
    "threshold",
    "training_id",
    "microflow_version",
    "training_calibrations",
    "epoch",
)
_VERSION = re.compile(r"(\d+)\.(\d+)")


@dataclass(frozen=True)
class ModelSpec:
    architecture: dict[str, int]
    normalization: Normalization
    threshold: float
    training_id: str
    microflow_version: str
    torch_version: str
    training_calibrations: tuple[tuple[float, str], ...]
    epoch: int
    warnings: tuple[str, ...] = ()


def minor_version(version: str) -> tuple[int, int] | None:
    match = _VERSION.search(version)
    return None if match is None else (int(match.group(1)), int(match.group(2)))


def validate_metadata(
    metadata: Mapping[str, Any],
    *,
    source: str,
    runtime_torch_version: str | None = None,
    microflow_version: str = __version__,
) -> ModelSpec:
    missing = [key for key in REQUIRED_KEYS if key not in metadata]
    if missing:
        raise ModelCompatibilityError(
            f"{source}: the checkpoint does not declare {', '.join(missing)}; "
            "retrain it with this MicroFlow version"
        )
    recorded = str(metadata["microflow_version"])
    if minor_version(recorded) != minor_version(microflow_version):
        raise ModelCompatibilityError(
            f"{source}: written by MicroFlow {recorded}, but this is MicroFlow "
            f"{microflow_version}; the minor versions must match, so retrain the model or use "
            "a matching MicroFlow version"
        )
    threshold = _number(metadata["threshold"], source, "threshold")
    if not 0 < threshold < 1:
        raise ModelCompatibilityError(f"{source}: threshold {threshold} is not between 0 and 1")
    training_id = metadata["training_id"]
    if not isinstance(training_id, str) or not training_id:
        raise ModelCompatibilityError(f"{source}: training_id must be a non-empty text value")
    epoch = metadata["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ModelCompatibilityError(f"{source}: epoch must be a non-negative integer")
    torch_version = str(metadata.get("torch_version", "unknown"))
    warnings: list[str] = []
    if runtime_torch_version is not None and minor_version(torch_version) != minor_version(
        runtime_torch_version
    ):
        warnings.append(
            f"the checkpoint was written with PyTorch {torch_version} and this environment has "
            f"{runtime_torch_version}; outputs may differ slightly"
        )
    return ModelSpec(
        architecture=_architecture(metadata["architecture"], source),
        normalization=_normalization(metadata["normalization"], source),
        threshold=threshold,
        training_id=training_id,
        microflow_version=recorded,
        torch_version=torch_version,
        training_calibrations=_calibrations(metadata["training_calibrations"], source),
        epoch=epoch,
        warnings=tuple(warnings),
    )


def _architecture(value: Any, source: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(ARCHITECTURE_LIMITS):
        raise ModelCompatibilityError(
            f"{source}: architecture must declare exactly {', '.join(sorted(ARCHITECTURE_LIMITS))}"
        )
    architecture: dict[str, int] = {}
    for key, (low, high) in ARCHITECTURE_LIMITS.items():
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, int) or not low <= number <= high:
            raise ModelCompatibilityError(
                f"{source}: architecture.{key} must be an integer from {low} to {high}; "
                f"got {number!r}"
            )
        architecture[key] = number
    return architecture


def _normalization(value: Any, source: str) -> Normalization:
    if not isinstance(value, Mapping) or set(value) != {"mean", "std"}:
        raise ModelCompatibilityError(f"{source}: normalization must declare mean and std")
    mean = _number(value["mean"], source, "normalization.mean")
    std = _number(value["std"], source, "normalization.std")
    if not std > 0:
        raise ModelCompatibilityError(f"{source}: normalization.std must be positive")
    return Normalization(mean=mean, std=std)


def _calibrations(value: Any, source: str) -> tuple[tuple[float, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ModelCompatibilityError(f"{source}: training_calibrations must be a list")
    found: set[tuple[float, str]] = set()
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 2:
            raise ModelCompatibilityError(
                f"{source}: each training calibration is a [pixel_size, unit] pair"
            )
        pixel_size = _number(item[0], source, "training calibration pixel size")
        if not pixel_size > 0 or not isinstance(item[1], str):
            raise ModelCompatibilityError(f"{source}: invalid training calibration {item!r}")
        found.add((pixel_size, item[1]))
    return tuple(sorted(found))


def _number(value: Any, source: str, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ModelCompatibilityError(f"{source}: {name} must be a finite number; got {value!r}")
    return float(value)
