"""Prediction quality for one plane: what the probabilities say, and what they imply for the masks.

Flags are warnings. They point at conditions a person should look at; they do not mark a result as
failed. Thresholds are module constants so that every report uses the same definitions. Torch-free.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

UNCERTAINTY_BAND = 0.1
"""Probabilities within this distance of the threshold count as uncertain."""
HIGH_UNCERTAINTY = 0.2
"""Flag the plane when more than this fraction of pixels is uncertain."""
EXCESSIVE_COVERAGE = 0.9
"""Flag the plane when more than this fraction of pixels is foreground."""


@dataclass(frozen=True)
class PlaneQuality:
    pixels: int
    threshold: float
    foreground_fraction: float
    object_count: int
    border_object_count: int
    mean_object_probability: float | None
    uncertain_fraction: float
    flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixels": self.pixels,
            "threshold": self.threshold,
            "foreground_fraction": self.foreground_fraction,
            "object_count": self.object_count,
            "border_object_count": self.border_object_count,
            "mean_object_probability": self.mean_object_probability,
            "uncertain_fraction": self.uncertain_fraction,
            "flags": list(self.flags),
        }


def assess_plane(
    probabilities: NDArray[np.floating],
    labels: NDArray[np.int32],
    *,
    threshold: float,
    band: float = UNCERTAINTY_BAND,
    extra_flags: Sequence[str] = (),
) -> PlaneQuality:
    pixels = int(labels.size)
    foreground = labels > 0
    objects = int(labels.max(initial=0))
    flags: list[str] = []
    if objects == 0:
        flags.append("empty_prediction")
    fraction = float(foreground.mean()) if pixels else 0.0
    if fraction > EXCESSIVE_COVERAGE:
        flags.append("excessive_coverage")
    uncertain = float((np.abs(probabilities - threshold) <= band).mean()) if pixels else 0.0
    if uncertain > HIGH_UNCERTAINTY:
        flags.append("high_uncertainty")
    flags.extend(extra_flags)
    return PlaneQuality(
        pixels=pixels,
        threshold=threshold,
        foreground_fraction=fraction,
        object_count=objects,
        border_object_count=_border_objects(labels),
        mean_object_probability=(
            float(probabilities[foreground].mean()) if foreground.any() else None
        ),
        uncertain_fraction=uncertain,
        flags=tuple(flags),
    )


def calibration_flag(
    calibration: tuple[float, str] | None,
    training: tuple[tuple[float, str], ...],
) -> str | None:
    """Compare the input's pixel size with the sizes the model was trained on. A model trained on
    uncalibrated images has nothing to compare with, so it raises no flag."""
    if not training:
        return None
    if calibration is None:
        return "calibration_missing"
    for pixel_size, unit in training:
        if unit == calibration[1] and np.isclose(pixel_size, calibration[0], rtol=1e-9, atol=0):
            return None
    return "calibration_mismatch"


def _border_objects(labels: NDArray[np.int32]) -> int:
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return int(np.unique(edge[edge > 0]).size)
