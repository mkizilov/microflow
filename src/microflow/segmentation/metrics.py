"""Overlap metrics between a predicted segmentation and a ground-truth segmentation."""

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict


class OverlapScores(BaseModel):
    model_config = ConfigDict(frozen=True)

    dice: float
    iou: float
    predicted_pixels: int
    truth_pixels: int


def score_segmentation(
    prediction: npt.ArrayLike, truth: npt.ArrayLike
) -> OverlapScores:
    """Foreground Dice and IoU. Any non-zero value counts as foreground, so label images work.

    Dice = 2|A∩B| / (|A|+|B|) and IoU = |A∩B| / |A∪B|. If both masks are empty they agree
    perfectly, so both scores are 1.0. If only one is empty, both scores are 0.0.
    """
    predicted_array = np.asarray(prediction)
    truth_array = np.asarray(truth)
    if predicted_array.shape != truth_array.shape:
        raise ValueError(
            f"shapes differ: prediction {predicted_array.shape}, truth {truth_array.shape}"
        )
    predicted = predicted_array != 0
    actual = truth_array != 0
    intersection = int(np.count_nonzero(predicted & actual))
    predicted_total = int(np.count_nonzero(predicted))
    truth_total = int(np.count_nonzero(actual))
    union = predicted_total + truth_total - intersection
    total = predicted_total + truth_total
    dice = 1.0 if total == 0 else 2 * intersection / total
    iou = 1.0 if union == 0 else intersection / union
    return OverlapScores(
        dice=dice,
        iou=iou,
        predicted_pixels=predicted_total,
        truth_pixels=truth_total,
    )
