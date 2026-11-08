"""Dice and IoU for predicted probabilities. Scoring itself is the segmentation module's
``score_segmentation``, so the two modules agree on edge cases: two empty masks score 1.0, and a
single empty mask scores 0.0."""

import numpy as np
from numpy.typing import NDArray

from microflow.segmentation.metrics import OverlapScores, score_segmentation


def binarize(probabilities: NDArray[np.floating], threshold: float) -> NDArray[np.bool_]:
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be between 0 and 1 exclusive; got {threshold}")
    return probabilities >= threshold


def score_probabilities(
    probabilities: NDArray[np.floating], truth: NDArray[np.uint8], threshold: float
) -> OverlapScores:
    return score_segmentation(binarize(probabilities, threshold), truth)
