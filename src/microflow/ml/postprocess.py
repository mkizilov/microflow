"""Labelled-mask postprocessing for probability maps. Torch-free.

ML and classical segmentation share one output contract. Label 0 is background, objects are
8-connected and numbered 1..N in raster order of their first pixel, and the minimum-area and
border rules are the classical ones. Thresholding uses :func:`microflow.ml.metrics.binarize`, the
same rule as scoring.
"""

import numpy as np
from numpy.typing import NDArray

from microflow.ml.metrics import binarize
from microflow.segmentation import algorithms as alg


def labels_from_probabilities(
    probabilities: NDArray[np.floating],
    *,
    threshold: float,
    min_area_px: float = 0.0,
    exclude_border: bool = False,
) -> NDArray[np.int32]:
    labels = alg.label_objects(binarize(probabilities, threshold))
    labels = alg.remove_small(labels, min_area_px)
    if exclude_border:
        labels = alg.remove_border(labels)
    return alg.relabel_in_raster_order(labels)
