"""Example segmenter: foreground is the set of pixels above a chosen intensity quantile."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import ndimage

from microflow.plugins.api import PLUGIN_API_VERSION


class QuantileOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    quantile: float = Field(default=0.9, gt=0, lt=1, description="intensity quantile threshold")


class ExampleQuantileSegmenter:
    name = "example-quantile"
    version = "1.0.0"
    api_version = PLUGIN_API_VERSION
    kind = "segmenter"
    options_model = QuantileOptions

    def segment(
        self, plane: np.ndarray, calibration: object, options: QuantileOptions
    ) -> np.ndarray:
        finite = plane[np.isfinite(plane)]
        if finite.size == 0:
            return np.zeros(plane.shape, dtype=np.int32)
        threshold = np.quantile(finite, options.quantile)
        labels, _ = ndimage.label(np.isfinite(plane) & (plane > threshold))
        return labels.astype(np.int32)
