"""Example quality check: warns when too many objects are smaller than a size in pixels."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from microflow.plugins.api import PLUGIN_API_VERSION


class SmallObjectOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_pixels: int = Field(default=4, ge=1, description="objects below this area are small")
    max_fraction: float = Field(default=0.5, gt=0, le=1, description="warn above this share")


class ExampleSmallObjectFraction:
    name = "example-small-fraction"
    version = "1.0.0"
    api_version = PLUGIN_API_VERSION
    kind = "qc"
    options_model = SmallObjectOptions

    def check(
        self,
        labels: np.ndarray,
        plane: np.ndarray,
        calibration: object,
        options: SmallObjectOptions,
    ) -> list[str]:
        areas = np.bincount(labels.ravel())[1:]
        areas = areas[areas > 0]
        if areas.size == 0:
            return []
        fraction = float(np.mean(areas < options.min_pixels))
        if fraction > options.max_fraction:
            return [
                f"{fraction:.0%} of objects are smaller than {options.min_pixels} pixels "
                f"(limit {options.max_fraction:.0%})"
            ]
        return []
