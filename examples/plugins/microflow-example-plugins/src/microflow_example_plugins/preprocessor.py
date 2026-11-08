"""Example preprocessor: adds a constant offset to every valid pixel."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from microflow.plugins.api import PLUGIN_API_VERSION


class OffsetOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    offset: float = Field(allow_inf_nan=False, description="value added to every pixel")


class ExampleOffset:
    name = "example-offset"
    version = "1.0.0"
    api_version = PLUGIN_API_VERSION
    kind = "preprocessor"
    options_model = OffsetOptions

    def process(self, data: np.ndarray, calibration: object, options: OffsetOptions) -> np.ndarray:
        return data + options.offset
