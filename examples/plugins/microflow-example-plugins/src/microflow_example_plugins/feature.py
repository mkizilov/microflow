"""Example feature extractor: the intensity range (maximum minus minimum) within each object."""

import numpy as np
from pydantic import BaseModel, ConfigDict

from microflow.plugins.api import PLUGIN_API_VERSION, FeatureColumn


class NoOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExampleObjectRange:
    name = "example-object-range"
    version = "1.0.0"
    api_version = PLUGIN_API_VERSION
    kind = "feature"
    options_model = NoOptions

    def features(self) -> tuple[FeatureColumn, ...]:
        return (
            FeatureColumn(
                name="intensity_range",
                unit="intensity",
                description="maximum minus minimum of the finite intensities in the object",
            ),
        )

    def measure(
        self, labels: np.ndarray, plane: np.ndarray, calibration: object, options: NoOptions
    ) -> dict[int, dict[str, float]]:
        values: dict[int, dict[str, float]] = {}
        for label in np.unique(labels):
            if label == 0:
                continue
            inside = plane[(labels == label) & np.isfinite(plane)]
            value = float(inside.max() - inside.min()) if inside.size else float("nan")
            values[int(label)] = {"intensity_range": value}
        return values
