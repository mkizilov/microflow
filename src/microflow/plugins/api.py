"""Plugin interfaces: the contract between MicroFlow and third-party algorithms.

A plugin is a class published under one of four entry-point groups (see ``ENTRY_POINT_GROUPS``).
MicroFlow reads the group and name from installed package metadata without importing anything.
It imports and runs a plugin only when a trusted configuration names it: naming a plugin in the
configuration is what enables it. A dataset cannot name a plugin.

Every plugin declares these attributes:

- ``name``: the entry-point name, lower-case, matching ``PLUGIN_NAME_PATTERN``
- ``version``: the plugin's own version, recorded in the provenance of every run that used it
- ``api_version``: the plugin API version it was written for; must equal ``PLUGIN_API_VERSION``
- ``kind``: one of ``KINDS``, equal to the group it is published under
- ``options_model``: a pydantic model class. The configured options are validated against it
  before the plugin is called, and the validated instance is passed to the plugin's methods

Each kind also has methods, described in the Protocol classes below. The outputs of those methods
are checked against the contracts in ``microflow.plugins.contracts``. A plugin whose output breaks
its contract fails the item it ran on, with a message that names the plugin and the broken rule.
"""

import re
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator

from microflow.units import Calibration

PLUGIN_API_VERSION = 1
KINDS = ("preprocessor", "segmenter", "feature", "qc")
ENTRY_POINT_GROUPS = {
    "preprocessor": "microflow.preprocessors",
    "segmenter": "microflow.segmenters",
    "feature": "microflow.features",
    "qc": "microflow.qc_checks",
}
PLUGIN_NAME_PATTERN = r"[a-z][a-z0-9_-]{0,63}"
FEATURE_NAME_PATTERN = r"[a-z][a-z0-9_]{0,63}"
REQUIRED_ATTRIBUTES = ("name", "version", "api_version", "kind", "options_model")
REQUIRED_METHODS = {
    "preprocessor": ("process",),
    "segmenter": ("segment",),
    "feature": ("features", "measure"),
    "qc": ("check",),
}


class PluginError(RuntimeError):
    """A plugin cannot be found, loaded, configured, or returned output that breaks its contract."""


class PluginSpec(BaseModel):
    """A plugin named in a trusted configuration, with its options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not re.fullmatch(PLUGIN_NAME_PATTERN, value):
            raise ValueError(
                "plugin names are lower-case letters, digits, '-' or '_', starting with a "
                f"letter (got {value!r})"
            )
        return value


@dataclass(frozen=True)
class FeatureColumn:
    """One value a feature plugin reports per object."""

    name: str
    unit: str
    description: str


class PreprocessorPlugin(Protocol):
    """Transforms a whole image, for example a background or flat-field correction.

    ``process`` receives a float64 copy of the image and must return an array of the same shape.
    Invalid pixels are NaN; infinite values are rejected. Normalising intensities is allowed, but
    it changes what intensity measurements mean, so the run records a warning when such a plugin
    ran.
    """

    name: str
    version: str
    api_version: int
    kind: str
    options_model: type[BaseModel]

    def process(
        self,
        data: npt.NDArray[np.float64],
        calibration: Calibration | None,
        options: BaseModel,
    ) -> npt.ArrayLike: ...


class SegmenterPlugin(Protocol):
    """Finds objects in one Y-X plane of the preprocessed image.

    ``segment`` receives a float64 copy of the plane and must return integer labels of the same
    shape: 0 is background, objects are positive, and no label exceeds the number of pixels. The
    output is then passed through MicroFlow's minimum-area and border rules and renumbered in
    raster order.
    """

    name: str
    version: str
    api_version: int
    kind: str
    options_model: type[BaseModel]

    def segment(
        self,
        plane: npt.NDArray[np.float64],
        calibration: Calibration | None,
        options: BaseModel,
    ) -> npt.ArrayLike: ...


class FeaturePlugin(Protocol):
    """Adds per-object values computed from the raw image.

    ``features`` declares the columns. ``measure`` receives copies of the object labels of one
    plane and the raw plane of the selected channel. It must return a mapping from every object
    label in the plane to a mapping from every declared feature name to a finite number (NaN is
    allowed). It must not return labels that are not objects. The values are written to a separate
    CSV file with its own data dictionary.
    """

    name: str
    version: str
    api_version: int
    kind: str
    options_model: type[BaseModel]

    def features(self) -> tuple[FeatureColumn, ...]: ...

    def measure(
        self,
        labels: npt.NDArray[np.int32],
        plane: npt.NDArray[np.float64],
        calibration: Calibration | None,
        options: BaseModel,
    ) -> Any: ...


class QualityCheckPlugin(Protocol):
    """Reports findings about one plane. Findings are warnings: they are recorded, but they do not
    fail the item.

    ``check`` receives copies of the labels and the raw plane, and returns a sequence of non-empty
    text messages. An empty sequence means no finding.
    """

    name: str
    version: str
    api_version: int
    kind: str
    options_model: type[BaseModel]

    def check(
        self,
        labels: npt.NDArray[np.int32],
        plane: npt.NDArray[np.float64],
        calibration: Calibration | None,
        options: BaseModel,
    ) -> Any: ...
