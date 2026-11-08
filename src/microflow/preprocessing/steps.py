"""Declarative preprocessing steps, validated from YAML.

A steps file has the form::

    steps:
      - kind: subtract_background
        radius: {value: 10, unit: um}
      - kind: gaussian
        sigma: {value: 1, unit: um}

Steps run in the order listed. Nothing runs unless it is listed.
"""

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from microflow.config import ConfigError, format_validation_error
from microflow.plugins.api import PLUGIN_NAME_PATTERN
from microflow.units import Length


class SubtractConstant(BaseModel):
    """Subtract a fixed intensity, for example a measured camera offset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["subtract_constant"]
    value: float


class SubtractBackground(BaseModel):
    """Subtract a morphological opening estimate of the background; radius sets the window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["subtract_background"]
    radius: Length


class GaussianFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["gaussian"]
    sigma: Length

    @field_validator("sigma")
    @classmethod
    def _positive_sigma(cls, sigma: Length) -> Length:
        if sigma.value <= 0:
            raise ValueError("sigma must be greater than zero")
        return sigma


class MedianFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["median"]
    radius: Length


class MaskArtifacts(BaseModel):
    """Mark saturated and/or zero-valued pixels invalid. Place this before arithmetic steps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["mask_artifacts"]
    saturated: bool = True
    zero: bool = False


class Normalize(BaseModel):
    """Affine rescale to a target range. Its scale and offset are recorded in the provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["normalize"]
    lower_percentile: float = Field(default=1.0, ge=0, le=100)
    upper_percentile: float = Field(default=99.0, ge=0, le=100)
    target_min: float = 0.0
    target_max: float = 1.0

    @model_validator(mode="after")
    def _ordered(self) -> "Normalize":
        if self.lower_percentile >= self.upper_percentile:
            raise ValueError("lower_percentile must be less than upper_percentile")
        if self.target_min >= self.target_max:
            raise ValueError("target_min must be less than target_max")
        return self


class PluginStep(BaseModel):
    """Runs a registered preprocessor plugin. Naming it here is what enables it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["plugin"]
    plugin: str
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("plugin")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not re.fullmatch(PLUGIN_NAME_PATTERN, value):
            raise ValueError(
                f"plugin names are lower-case letters, digits, '-' or '_' (got {value!r})"
            )
        return value


Step = Annotated[
    SubtractConstant
    | SubtractBackground
    | GaussianFilter
    | MedianFilter
    | MaskArtifacts
    | Normalize
    | PluginStep,
    Field(discriminator="kind"),
]


class StepFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[Step] = Field(default_factory=list)


def load_steps(path: Path) -> list[Step]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read file: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if raw is None:
        return []
    try:
        return StepFile.model_validate(raw).steps
    except ValidationError as exc:
        raise ConfigError(format_validation_error(path, exc)) from exc
