"""Pipeline configuration models and YAML loading.

Paths in a configuration file are resolved relative to the directory containing that file.
Unknown keys are rejected so that misspelled options fail loudly instead of being ignored.
"""

from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from microflow.units import Area, Calibration, Length

AXIS_LETTERS = frozenset("TCZYX")


class ConfigError(Exception):
    """A configuration file is unreadable, malformed, or inconsistent with its images."""


def _validate_axes(value: str) -> str:
    if not value:
        raise ValueError("axes must not be empty")
    unknown = sorted(set(value) - AXIS_LETTERS)
    if unknown:
        raise ValueError(f"unknown axis letters {unknown} in {value!r}; allowed: T, C, Z, Y, X")
    if len(set(value)) != len(value):
        raise ValueError(f"axis letters must not repeat in {value!r}")
    if "Y" not in value or "X" not in value:
        raise ValueError(f"axes must include Y and X, got {value!r}")
    return value


Axes = Annotated[str, AfterValidator(_validate_axes)]


class ImageRecord(BaseModel):
    """One dataset manifest entry. Each image may have its own axes, channels and calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    path: Path
    axes: Axes = "YX"
    channels: list[str] | None = None
    calibration: Calibration | None = None

    @field_validator("channels")
    @classmethod
    def _unique_channels(cls, channels: list[str] | None) -> list[str] | None:
        if channels is None:
            return channels
        if not channels or any(not channel.strip() for channel in channels):
            raise ValueError("channel names must be non-empty when channels are given")
        if len(set(channels)) != len(channels):
            raise ValueError(f"channel names must be unique, got {channels}")
        return channels


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images: list[ImageRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> "DatasetManifest":
        names = [image.name for image in self.images]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate image names: {', '.join(duplicates)}")
        return self


class ProcessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smoothing_sigma: Length = Length(value=1.0, unit="um")
    subtract_background: bool = False


class SegmentationConfig(BaseModel):
    """Classical segmentation settings. Lengths and areas convert through the image calibration."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["otsu", "threshold", "adaptive"] = "otsu"
    threshold: float | None = None
    adaptive_block_size: Length = Length(value=10.0, unit="um")
    adaptive_offset: float = 0.0
    opening_radius: Length | None = None
    closing_radius: Length | None = None
    watershed: bool = False
    watershed_min_distance: Length = Length(value=2.0, unit="um")
    min_object_area: Area = Area(value=4.0, unit="um")
    exclude_border: bool = False

    @field_validator("adaptive_block_size", "watershed_min_distance")
    @classmethod
    def _positive_length(cls, value: Length) -> Length:
        if value.value <= 0:
            raise ValueError("must be greater than zero")
        return value

    @model_validator(mode="after")
    def _threshold_matches_method(self) -> "SegmentationConfig":
        if self.method == "threshold" and self.threshold is None:
            raise ValueError("threshold is required when method is 'threshold'")
        if self.method != "threshold" and self.threshold is not None:
            raise ValueError("threshold is only used when method is 'threshold'")
        return self


class QuantificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measures: list[Literal["area", "mean_intensity", "integrated_intensity"]] = Field(
        default=["area", "mean_intensity"], min_length=1
    )
    require_calibration: bool = True


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: Path = Path("results")
    formats: list[Literal["csv", "json"]] = Field(default=["csv"], min_length=1)
    overwrite: bool = False


class MicroFlowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="microflow-run", min_length=1)
    dataset: DatasetManifest
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    quantification: QuantificationConfig = Field(default_factory=QuantificationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _calibration_present_for_quantification(self) -> "MicroFlowConfig":
        if not self.quantification.require_calibration:
            return self
        missing = [image.name for image in self.dataset.images if image.calibration is None]
        if missing:
            raise ValueError(
                "calibration (pixel_size and unit) is required for quantification but is "
                f"missing for: {', '.join(missing)}. Set quantification.require_calibration "
                "to false to allow uncalibrated analysis."
            )
        return self


class QualityControlConfig(BaseModel):
    """Thresholds for flagging suspicious segmentation results.

    Fractions are shares of pixels in one Y-X plane. Area limits convert through the image
    calibration.
    """

    model_config = ConfigDict(extra="forbid")

    max_foreground_fraction: float = Field(default=0.6, ge=0, le=1)
    max_saturated_fraction: float = Field(default=0.001, ge=0, le=1)
    saturation_level: float | None = None
    max_invalid_fraction: float = Field(default=0.01, ge=0, le=1)
    min_object_area: Area | None = None
    max_object_area: Area | None = None

    @model_validator(mode="after")
    def _area_range_ordered(self) -> "QualityControlConfig":
        if self.min_object_area is None or self.max_object_area is None:
            return self
        minimum = self.min_object_area.to(self.max_object_area.unit).value
        if minimum > self.max_object_area.value:
            raise ValueError("min_object_area must not exceed max_object_area")
        return self


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read file: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc


_Section = TypeVar("_Section", bound=BaseModel)


def _load_section(path: Path, model: type[_Section]) -> _Section:
    """Validate a YAML mapping as one settings model. An empty file means the defaults."""
    raw = _read_yaml(path)
    if raw is None:
        raw = {}
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(path, exc)) from exc


def load_segmentation_config(path: Path) -> SegmentationConfig:
    return _load_section(path, SegmentationConfig)


def load_quality_config(path: Path) -> QualityControlConfig:
    return _load_section(path, QualityControlConfig)


def load_config(path: Path) -> MicroFlowConfig:
    raw = _read_yaml(path)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping of configuration sections")
    try:
        config = MicroFlowConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(path, exc)) from exc
    return _resolve_paths(config, path.parent)


def format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"{path}: {exc.error_count()} validation error(s)"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)


def _resolve_paths(config: MicroFlowConfig, base: Path) -> MicroFlowConfig:
    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else base / path

    images = [
        image.model_copy(update={"path": resolve(image.path)}) for image in config.dataset.images
    ]
    dataset = config.dataset.model_copy(update={"images": images})
    output = config.output.model_copy(update={"directory": resolve(config.output.directory)})
    return config.model_copy(update={"dataset": dataset, "output": output})
