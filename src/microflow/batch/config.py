"""Batch configuration: the dataset, the shared processing settings and the execution limits.

Relative paths in the file (image paths and the output directory) resolve against the directory
that contains the file, as they do for the pipeline configuration.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from microflow.config import (
    ConfigError,
    DatasetManifest,
    QualityControlConfig,
    SegmentationConfig,
    format_validation_error,
)
from microflow.preprocessing.steps import Step


class BatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="microflow-batch", min_length=1)
    dataset: DatasetManifest
    preprocessing: list[Step] = Field(default_factory=list)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    segmentation_channel: str | None = None
    quality_control: QualityControlConfig = Field(default_factory=QualityControlConfig)
    output_directory: Path = Path("batch-results")
    workers: int = Field(default=2, ge=1, le=8)
    memory_budget_mb: float | None = Field(default=None, gt=0)


def load_batch_config(path: Path) -> BatchConfig:
    raw = _read_mapping(path)
    try:
        config = BatchConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(path, exc)) from exc
    base = path.parent
    images = [
        image.model_copy(update={"path": _resolve(image.path, base)})
        for image in config.dataset.images
    ]
    dataset = config.dataset.model_copy(update={"images": images})
    return config.model_copy(
        update={
            "dataset": dataset,
            "output_directory": _resolve(config.output_directory, base),
        }
    )


def _read_mapping(path: Path) -> dict[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read file: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping of batch settings")
    return raw


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path
