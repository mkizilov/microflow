"""Training configuration: data, split, model, patches, augmentation and optimisation settings.

Relative paths in the file (the manifest and the output directory) resolve against the directory
that contains the file.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from microflow.config import ConfigError, format_validation_error


class SplitConfig(BaseModel):
    """Fractions of original samples held out for validation and test. Training gets the rest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    validation: float = Field(default=0.2, gt=0, lt=1)
    test: float = Field(default=0.2, gt=0, lt=1)

    @model_validator(mode="after")
    def _leaves_room_for_training(self) -> "SplitConfig":
        if self.validation + self.test >= 1:
            raise ValueError("validation and test together must be less than 1")
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_channels: int = Field(default=8, ge=1, le=128)
    depth: int = Field(default=2, ge=1, le=5)


class PatchConfig(BaseModel):
    """Training patches are square crops of this size, drawn at random from each training image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    size: int = Field(default=64, ge=8, le=1024)
    per_image: int = Field(default=4, ge=1, le=1000)


class AugmentConfig(BaseModel):
    """Geometric changes apply to the image and its mask together. Intensity changes apply to the
    image only. ``intensity`` scales the normalised image by 1 +/- value and shifts it by +/- value;
    ``noise_std`` adds Gaussian noise with that standard deviation, in normalised units."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    flip: bool = True
    rotate90: bool = True
    intensity: float = Field(default=0.1, ge=0, le=1)
    noise_std: float = Field(default=0.0, ge=0, le=1)


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(default=10, ge=1, le=10000)
    batch_size: int = Field(default=4, ge=1, le=512)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    bce_weight: float = Field(default=0.5, ge=0, le=1)
    threshold: float = Field(default=0.5, gt=0, lt=1)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="microflow-unet", min_length=1)
    manifest: Path
    output_directory: Path = Path("ml-results")
    seed: int = Field(default=0, ge=0)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    patches: PatchConfig = Field(default_factory=PatchConfig)
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    training: OptimizationConfig = Field(default_factory=OptimizationConfig)
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    num_workers: int = Field(default=0, ge=0, le=8)

    @model_validator(mode="after")
    def _patches_fit_the_network(self) -> "TrainingConfig":
        multiple = 2**self.model.depth
        if self.patches.size % multiple:
            raise ValueError(
                f"patches.size ({self.patches.size}) must be a multiple of 2**model.depth "
                f"({multiple}), because the network halves the image {self.model.depth} time(s)"
            )
        return self

    def architecture(self) -> dict[str, int]:
        """What a checkpoint must match to be loaded into this configuration's network."""
        return {
            "in_channels": 1,
            "base_channels": self.model.base_channels,
            "depth": self.model.depth,
        }


def load_training_config(path: Path) -> TrainingConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read file: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping of training settings")
    try:
        config = TrainingConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(path, exc)) from exc
    base = path.parent
    return config.model_copy(
        update={
            "manifest": _resolve(config.manifest, base),
            "output_directory": _resolve(config.output_directory, base),
        }
    )


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path
