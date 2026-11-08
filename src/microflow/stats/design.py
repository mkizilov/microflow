"""Study designs: which images belong to which biological samples, and which samples to which
groups.

Every image (a batch item) belongs to exactly one biological sample, and every sample to exactly one
group. Nothing is inferred from file names. The design also states the measures, the comparisons and
the analysis settings, so that a result can be reproduced from the design file and the measurement
table alone. Relative paths resolve against the directory that contains the design file.
"""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from microflow.config import ConfigError, format_validation_error

NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"


class SampleDefinition(BaseModel):
    """One biological sample: a specimen, donor, culture or animal. ``items`` are the images
    (batch item names) taken from it. Technical replicates are images, not samples."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: str = Field(pattern=NAME_PATTERN)
    items: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _items_are_distinct(self) -> "SampleDefinition":
        if len(set(self.items)) != len(self.items):
            raise ValueError(f"sample {self.sample}: an item is listed more than once")
        if any(not item.strip() for item in self.items):
            raise ValueError(f"sample {self.sample}: item names cannot be blank")
        return self


class GroupDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    samples: tuple[SampleDefinition, ...] = Field(min_length=1)


class BootstrapSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resamples: int = Field(default=10000, ge=100, le=1_000_000)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    seed: int = Field(default=0, ge=0)


class PermutationSettings(BaseModel):
    """Permutation tests enumerate every split when there are at most ``max_exact`` of them, and
    otherwise draw ``resamples`` random splits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resamples: int = Field(default=10000, ge=100, le=1_000_000)
    max_exact: int = Field(default=20000, ge=1)
    seed: int = Field(default=0, ge=0)


class StatsDesign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    measurements: Path
    data_dictionary: Path | None = None
    output_directory: Path = Path("stats-results")
    measures: tuple[str, ...] = Field(min_length=1)
    per_sample: Literal["mean", "median"] = "mean"
    groups: dict[str, GroupDefinition]
    comparisons: tuple[tuple[str, str], ...] = Field(min_length=1)
    bootstrap: BootstrapSettings = Field(default_factory=BootstrapSettings)
    permutation: PermutationSettings = Field(default_factory=PermutationSettings)
    correction: Literal["bh", "holm", "none"] = "bh"
    alpha: float = Field(default=0.05, gt=0, lt=1)
    min_samples_per_group: int = Field(default=2, ge=2)
    plots: bool = True

    @model_validator(mode="after")
    def _consistent(self) -> "StatsDesign":
        if len(self.groups) < 2:
            raise ValueError("a comparison needs at least two groups")
        for name in self.groups:
            if not _is_name(name):
                raise ValueError(f"group name {name!r} must use letters, digits, '.', '_' or '-'")
        samples = [sample.sample for group in self.groups.values() for sample in group.samples]
        if len(set(samples)) != len(samples):
            repeated = sorted({name for name in samples if samples.count(name) > 1})
            raise ValueError(f"samples belong to more than one group or repeat: {repeated}")
        items = [item for group in self.groups.values() for s in group.samples for item in s.items]
        if len(set(items)) != len(items):
            repeated = sorted({name for name in items if items.count(name) > 1})
            raise ValueError(f"items belong to more than one sample: {repeated}")
        if len(set(self.measures)) != len(self.measures):
            raise ValueError("measures must be listed once each")
        seen: set[frozenset[str]] = set()
        for first, second in self.comparisons:
            for name in (first, second):
                if name not in self.groups:
                    raise ValueError(f"comparison refers to undeclared group {name!r}")
            if first == second:
                raise ValueError(f"comparison {first} vs {second} compares a group with itself")
            pair = frozenset((first, second))
            if pair in seen:
                raise ValueError(f"comparison {first} vs {second} is listed more than once")
            seen.add(pair)
        return self


def load_design(path: Path) -> StatsDesign:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read file: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping of study settings")
    try:
        design = StatsDesign.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(path, exc)) from exc
    base = path.parent
    update = {
        "measurements": _resolve(design.measurements, base),
        "output_directory": _resolve(design.output_directory, base),
    }
    if design.data_dictionary is not None:
        update["data_dictionary"] = _resolve(design.data_dictionary, base)
    return design.model_copy(update=update)


def design_settings(design: StatsDesign) -> dict[str, object]:
    """The settings that determine the results. Locations are excluded, because moving the files
    does not change the analysis; their content is hashed separately."""
    return design.model_dump(
        mode="json",
        exclude={"output_directory", "measurements", "data_dictionary"},
    )


def _is_name(value: str) -> bool:
    return re.fullmatch(NAME_PATTERN, value) is not None


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path
