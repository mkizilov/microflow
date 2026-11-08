"""The sample manifest: which image and mask belong to which original sample.

A sample is the unit that splitting works on. Every image of one original specimen, field of view
or acquisition must carry the same ``sample`` name, so that all patches cut from it stay in one
split.
"""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from microflow.config import ConfigError, format_validation_error

SAMPLE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


class SampleImage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: str
    image: Path
    mask: Path

    @field_validator("sample")
    @classmethod
    def _sample_name(cls, value: str) -> str:
        if not SAMPLE_NAME.fullmatch(value):
            raise ValueError(
                "sample names use letters, digits, '.', '_' or '-', and start with a letter "
                f"or digit (got {value!r})"
            )
        return value


class _ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: list[SampleImage] = Field(min_length=1)


def read_manifest(path: Path) -> tuple[SampleImage, ...]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read file: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    try:
        manifest = _ManifestFile.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(path, exc)) from exc

    records = tuple(
        record.model_copy(
            update={
                "image": _resolve(record.image, path.parent),
                "mask": _resolve(record.mask, path.parent),
            }
        )
        for record in manifest.samples
    )
    _check_files(path, records)
    return records


def _check_files(path: Path, records: tuple[SampleImage, ...]) -> None:
    seen: set[Path] = set()
    for record in records:
        for role, file in (("image", record.image), ("mask", record.mask)):
            if not file.is_file():
                raise ConfigError(f"{path}: {role} file not found: {file}")
        if record.image.resolve() == record.mask.resolve():
            raise ConfigError(f"{path}: {record.sample}: an image cannot be its own mask")
        if record.image.resolve() in seen:
            raise ConfigError(f"{path}: image {record.image} is listed more than once")
        seen.add(record.image.resolve())


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path
