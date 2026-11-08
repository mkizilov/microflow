"""Service settings and start-up checks. Torch-free and FastAPI-free.

The settings file names the directories the service may read and write, its limits and its bind
address. The access token is not in the file: it comes from the environment variable ``TOKEN_ENV``
and is never logged or returned. Relative paths in the file resolve against the file's directory.
"""

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from microflow.config import ConfigError, format_validation_error

TOKEN_ENV = "MICROFLOW_SERVICE_TOKEN"
MIN_TOKEN_LENGTH = 32
DEFAULT_PORT = 21300
DATABASE_NAME = "jobs.sqlite"
# One worker per batch, run inline in its job thread. A larger value makes the batch engine start a
# process pool, which forks from this multithreaded service. Concurrency comes from running jobs.
SERVICE_WORKERS = 1
_LISTS = ("input_roots", "output_roots")


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _bind_refused(host: str) -> str:
    return (
        f"refusing to listen on {host}: only loopback addresses are accepted unless allow_remote "
        "is set (--allow-remote). Put a TLS-terminating proxy in front before doing that."
    )


def check_bind(host: str, allow_remote: bool) -> None:
    if not allow_remote and not is_loopback(host):
        raise ConfigError(_bind_refused(host))


class ServiceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state_directory: Path
    input_roots: tuple[Path, ...] = Field(min_length=1, max_length=32)
    output_roots: tuple[Path, ...] = Field(min_length=1, max_length=32)
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)
    max_queued_jobs: int = Field(default=8, ge=0, le=256)
    host: str = "127.0.0.1"
    port: int = Field(default=DEFAULT_PORT, ge=1, le=65535)
    allow_remote: bool = False

    @field_validator("state_directory")
    @classmethod
    def _state_is_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("must be an absolute path (relative paths resolve from the file)")
        return value

    @field_validator("input_roots", "output_roots")
    @classmethod
    def _roots_are_absolute_and_distinct(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        if not all(root.is_absolute() for root in value):
            raise ValueError("every directory must be an absolute path")
        if len(set(value)) != len(value):
            raise ValueError("a directory is listed more than once")
        return value

    @model_validator(mode="after")
    def _bind_is_restricted(self) -> "ServiceSettings":
        if not self.allow_remote and not is_loopback(self.host):
            raise ValueError(_bind_refused(self.host))
        return self


@dataclass(frozen=True)
class Layout:
    """The settings with every directory resolved, for the path policy to compare against."""

    input_roots: tuple[Path, ...]
    output_roots: tuple[Path, ...]
    state_path: Path


def load_settings(path: Path) -> ServiceSettings:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read file: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping of service settings")
    base = path.parent
    if isinstance(raw.get("state_directory"), str):
        raw["state_directory"] = str(_resolve(raw["state_directory"], base))
    for key in _LISTS:
        value = raw.get(key)
        if isinstance(value, list):
            raw[key] = [
                str(_resolve(item, base)) if isinstance(item, str) else item for item in value
            ]
    try:
        return ServiceSettings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(path, exc)) from exc


def token_from_environment(environ: Mapping[str, str] = os.environ) -> str:
    token = environ.get(TOKEN_ENV, "")
    if len(token) < MIN_TOKEN_LENGTH:
        raise ConfigError(
            f"set {TOKEN_ENV} to a secret of at least {MIN_TOKEN_LENGTH} characters; the service "
            "does not start without one"
        )
    return token


def prepare_layout(settings: ServiceSettings) -> Layout:
    """Resolve every directory and check the start-up conditions. Creates the state directory."""
    inputs = tuple(root.resolve() for root in settings.input_roots)
    outputs = tuple(root.resolve() for root in settings.output_roots)
    state = settings.state_directory.resolve()
    for label, roots in (("input", inputs), ("output", outputs)):
        for root in roots:
            if not root.is_dir():
                raise ConfigError(f"the {label} directory {root} does not exist")
    if any(state.is_relative_to(root) for root in outputs):
        raise ConfigError(
            "the state directory must not be inside an output directory, because results are "
            "served from output directories"
        )
    state.mkdir(parents=True, exist_ok=True)
    return Layout(input_roots=inputs, output_roots=outputs, state_path=state / DATABASE_NAME)


def _resolve(text: str, base: Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else base / path
