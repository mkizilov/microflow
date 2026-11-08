"""Typed records for batch runs: per-image outcomes, the run manifest and progress events."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ItemStatus = Literal["ok", "failed", "cancelled"]
RunStatus = Literal["complete", "completed_with_failures", "cancelled"]


class OutputFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str
    size_bytes: int


class ItemRecord(BaseModel):
    """Outcome of one image, stored as the item's result.json. That file is written last."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: ItemStatus
    resumed: bool = False
    input_sha256: str | None = None
    config_fingerprint: str | None = None
    estimated_mb: float = 0.0
    objects: int = 0
    flags: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: str | None = None
    detail: str | None = None
    elapsed_s: float = 0.0
    outputs: dict[str, OutputFile] = Field(default_factory=dict)


class BatchManifest(BaseModel):
    """Final record of a finished run. It is written only when the run ends, so a manifest
    exists only for a run that finished; its ``status`` says how it finished."""

    model_config = ConfigDict(frozen=True)

    schema_version: int
    name: str
    status: RunStatus
    microflow_version: str
    config_fingerprint: str
    workers: int
    memory_budget_mb: float | None
    peak_estimated_mb: float
    counts: dict[str, int]
    items: tuple[ItemRecord, ...]


@dataclass(frozen=True)
class ProgressEvent:
    completed: int
    total: int
    name: str
    status: ItemStatus
    detail: str
