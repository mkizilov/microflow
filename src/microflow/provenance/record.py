"""Provenance records and the identities that tie results to the analysis that produced them.

Two identities are used:

- ``analysis_id`` identifies a method: software version, algorithm versions, computational
  library versions, the output schema version and the analysis settings. It does not depend on
  input files, dataset membership, timestamps, runtime details, worker count or memory budget.
- ``result_set_id`` identifies a set of results: the analysis plus the hashes of the input files
  that were used.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict

from microflow import __version__
from microflow.provenance.hashing import digest
from microflow.provenance.versions import (
    ALGORITHM_VERSIONS,
    COMPUTATIONAL_LIBRARIES,
    library_versions,
    runtime_info,
)

OUTPUT_SCHEMA_VERSION = 1


class InputFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    path: str
    sha256: str
    size_bytes: int | None


class OutputFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str
    size_bytes: int


class ProvenanceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int
    microflow_version: str
    analysis_id: str
    result_set_id: str
    started_utc: str
    finished_utc: str
    resolved_config: dict[str, Any]
    algorithm_versions: dict[str, str]
    library_versions: dict[str, str]
    runtime: dict[str, str]
    inputs: tuple[InputFile, ...]
    warnings: tuple[str, ...]
    outputs: tuple[OutputFile, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def analysis_identity(analysis: dict[str, Any]) -> str:
    """Identity of a method, from its analysis settings and the versions that affect them."""
    return digest(
        {
            "schema": OUTPUT_SCHEMA_VERSION,
            "microflow": __version__,
            "algorithms": ALGORITHM_VERSIONS,
            "libraries": library_versions(COMPUTATIONAL_LIBRARIES),
            "analysis": analysis,
        }
    )


def result_set_identity(analysis_id: str, inputs: Sequence[InputFile]) -> str:
    return digest(
        {
            "analysis_id": analysis_id,
            "inputs": sorted([item.name, item.sha256] for item in inputs),
        }
    )


def make_record(
    *,
    analysis_id: str,
    resolved_config: dict[str, Any],
    inputs: Sequence[InputFile],
    warnings: Sequence[str],
    outputs: Sequence[OutputFile],
    started_utc: str,
    finished_utc: str,
    extra_algorithm_versions: Mapping[str, str] | None = None,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        schema_version=OUTPUT_SCHEMA_VERSION,
        microflow_version=__version__,
        analysis_id=analysis_id,
        result_set_id=result_set_identity(analysis_id, inputs),
        started_utc=started_utc,
        finished_utc=finished_utc,
        resolved_config=resolved_config,
        algorithm_versions={**ALGORITHM_VERSIONS, **dict(extra_algorithm_versions or {})},
        library_versions=library_versions(),
        runtime=runtime_info(),
        inputs=tuple(inputs),
        warnings=tuple(warnings),
        outputs=tuple(outputs),
    )
