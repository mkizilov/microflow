"""Request and response models of the HTTP API. Pydantic only, so they load without FastAPI."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from microflow.service.paths import MAX_PATH_LENGTH
from microflow.service.store import JobRow


class ValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["batch", "stats"]
    path: str = Field(min_length=1, max_length=MAX_PATH_LENGTH)


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["batch"]
    path: str = Field(min_length=1, max_length=MAX_PATH_LENGTH)


class ValidationView(BaseModel):
    valid: bool = True
    kind: str
    summary: dict[str, Any]
    warnings: list[str]


class ErrorView(BaseModel):
    code: str
    message: str


class ProgressView(BaseModel):
    completed: int
    total: int


class JobView(BaseModel):
    id: str
    kind: str
    status: str
    config: str
    output: str
    submitted_utc: str
    started_utc: str | None
    finished_utc: str | None
    exit_code: int | None
    cancel_requested: bool
    error: ErrorView | None
    progress: ProgressView | None
    summary: dict[str, Any] | None


class JobList(BaseModel):
    jobs: list[JobView]


class AnalysisList(BaseModel):
    analyses: list[JobView]


class FileView(BaseModel):
    path: str
    size_bytes: int


class FileList(BaseModel):
    files: list[FileView]
    truncated: bool


class QcItemView(BaseModel):
    name: str
    status: str
    objects: int
    flags: list[str]
    warnings: list[str]
    qc_report: str | None


class QcView(BaseModel):
    job_id: str
    items: list[QcItemView]


def job_view(row: JobRow) -> JobView:
    error = None
    if row.error_code:
        error = ErrorView(code=row.error_code, message=row.error_message or "")
    progress = None
    if row.progress_completed is not None and row.progress_total is not None:
        progress = ProgressView(completed=row.progress_completed, total=row.progress_total)
    return JobView(
        id=row.id,
        kind=row.kind,
        status=row.status,
        config=row.config_name,
        output=row.output_name,
        submitted_utc=row.submitted_utc,
        started_utc=row.started_utc,
        finished_utc=row.finished_utc,
        exit_code=row.exit_code,
        cancel_requested=row.cancel_requested,
        error=error,
        progress=progress,
        summary=row.summary,
    )
