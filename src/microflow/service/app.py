"""The HTTP API. The only service module that imports FastAPI.

Every route except ``/v1/health`` needs ``Authorization: Bearer <token>``. The OpenAPI schema
and the interactive docs are disabled, so the surface is not advertised. Errors have one shape:
``{"error": {"code", "message", "details"}}``. Downloads are attachments with a restrictive content
security policy, because quality-control reports are HTML.
"""

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from microflow import __version__
from microflow.config import ConfigError
from microflow.service import analyses
from microflow.service.errors import ServiceError
from microflow.service.jobs import JobManager, Runner, make_batch_runner
from microflow.service.models import (
    AnalysisList,
    FileList,
    FileView,
    JobList,
    JobView,
    QcView,
    SubmitRequest,
    ValidateRequest,
    ValidationView,
    job_view,
)
from microflow.service.settings import MIN_TOKEN_LENGTH, ServiceSettings, prepare_layout
from microflow.service.store import COMPLETED_STATES, JOB_STATES, JobStore
from microflow.service.validation import batch_summary, prepare_batch, prepare_stats

LOG = logging.getLogger("microflow.service")
DOWNLOAD_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; img-src data:; style-src 'unsafe-inline'",
    "Cache-Control": "no-store",
}


def _manager(request: Request) -> JobManager:
    manager: JobManager = request.app.state.manager
    return manager


Manager = Annotated[JobManager, Depends(_manager)]
Limit = Annotated[int, Query(ge=1, le=200)]


def create_app(settings: ServiceSettings, *, token: str, runner: Runner | None = None) -> FastAPI:
    expected = _expected_token(token)
    layout = prepare_layout(settings)
    batch_runner = runner if runner is not None else make_batch_runner()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = JobStore(layout.state_path)
        manager = JobManager(settings, store, batch_runner)
        if manager.interrupted:
            LOG.warning(
                "%d job(s) left unfinished by an earlier run were marked interrupted",
                manager.interrupted,
            )
        app.state.manager = manager
        try:
            yield
        finally:
            await asyncio.to_thread(manager.shutdown)
            store.close()

    app = FastAPI(
        title="MicroFlow analysis service",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        scheme, _, presented = (authorization or "").partition(" ")
        supplied = presented.strip().encode("utf-8")
        valid = scheme.lower() == "bearer" and secrets.compare_digest(supplied, expected)
        if not valid or not supplied:
            raise ServiceError("unauthorized", "a valid bearer token is required")

    public = APIRouter(prefix="/v1")
    protected = APIRouter(prefix="/v1", dependencies=[Depends(require_token)])

    @public.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @protected.post("/validate", response_model=ValidationView)
    def validate(body: ValidateRequest) -> ValidationView:
        if body.kind == "stats":
            summary = prepare_stats(body.path, layout)
            return ValidationView(kind="stats", summary=summary, warnings=[])
        prepared = prepare_batch(body.path, layout)
        return ValidationView(
            kind="batch",
            summary=batch_summary(prepared),
            warnings=list(prepared.warnings),
        )

    @protected.post("/jobs", status_code=202, response_model=JobView)
    def submit(body: SubmitRequest, manager: Manager) -> JobView:
        prepared = prepare_batch(body.path, layout)
        job = manager.submit(
            kind="batch",
            prepared=prepared,
            recheck=lambda: prepare_batch(body.path, layout),
        )
        return job_view(job)

    @protected.get("/jobs", response_model=JobList)
    def list_jobs(manager: Manager, status: str | None = None, limit: Limit = 50) -> JobList:
        if status is not None and status not in JOB_STATES:
            raise ServiceError("bad_request", f"status must be one of: {', '.join(JOB_STATES)}")
        rows = manager.list_jobs(statuses=(status,) if status is not None else None, limit=limit)
        return JobList(jobs=[job_view(row) for row in rows])

    @protected.get("/jobs/{job_id}", response_model=JobView)
    def get_job(job_id: str, manager: Manager) -> JobView:
        return job_view(manager.get(job_id))

    @protected.post("/jobs/{job_id}/cancel", status_code=202, response_model=JobView)
    def cancel_job(job_id: str, manager: Manager) -> JobView:
        return job_view(manager.cancel(job_id))

    @protected.get("/analyses", response_model=AnalysisList)
    def list_analyses(manager: Manager, limit: Limit = 50) -> AnalysisList:
        rows = manager.list_jobs(statuses=COMPLETED_STATES, limit=limit)
        return AnalysisList(analyses=[job_view(row) for row in rows])

    @protected.get("/analyses/{job_id}/files", response_model=FileList)
    def analysis_files(job_id: str, manager: Manager) -> FileList:
        files, truncated = analyses.list_files(manager.get(job_id), layout.output_roots)
        return FileList(
            files=[FileView(path=item.path, size_bytes=item.size_bytes) for item in files],
            truncated=truncated,
        )

    @protected.get("/analyses/{job_id}/files/{file_path:path}")
    def analysis_file(job_id: str, file_path: str, manager: Manager) -> FileResponse:
        path = analyses.download_path(manager.get(job_id), layout.output_roots, file_path)
        return FileResponse(path, filename=path.name, headers=DOWNLOAD_HEADERS)

    @protected.get("/analyses/{job_id}/qc", response_model=QcView)
    def analysis_qc(job_id: str, manager: Manager) -> QcView:
        job = manager.get(job_id)
        return QcView(job_id=job.id, items=analyses.qc_items(job, layout.output_roots))

    app.include_router(public)
    app.include_router(protected)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if exc.code == "unauthorized" else None
        body = {"error": {"code": exc.code, "message": exc.message, "details": list(exc.details)}}
        return JSONResponse(status_code=exc.status, content=body, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        ]
        body = {
            "error": {
                "code": "invalid_request",
                "message": "the request does not match the API",
                "details": details,
            }
        }
        return JSONResponse(status_code=422, content=body)

    return app


def _expected_token(token: str) -> bytes:
    if len(token) < MIN_TOKEN_LENGTH:
        raise ConfigError(f"the service token must be at least {MIN_TOKEN_LENGTH} characters")
    return token.encode("utf-8")
