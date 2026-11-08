"""The job executor: bounded admission, a bounded thread pool, cancellation and persistent status.

A job moves queued -> running -> one final state. Every transition is a guarded store update, so a
cancellation and the start of a job cannot both win. Jobs run inline inside their pool thread (batch
workers default to 1), so the service never forks from a multithreaded process. Torch-free and
FastAPI-free.

Final states and exit codes follow the batch command: succeeded (0), completed_with_failures (3),
cancelled (130). Refused or broken runs are failed (1), with a stable error code and a message that
has the server's absolute paths replaced by the labels clients may see.
"""

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from microflow.batch import BatchConfig, BatchError, run_batch
from microflow.config import ConfigError
from microflow.plugins import PluginError
from microflow.provenance.record import utc_now
from microflow.service.errors import ServiceError
from microflow.service.paths import redact
from microflow.service.settings import SERVICE_WORKERS, ServiceSettings
from microflow.service.store import JobRow, JobStore
from microflow.service.validation import PreparedBatch

LOG = logging.getLogger("microflow.service")
EXIT_CODES: dict[str, int] = {
    "succeeded": 0,
    "completed_with_failures": 3,
    "cancelled": 130,
    "failed": 1,
}
OUTCOME_STATES: dict[str, str] = {
    "complete": "succeeded",
    "completed_with_failures": "completed_with_failures",
    "cancelled": "cancelled",
}
INTERRUPTED_MESSAGE = (
    "the service stopped before this job finished; submit it again to reuse its cached results"
)
INTERNAL_MESSAGE = "unexpected error; see the service log"


@dataclass(frozen=True)
class RunControl:
    cancelled: Callable[[], bool]
    progress: Callable[[int, int], None]


@dataclass(frozen=True)
class RunOutcome:
    status: str
    counts: dict[str, int]
    items: int
    workers: int


Runner = Callable[[BatchConfig, RunControl], RunOutcome]


def make_batch_runner() -> Runner:
    def run(config: BatchConfig, control: RunControl) -> RunOutcome:
        manifest = run_batch(
            config,
            progress=lambda event: control.progress(event.completed, event.total),
            cancel=control.cancelled,
            workers=SERVICE_WORKERS,
        )
        return RunOutcome(
            status=manifest.status,
            counts=dict(manifest.counts),
            items=len(manifest.items),
            workers=manifest.workers,
        )

    return run


class JobManager:
    def __init__(self, settings: ServiceSettings, store: JobStore, runner: Runner) -> None:
        self._store = store
        self._runner = runner
        self._capacity = settings.max_concurrent_jobs + settings.max_queued_jobs
        self._lock = threading.Lock()
        self._controls: dict[str, threading.Event] = {}
        self._closing = False
        self.interrupted = store.mark_interrupted(utc_now(), INTERRUPTED_MESSAGE)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_jobs, thread_name_prefix="microflow-job"
        )

    def submit(
        self,
        *,
        kind: str,
        prepared: PreparedBatch,
        recheck: Callable[[], PreparedBatch],
    ) -> JobRow:
        with self._lock:
            if self._closing:
                raise ServiceError("unavailable", "the service is shutting down")
            active = self._store.count_active()
            if active >= self._capacity:
                raise ServiceError(
                    "queue_full",
                    f"{active} job(s) are queued or running and the limit is {self._capacity}; "
                    "try again later",
                )
            job = JobRow(
                id=uuid.uuid4().hex,
                kind=kind,
                status="queued",
                config_name=prepared.config_path.name,
                config_path=str(prepared.config_path),
                output_name=prepared.output_name,
                output_directory=str(prepared.output_directory),
                submitted_utc=utc_now(),
            )
            self._store.insert(job)
            control = threading.Event()
            self._controls[job.id] = control
            self._executor.submit(self._run, job.id, recheck, control)
        return job

    def get(self, job_id: str) -> JobRow:
        job = self._store.get(job_id)
        if job is None:
            raise ServiceError("not_found", "no such job")
        return job

    def list_jobs(self, *, statuses: tuple[str, ...] | None, limit: int) -> list[JobRow]:
        return self._store.list_jobs(statuses=statuses, limit=limit)

    def cancel(self, job_id: str) -> JobRow:
        self.get(job_id)
        if self._store.cancel_queued(job_id, utc_now()):
            return self.get(job_id)
        if self._store.request_cancel(job_id):
            with self._lock:
                control = self._controls.get(job_id)
            if control is not None:
                control.set()
            return self.get(job_id)
        current = self.get(job_id)
        raise ServiceError("job_finished", f"the job is already {current.status}")

    def shutdown(self) -> None:
        """Stop accepting jobs, cancel the running ones and wait for them to end. Jobs that never
        started are marked interrupted, so they can be submitted again."""
        with self._lock:
            self._closing = True
            for control in self._controls.values():
                control.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._store.mark_interrupted(utc_now(), INTERRUPTED_MESSAGE)

    def _run(
        self, job_id: str, recheck: Callable[[], PreparedBatch], control: threading.Event
    ) -> None:
        try:
            if self._store.claim(job_id, utc_now()):
                self._execute(job_id, recheck, control)
        finally:
            with self._lock:
                self._controls.pop(job_id, None)

    def _execute(
        self, job_id: str, recheck: Callable[[], PreparedBatch], control: threading.Event
    ) -> None:
        job = self.get(job_id)
        replacements = {job.output_directory: job.output_name, job.config_path: job.config_name}
        try:
            prepared = recheck()
            if str(prepared.output_directory) != job.output_directory:
                raise ServiceError(
                    "invalid_config",
                    "the output directory changed after submission; submit the configuration again",
                )
            outcome = self._runner(
                prepared.config,
                RunControl(
                    cancelled=control.is_set,
                    progress=lambda done, total: self._store.update_progress(job_id, done, total),
                ),
            )
            self._finish(job_id, outcome)
        except ServiceError as exc:
            self._fail(job_id, exc.code, redact(exc.message, replacements))
        except BatchError as exc:
            self._fail(job_id, "batch_refused", redact(str(exc), replacements))
        except (ConfigError, PluginError) as exc:
            self._fail(job_id, "invalid_config", redact(str(exc), replacements))
        except OSError as exc:
            self._fail(job_id, "io_error", redact(str(exc), replacements))
        except Exception:
            LOG.exception("job %s failed unexpectedly", job_id)
            self._fail(job_id, "internal", INTERNAL_MESSAGE)

    def _finish(self, job_id: str, outcome: RunOutcome) -> None:
        status = OUTCOME_STATES[outcome.status]
        self._store.finish(
            job_id,
            status=status,
            exit_code=EXIT_CODES[status],
            error_code=None,
            error_message=None,
            summary={"counts": outcome.counts, "items": outcome.items, "workers": outcome.workers},
            finished_utc=utc_now(),
        )

    def _fail(self, job_id: str, code: str, message: str) -> None:
        self._store.finish(
            job_id,
            status="failed",
            exit_code=EXIT_CODES["failed"],
            error_code=code,
            error_message=message,
            summary=None,
            finished_utc=utc_now(),
        )
