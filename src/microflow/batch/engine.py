"""Batch engine: plan each image, run the pending ones with bounded parallelism, then publish.

Identity. A run directory records the analysis identity it was created with. A run with a
different identity is refused, so results of one configuration are never silently replaced by
another. ``overwrite=True`` replaces the managed files of the old analysis; unrelated files are
left alone.

Planning. Each image's input is hashed. A previous result is reused only if it still validates
(see :func:`microflow.batch.store.check_cache`). Otherwise the image is probed for its shape, its
memory estimate is compared with the budget, and it is queued.

Scheduling. At most ``workers`` images run at once, and the sum of their estimated memory stays
within ``memory_budget_mb``. One worker runs in-process.

Cancellation. A cancel request (the callable, or a ``cancel.requested`` file in the run directory)
stops new work. Images already running finish, and the rest are recorded as cancelled. Ctrl-C
has the same effect. Each image is published when it finishes; the manifest only at the end.

Publishing. The run directory is marked "running" while work proceeds. If the run raises, it is
marked "aborted". Only a finished run gets a manifest, so the manifest's presence marks success
or a recorded non-success status, never a half-finished run.
"""

import json
import math
from collections import deque
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from microflow import __version__
from microflow.batch import store
from microflow.batch.config import BatchConfig
from microflow.batch.records import BatchManifest, ItemRecord, ProgressEvent
from microflow.batch.store import BatchError
from microflow.batch.worker import ImageJob, process_image
from microflow.image_io import read_shape_and_dtype
from microflow.provenance.guard import OutputConflictError, prepare_directory, read_identity
from microflow.provenance.hashing import digest
from microflow.provenance.record import InputFile, analysis_identity, make_record, utc_now

BATCH_SCHEMA_VERSION = 2
BYTES_PER_ELEMENT = 40
POLL_SECONDS = 0.2
ANALYSIS_SECTIONS = ("preprocessing", "segmentation", "segmentation_channel", "quality_control")
_BATCH_MANAGED = (
    store.ITEMS_DIR,
    store.MANIFEST_FILE,
    store.SUMMARY_FILE,
    store.RUN_FILE,
    store.CANCEL_FILE,
    store.ANALYSIS_FILE,
)

ProgressCallback = Callable[[ProgressEvent], None]
CancelCheck = Callable[[], bool]


class CacheStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    reusable: bool
    reason: str


class CacheReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    analysis_id: str
    directory_analysis: str | None
    analysis_matches: bool
    images: tuple[CacheStatus, ...]


@dataclass
class _Slot:
    index: int
    name: str
    estimated_mb: float = 0.0
    job: ImageJob | None = None
    record: ItemRecord | None = None
    note: str = ""


class _InlineExecutor(Executor):
    """Runs each submission in the calling process. Used for one worker and for tests."""

    def submit(self, fn, /, *args, **kwargs) -> Future:  # type: ignore[override]
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


class _Tracker:
    def __init__(self, total: int, callback: ProgressCallback | None) -> None:
        self.total = total
        self.completed = 0
        self._callback = callback

    def emit(self, record: ItemRecord, note: str) -> None:
        self.completed += 1
        if self._callback is not None:
            self._callback(
                ProgressEvent(
                    completed=self.completed,
                    total=self.total,
                    name=record.name,
                    status=record.status,
                    detail=_detail(record, note),
                )
            )


def analysis_id(config: BatchConfig) -> str:
    """Identity of the analysis a configuration describes. Workers, memory budget, output
    directory and dataset membership do not change it."""
    return analysis_identity(_analysis_settings(config))


def inspect_cache(config: BatchConfig) -> CacheReport:
    """Report which images' cached results remain valid. Nothing is written or changed."""
    run_dir = config.output_directory
    current = analysis_id(config)
    recorded = read_identity(run_dir / store.ANALYSIS_FILE, "analysis_id")
    statuses: list[CacheStatus] = []
    for image in config.dataset.images:
        item_dir = run_dir / store.ITEMS_DIR / store.item_directory_name(image.name)
        try:
            input_sha = store.sha256_file(image.path)
        except OSError as exc:
            statuses.append(
                CacheStatus(name=image.name, reusable=False, reason=f"cannot read input: {exc}")
            )
            continue
        cached, reason = store.check_cache(
            item_dir, input_sha, _item_fingerprint(current, image)
        )
        statuses.append(
            CacheStatus(
                name=image.name,
                reusable=cached is not None,
                reason="valid" if cached is not None else reason,
            )
        )
    return CacheReport(
        analysis_id=current,
        directory_analysis=recorded,
        analysis_matches=recorded == current,
        images=tuple(statuses),
    )


def run_batch(
    config: BatchConfig,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
    workers: int | None = None,
    overwrite: bool = False,
) -> BatchManifest:
    """Run a batch and return its manifest.

    Raises :class:`BatchError` if the directory is busy or holds results of a different analysis
    and ``overwrite`` is not set.
    """
    worker_count = config.workers if workers is None else workers
    if worker_count < 1:
        raise BatchError("workers must be at least 1")
    run_dir = config.output_directory
    started = utc_now()
    shared = analysis_id(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    with store.run_lock(run_dir):
        try:
            prepare_directory(
                run_dir,
                identity=shared,
                identity_file=store.ANALYSIS_FILE,
                identity_key="analysis_id",
                managed=_BATCH_MANAGED,
                overwrite=overwrite,
                label="batch results",
            )
        except OutputConflictError as exc:
            raise BatchError(str(exc)) from exc
        store.reset_run_directory(run_dir)
        store.write_text_atomic(
            run_dir / store.ANALYSIS_FILE,
            json.dumps({"analysis_id": shared, "microflow_version": __version__}, indent=2),
        )
        _write_run_state(run_dir, {"state": "running", "name": config.name, "started_utc": started})
        try:
            return _execute(config, run_dir, worker_count, progress, cancel, shared, started)
        except BaseException as exc:
            _write_run_state(
                run_dir, {"state": "aborted", "error": f"{type(exc).__name__}: {exc}"}
            )
            raise


def _execute(
    config: BatchConfig,
    run_dir: Path,
    worker_count: int,
    progress: ProgressCallback | None,
    cancel: CancelCheck | None,
    shared: str,
    started: str,
) -> BatchManifest:
    slots = _plan(config, run_dir, shared)
    tracker = _Tracker(len(slots), progress)
    for slot in slots:
        if slot.record is not None:
            tracker.emit(slot.record, slot.note)

    budget = config.memory_budget_mb
    queue: deque[_Slot] = deque(slot for slot in slots if slot.job is not None)
    in_flight: dict[Future[ItemRecord], _Slot] = {}
    reserved = 0.0
    peak = 0.0
    cancelled = False
    executor = _make_executor(worker_count)
    try:
        while queue or in_flight:
            if not cancelled and _cancel_requested(run_dir, cancel):
                cancelled = True
            while not cancelled and queue and len(in_flight) < worker_count:
                head = queue[0]
                if in_flight and budget is not None and reserved + head.estimated_mb > budget:
                    break
                queue.popleft()
                try:
                    future = executor.submit(process_image, head.job)
                except BrokenProcessPool:
                    executor = _replace(executor, worker_count)
                    future = executor.submit(process_image, head.job)
                in_flight[future] = head
                reserved += head.estimated_mb
                peak = max(peak, reserved)
            if not in_flight:
                break
            try:
                done, _ = wait(in_flight, timeout=POLL_SECONDS, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:
                cancelled = True
                continue
            pool_broken = False
            for future in done:
                slot = in_flight.pop(future)
                reserved -= slot.estimated_mb
                record, broke = _collect(future, slot)
                pool_broken = pool_broken or broke
                slot.record = record
                tracker.emit(record, slot.note)
            if pool_broken:
                executor = _replace(executor, worker_count)
    finally:
        executor.shutdown(wait=True)

    for slot in queue:
        slot.record = ItemRecord(
            name=slot.name,
            status="cancelled",
            estimated_mb=slot.estimated_mb,
            detail="not started: the batch was cancelled",
        )
        tracker.emit(slot.record, slot.note)

    items = tuple(_settled(slot) for slot in slots)
    return _publish(run_dir, config, items, shared, worker_count, peak, started)


def _analysis_settings(config: BatchConfig) -> dict[str, Any]:
    return {
        "preprocessing": [step.model_dump(mode="json") for step in config.preprocessing],
        "segmentation": config.segmentation.model_dump(mode="json"),
        "segmentation_channel": config.segmentation_channel,
        "quality_control": config.quality_control.model_dump(mode="json"),
    }


def _item_fingerprint(shared: str, image: Any) -> str:
    return digest({"shared": shared, "image": image.model_dump(mode="json", exclude={"path"})})


def _plan(config: BatchConfig, run_dir: Path, shared: str) -> list[_Slot]:
    budget = config.memory_budget_mb
    slots: list[_Slot] = []
    for index, image in enumerate(config.dataset.images):
        item_dir = run_dir / store.ITEMS_DIR / store.item_directory_name(image.name)
        fingerprint = _item_fingerprint(shared, image)
        try:
            input_sha = store.sha256_file(image.path)
        except OSError as exc:
            error = f"cannot read input: {exc}"
            slots.append(_failed_slot(index, image.name, item_dir, error, None, fingerprint))
            continue

        cached, reason = store.check_cache(item_dir, input_sha, fingerprint)
        if cached is not None:
            slots.append(_Slot(index=index, name=image.name, record=cached))
            continue

        try:
            elements = math.prod(_probe_shape(image.path))
        except Exception as exc:
            slots.append(
                _failed_slot(
                    index,
                    image.name,
                    item_dir,
                    f"cannot read image header: {exc}",
                    input_sha,
                    fingerprint,
                )
            )
            continue
        estimated = elements * BYTES_PER_ELEMENT / 2**20
        if budget is not None and estimated > budget:
            slots.append(
                _failed_slot(
                    index,
                    image.name,
                    item_dir,
                    f"exceeds memory budget: estimated {estimated:.2f} MB, "
                    f"budget {budget:.2f} MB",
                    input_sha,
                    fingerprint,
                    estimated,
                )
            )
            continue

        job = ImageJob(
            name=image.name,
            path=image.path,
            expected_axes=image.axes if "axes" in image.model_fields_set else None,
            channels=tuple(image.channels) if image.channels is not None else None,
            calibration=image.calibration,
            item_dir=item_dir,
            input_sha256=input_sha,
            config_fingerprint=fingerprint,
            estimated_mb=estimated,
            steps=tuple(config.preprocessing),
            segmentation=config.segmentation,
            segmentation_channel=config.segmentation_channel,
            quality=config.quality_control,
        )
        note = "" if reason == store.NO_PREVIOUS else reason
        slots.append(
            _Slot(index=index, name=image.name, estimated_mb=estimated, job=job, note=note)
        )
    return slots


def _failed_slot(
    index: int,
    name: str,
    item_dir: Path,
    error: str,
    input_sha: str | None,
    fingerprint: str,
    estimated: float = 0.0,
) -> _Slot:
    record = ItemRecord(
        name=name,
        status="failed",
        input_sha256=input_sha,
        config_fingerprint=fingerprint,
        estimated_mb=estimated,
        error=error,
    )
    store.clear_item(item_dir)
    store.write_result(item_dir, record)
    return _Slot(index=index, name=name, record=record)


def _collect(future: Future[ItemRecord], slot: _Slot) -> tuple[ItemRecord, bool]:
    """Return the outcome of a finished submission, and whether the worker pool broke."""
    try:
        return future.result(), False
    except BrokenProcessPool:
        error = "worker process terminated unexpectedly (possibly out of memory)"
        broken = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        broken = False
    job = slot.job
    if job is None:
        raise RuntimeError(f"{slot.name}: a finished submission has no job")
    record = ItemRecord(
        name=slot.name,
        status="failed",
        input_sha256=job.input_sha256,
        config_fingerprint=job.config_fingerprint,
        estimated_mb=slot.estimated_mb,
        error=error,
    )
    store.clear_item(job.item_dir)
    store.write_result(job.item_dir, record)
    return record, broken


def _publish(
    run_dir: Path,
    config: BatchConfig,
    items: tuple[ItemRecord, ...],
    shared: str,
    workers: int,
    peak: float,
    started: str,
) -> BatchManifest:
    counts = {
        "ok": sum(item.status == "ok" for item in items),
        "failed": sum(item.status == "failed" for item in items),
        "cancelled": sum(item.status == "cancelled" for item in items),
        "resumed": sum(item.resumed for item in items),
    }
    if counts["cancelled"]:
        status = "cancelled"
    elif counts["ok"] == len(items):
        status = "complete"
    else:
        status = "completed_with_failures"
    provenance = make_record(
        analysis_id=shared,
        resolved_config=config.model_dump(mode="json"),
        inputs=_inputs(config, items),
        warnings=_run_warnings(items, status),
        outputs=(),
        started_utc=started,
        finished_utc=utc_now(),
    )
    manifest = BatchManifest(
        schema_version=BATCH_SCHEMA_VERSION,
        name=config.name,
        status=status,
        microflow_version=__version__,
        config_fingerprint=shared,
        workers=workers,
        memory_budget_mb=config.memory_budget_mb,
        peak_estimated_mb=peak,
        counts=counts,
        items=items,
        provenance=provenance,
    )
    store.write_summary(run_dir / store.SUMMARY_FILE, items)
    (run_dir / store.RUN_FILE).unlink(missing_ok=True)
    store.write_text_atomic(run_dir / store.MANIFEST_FILE, manifest.model_dump_json(indent=2))
    return manifest


def _inputs(config: BatchConfig, items: tuple[ItemRecord, ...]) -> tuple[InputFile, ...]:
    by_name = {item.name: item for item in items}
    inputs: list[InputFile] = []
    for image in config.dataset.images:
        item = by_name.get(image.name)
        if item is None or item.input_sha256 is None:
            continue
        size = image.path.stat().st_size if image.path.is_file() else None
        inputs.append(
            InputFile(
                name=image.name,
                path=str(image.path),
                sha256=item.input_sha256,
                size_bytes=size,
            )
        )
    return tuple(inputs)


def _run_warnings(items: tuple[ItemRecord, ...], status: str) -> list[str]:
    warnings: list[str] = []
    for item in items:
        if item.status == "failed":
            warnings.append(f"{item.name}: failed: {item.error}")
        elif item.status == "cancelled":
            warnings.append(f"{item.name}: not started (cancelled)")
        warnings.extend(f"{item.name}: {text}" for text in item.warnings)
    if status != "complete":
        warnings.append(f"run status: {status}")
    return warnings


def _probe_shape(path: Path) -> tuple[int, ...]:
    if path.suffix.lower() == ".npy":
        return tuple(np.load(path, mmap_mode="r").shape)
    return read_shape_and_dtype(path)[0]


def _make_executor(workers: int) -> Executor:
    return _InlineExecutor() if workers == 1 else ProcessPoolExecutor(max_workers=workers)


def _replace(executor: Executor, workers: int) -> Executor:
    executor.shutdown(wait=False, cancel_futures=True)
    return _make_executor(workers)


def _cancel_requested(run_dir: Path, cancel: CancelCheck | None) -> bool:
    return (run_dir / store.CANCEL_FILE).exists() or (cancel is not None and cancel())


def _settled(slot: _Slot) -> ItemRecord:
    if slot.record is None:
        raise RuntimeError(f"{slot.name}: no outcome was recorded")
    return slot.record


def _detail(record: ItemRecord, note: str) -> str:
    if record.status == "ok":
        if record.resumed:
            return "cached"
        text = f"objects {record.objects}, {len(record.flags)} flag(s), {record.elapsed_s:.2f}s"
        return f"{text}; rerun: {note}" if note else text
    if record.status == "failed":
        return f"error: {record.error}"
    return record.detail or "cancelled"


def _write_run_state(run_dir: Path, state: dict[str, object]) -> None:
    store.write_text_atomic(run_dir / store.RUN_FILE, json.dumps(state, indent=2))
