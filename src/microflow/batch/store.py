"""Filesystem state for batch runs: atomic publishing, cache validation and the run lock.

Publishing rules. Every file is written to a ``.partial`` sibling and moved into place with
``os.replace``. An image's result.json is written last, so an image is only ever recorded as
"ok" once all of its outputs exist and are hashed. The run manifest is written last of all, and
only when the run finishes.
"""

import csv
import hashlib
import io
import os
import re
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from microflow.batch.records import ItemRecord
from microflow.provenance.hashing import sha256_file
from microflow.provenance.record import OutputFile

RESULT_FILE = "result.json"
ITEMS_DIR = "items"
MANIFEST_FILE = "batch_manifest.json"
SUMMARY_FILE = "summary.csv"
RUN_FILE = "run.json"
ANALYSIS_FILE = "analysis.json"
LOCK_FILE = "run.lock"
CANCEL_FILE = "cancel.requested"
PARTIAL = ".partial"
NO_PREVIOUS = "no previous result"
CACHED = "cached"
_SUMMARY_COLUMNS = (
    "name",
    "status",
    "resumed",
    "objects",
    "flags",
    "warnings",
    "error",
    "estimated_mb",
    "elapsed_s",
)


class BatchError(RuntimeError):
    """The batch cannot start or continue, for example because another run holds the directory."""


def item_directory_name(name: str) -> str:
    """A readable, filesystem-safe directory name that stays unique for distinct image names."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:48] or "item"
    return f"{safe}-{hashlib.sha256(name.encode('utf-8')).hexdigest()[:8]}"


def write_text_atomic(path: Path, text: str) -> None:
    partial = path.with_name(path.name + PARTIAL)
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def write_output(item_dir: Path, name: str, writer: Callable[[Path], None]) -> OutputFile:
    """Write one output through a ``.partial`` file, then publish it and record its hash."""
    final = item_dir / name
    partial = item_dir / (name + PARTIAL)
    writer(partial)
    os.replace(partial, final)
    return OutputFile(path=name, sha256=sha256_file(final), size_bytes=final.stat().st_size)


def write_result(item_dir: Path, record: ItemRecord) -> None:
    write_text_atomic(item_dir / RESULT_FILE, record.model_dump_json(indent=2))


def clear_item(item_dir: Path) -> None:
    """Empty an item directory, so no output from an earlier attempt can be mistaken for current."""
    if item_dir.exists():
        shutil.rmtree(item_dir)
    item_dir.mkdir(parents=True)


def validate_outputs(item_dir: Path, record: ItemRecord) -> str | None:
    """Why the recorded outputs of an item cannot be trusted, or None if they all check out."""
    for output in record.outputs.values():
        path = item_dir / output.path
        if not path.is_file():
            return f"output {output.path} is missing"
        if path.stat().st_size != output.size_bytes or sha256_file(path) != output.sha256:
            return f"output {output.path} was modified"
    return None


def check_cache(item_dir: Path, input_sha: str, fingerprint: str) -> tuple[ItemRecord | None, str]:
    """Return the recorded result if it can be reused, otherwise None and the reason.

    Reuse requires an "ok" result whose input hash and configuration fingerprint match and whose
    every output is present with the recorded size and hash.
    """
    result_path = item_dir / RESULT_FILE
    if not result_path.exists():
        return None, NO_PREVIOUS
    try:
        previous = ItemRecord.model_validate_json(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "previous result unreadable"
    if previous.status != "ok":
        return None, f"previous run {previous.status}"
    if previous.input_sha256 != input_sha:
        return None, "input changed"
    if previous.config_fingerprint != fingerprint:
        return None, "configuration changed"
    problem = validate_outputs(item_dir, previous)
    if problem is not None:
        return None, problem
    return previous.model_copy(update={"resumed": True}), CACHED


def reset_run_directory(run_dir: Path) -> None:
    """Remove final markers and partial files left by an earlier run. Results are kept."""
    for name in (MANIFEST_FILE, SUMMARY_FILE, CANCEL_FILE):
        (run_dir / name).unlink(missing_ok=True)
    for stale in run_dir.rglob(f"*{PARTIAL}"):
        stale.unlink(missing_ok=True)


def write_summary(path: Path, items: tuple[ItemRecord, ...]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_SUMMARY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for item in items:
        writer.writerow(
            {
                "name": item.name,
                "status": item.status,
                "resumed": item.resumed,
                "objects": item.objects,
                "flags": ";".join(item.flags),
                "warnings": "; ".join(item.warnings),
                "error": item.error or "",
                "estimated_mb": f"{item.estimated_mb:.3f}",
                "elapsed_s": f"{item.elapsed_s:.3f}",
            }
        )
    write_text_atomic(path, buffer.getvalue())


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """Hold the run directory for one run. A lock whose process has exited is taken over."""
    lock = run_dir / LOCK_FILE
    _acquire(lock)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _acquire(lock: Path) -> None:
    for _ in range(2):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = _read_pid(lock)
            if holder is not None and _pid_alive(holder):
                raise BatchError(
                    f"{lock.parent}: another batch run (pid {holder}) is using this directory"
                ) from None
            lock.unlink(missing_ok=True)
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return
    raise BatchError(f"{lock.parent}: could not acquire the run lock")


def _read_pid(lock: Path) -> int | None:
    try:
        return int(lock.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
