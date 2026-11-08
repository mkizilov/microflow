"""Persistent job records in SQLite. Torch-free.

Every state change is one guarded statement (for example, ``UPDATE ... WHERE status =
'queued'``), so two threads cannot both move a job through a transition. Autocommit makes each
statement atomic, and a lock serialises access to the single connection.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ACTIVE_STATES = ("queued", "running")
COMPLETED_STATES = ("succeeded", "completed_with_failures")
JOB_STATES = (
    "queued",
    "running",
    "succeeded",
    "completed_with_failures",
    "cancelled",
    "failed",
    "interrupted",
)
CANCELLED_EXIT_CODE = 130

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    config_name TEXT NOT NULL,
    config_path TEXT NOT NULL,
    output_name TEXT NOT NULL,
    output_directory TEXT NOT NULL,
    submitted_utc TEXT NOT NULL,
    started_utc TEXT,
    finished_utc TEXT,
    exit_code INTEGER,
    error_code TEXT,
    error_message TEXT,
    summary_json TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    progress_completed INTEGER,
    progress_total INTEGER
)
"""
_COLUMNS = (
    "id, kind, status, config_name, config_path, output_name, output_directory, submitted_utc, "
    "started_utc, finished_utc, exit_code, error_code, error_message, summary_json, "
    "cancel_requested, progress_completed, progress_total"
)


@dataclass(frozen=True)
class JobRow:
    id: str
    kind: str
    status: str
    config_name: str
    config_path: str
    output_name: str
    output_directory: str
    submitted_utc: str
    started_utc: str | None = None
    finished_utc: str | None = None
    exit_code: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    summary: dict[str, Any] | None = None
    cancel_requested: bool = False
    progress_completed: int | None = None
    progress_total: int | None = None


class JobStore:
    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._connection.execute(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def insert(self, job: JobRow) -> None:
        self._write(
            f"INSERT INTO jobs ({_COLUMNS}) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL)",
            (
                job.id,
                job.kind,
                job.status,
                job.config_name,
                job.config_path,
                job.output_name,
                job.output_directory,
                job.submitted_utc,
            ),
        )

    def get(self, job_id: str) -> JobRow | None:
        rows = self._read(f"SELECT {_COLUMNS} FROM jobs WHERE id = ?", (job_id,))
        return _job(rows[0]) if rows else None

    def list_jobs(self, *, statuses: tuple[str, ...] | None, limit: int) -> list[JobRow]:
        if statuses is None:
            rows = self._read(f"SELECT {_COLUMNS} FROM jobs ORDER BY rowid DESC LIMIT ?", (limit,))
        else:
            marks = ", ".join("?" for _ in statuses)
            rows = self._read(
                f"SELECT {_COLUMNS} FROM jobs WHERE status IN ({marks}) "
                "ORDER BY rowid DESC LIMIT ?",
                (*statuses, limit),
            )
        return [_job(row) for row in rows]

    def count_active(self) -> int:
        marks = ", ".join("?" for _ in ACTIVE_STATES)
        rows = self._read(f"SELECT COUNT(*) FROM jobs WHERE status IN ({marks})", ACTIVE_STATES)
        return int(rows[0][0])

    def claim(self, job_id: str, started_utc: str) -> bool:
        return self._write(
            "UPDATE jobs SET status = 'running', started_utc = ? "
            "WHERE id = ? AND status = 'queued'",
            (started_utc, job_id),
        )

    def cancel_queued(self, job_id: str, finished_utc: str) -> bool:
        return self._write(
            "UPDATE jobs SET status = 'cancelled', finished_utc = ?, exit_code = ? "
            "WHERE id = ? AND status = 'queued'",
            (finished_utc, CANCELLED_EXIT_CODE, job_id),
        )

    def request_cancel(self, job_id: str) -> bool:
        return self._write(
            "UPDATE jobs SET cancel_requested = 1 WHERE id = ? AND status = 'running'", (job_id,)
        )

    def update_progress(self, job_id: str, completed: int, total: int) -> bool:
        return self._write(
            "UPDATE jobs SET progress_completed = ?, progress_total = ? "
            "WHERE id = ? AND status = 'running'",
            (completed, total, job_id),
        )

    def finish(
        self,
        job_id: str,
        *,
        status: str,
        exit_code: int | None,
        error_code: str | None,
        error_message: str | None,
        summary: dict[str, Any] | None,
        finished_utc: str,
    ) -> bool:
        return self._write(
            "UPDATE jobs SET status = ?, exit_code = ?, error_code = ?, error_message = ?, "
            "summary_json = ?, finished_utc = ? WHERE id = ? AND status = 'running'",
            (
                status,
                exit_code,
                error_code,
                error_message,
                None if summary is None else json.dumps(summary, sort_keys=True),
                finished_utc,
                job_id,
            ),
        )

    def mark_interrupted(self, finished_utc: str, message: str) -> int:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE jobs SET status = 'interrupted', finished_utc = ?, error_code = "
                "'interrupted', error_message = ? WHERE status IN ('queued', 'running')",
                (finished_utc, message),
            )
            return cursor.rowcount

    def _write(self, sql: str, params: tuple[Any, ...]) -> bool:
        with self._lock:
            return self._connection.execute(sql, params).rowcount == 1

    def _read(self, sql: str, params: tuple[Any, ...]) -> list[Any]:
        with self._lock:
            return self._connection.execute(sql, params).fetchall()


def _job(row: sqlite3.Row) -> JobRow:
    summary = row["summary_json"]
    return JobRow(
        id=row["id"],
        kind=row["kind"],
        status=row["status"],
        config_name=row["config_name"],
        config_path=row["config_path"],
        output_name=row["output_name"],
        output_directory=row["output_directory"],
        submitted_utc=row["submitted_utc"],
        started_utc=row["started_utc"],
        finished_utc=row["finished_utc"],
        exit_code=row["exit_code"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        summary=None if summary is None else json.loads(summary),
        cancel_requested=bool(row["cancel_requested"]),
        progress_completed=row["progress_completed"],
        progress_total=row["progress_total"],
    )
