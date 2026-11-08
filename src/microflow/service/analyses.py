"""Completed analyses: the files of a finished run, and its quality-control summary. Torch-free.

Files and QC are served only for completed analyses (succeeded or completed with failures). A run
directory is checked again on every request, so a directory that was moved or deleted after the job
finished is reported as missing rather than read from somewhere else.
"""

from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from microflow.batch.records import BatchManifest
from microflow.batch.store import ITEMS_DIR, MANIFEST_FILE, item_directory_name
from microflow.service.errors import ServiceError
from microflow.service.models import QcItemView
from microflow.service.paths import ResultFile, list_result_files, resolve_result_file
from microflow.service.store import COMPLETED_STATES, JobRow


def run_directory(job: JobRow, output_roots: Sequence[Path]) -> Path:
    if job.status not in COMPLETED_STATES:
        raise ServiceError(
            "not_completed",
            f"the job is {job.status}; results are served only for completed analyses",
        )
    directory = Path(job.output_directory).resolve()
    if not directory.is_dir() or not any(directory.is_relative_to(root) for root in output_roots):
        raise ServiceError("not_found", "the results of this job are no longer present")
    return directory


def list_files(job: JobRow, output_roots: Sequence[Path]) -> tuple[tuple[ResultFile, ...], bool]:
    return list_result_files(run_directory(job, output_roots))


def download_path(job: JobRow, output_roots: Sequence[Path], relative: object) -> Path:
    return resolve_result_file(run_directory(job, output_roots), relative)


def qc_items(job: JobRow, output_roots: Sequence[Path]) -> list[QcItemView]:
    """Per-image quality control from the run's manifest. Each report path is relative to the run
    directory, so it can be downloaded with the files endpoint."""
    manifest_path = run_directory(job, output_roots) / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ServiceError("not_completed", "the run has no final manifest")
    try:
        manifest = BatchManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ServiceError("internal", "the run manifest cannot be read") from exc
    items: list[QcItemView] = []
    for item in manifest.items:
        report = item.outputs.get("qc_report")
        report_path = None
        if report is not None:
            report_path = f"{ITEMS_DIR}/{item_directory_name(item.name)}/{report.path}"
        items.append(
            QcItemView(
                name=item.name,
                status=item.status,
                objects=item.objects,
                flags=list(item.flags),
                warnings=list(item.warnings),
                qc_report=report_path,
            )
        )
    return items
