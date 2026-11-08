"""Batch processing: many datasets through preprocessing, segmentation, measurement and reports.

Runs are bounded (workers and an optional memory budget), resumable (validated cached results),
cancellable, and publish their final status only when they finish. Each run records its
provenance, refuses to overwrite results of a different analysis, and can be exported to CSV and
Parquet. See docs/batch.md and docs/provenance.md.
"""

from microflow.batch.config import BatchConfig, load_batch_config
from microflow.batch.engine import (
    ANALYSIS_SECTIONS,
    CacheReport,
    CacheStatus,
    analysis_id,
    inspect_cache,
    run_batch,
)
from microflow.batch.export import ExportError, ExportResult, export_run
from microflow.batch.records import BatchManifest, ItemRecord, OutputFile, ProgressEvent
from microflow.batch.store import BatchError

__all__ = [
    "ANALYSIS_SECTIONS",
    "BatchConfig",
    "BatchError",
    "BatchManifest",
    "CacheReport",
    "CacheStatus",
    "ExportError",
    "ExportResult",
    "ItemRecord",
    "OutputFile",
    "ProgressEvent",
    "analysis_id",
    "export_run",
    "inspect_cache",
    "load_batch_config",
    "run_batch",
]
