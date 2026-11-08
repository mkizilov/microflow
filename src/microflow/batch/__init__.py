"""Batch processing: many datasets through preprocessing, segmentation, measurement and reports.

Runs are bounded (workers and an optional memory budget), resumable (validated cached results),
cancellable, and publish their final status only when they finish. See docs/batch.md.
"""

from microflow.batch.config import BatchConfig, load_batch_config
from microflow.batch.engine import run_batch
from microflow.batch.records import BatchManifest, ItemRecord, OutputFile, ProgressEvent
from microflow.batch.store import BatchError

__all__ = [
    "BatchConfig",
    "BatchError",
    "BatchManifest",
    "ItemRecord",
    "OutputFile",
    "ProgressEvent",
    "load_batch_config",
    "run_batch",
]
