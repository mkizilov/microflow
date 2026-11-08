"""Writing statistical results: the tables, the run summary and the output-directory guard.

Tables are CSV with LF line endings. Floats use their shortest round-trip form, so identical results
give identical bytes. Missing numbers are written as ``NA``, never as an empty cell, so a gap cannot
be
mistaken for zero. ``summary.json`` is written last: its presence marks a finished run.
"""

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

from microflow import __version__
from microflow.provenance.guard import prepare_directory
from microflow.provenance.hashing import digest
from microflow.provenance.record import utc_now
from microflow.provenance.versions import library_versions
from microflow.stats.analysis import AnalysisResult, ComparisonRow, DescriptiveRow, SampleRow
from microflow.stats.design import StatsDesign, design_settings

SCHEMA_VERSION = 1
SUMMARY_FILE = "summary.json"
ANALYSIS_KEY = "analysis_id"
DESCRIPTIVE_FILE = "descriptive.csv"
COMPARISON_FILE = "comparisons.csv"
SAMPLE_FILE = "sample_values.csv"
PLOT_DIR = "plots"
MANAGED = (DESCRIPTIVE_FILE, COMPARISON_FILE, SAMPLE_FILE, PLOT_DIR, SUMMARY_FILE)


def analysis_identity(
    design: StatsDesign, measurements_sha256: str, dictionary_sha256: str | None
) -> str:
    """Identity of a statistical analysis: its settings, its input content, and the software that
    computed it. Output location is excluded."""
    return digest(
        {
            "schema": SCHEMA_VERSION,
            "microflow": __version__,
            "design": design_settings(design),
            "measurements": measurements_sha256,
            "data_dictionary": dictionary_sha256,
            "libraries": library_versions(("numpy", "scipy", "matplotlib")),
        }
    )


def write_tables(directory: Path, result: AnalysisResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_rows(directory / DESCRIPTIVE_FILE, DescriptiveRow, result.descriptive)
    _write_rows(directory / COMPARISON_FILE, ComparisonRow, result.comparisons)
    _write_rows(directory / SAMPLE_FILE, SampleRow, result.samples)


def write_summary(
    directory: Path,
    result: AnalysisResult,
    *,
    design: StatsDesign,
    identity: str,
    inputs: Mapping[str, str | None],
    plots: Sequence[str],
) -> None:
    summary: dict[str, Any] = {
        "analysis_id": identity,
        "schema_version": SCHEMA_VERSION,
        "name": result.name,
        "microflow_version": __version__,
        "libraries": library_versions(("numpy", "scipy", "matplotlib")),
        "generated_utc": utc_now(),
        "inputs": dict(inputs),
        "design": design_settings(design),
        "correction": result.correction,
        "confidence": result.confidence,
        "test_families": result.families,
        "counts": {
            "measures": len(design.measures),
            "groups": len(design.groups),
            "comparisons": len(result.comparisons),
            "samples": sum(1 for row in result.samples if row.included),
            "excluded_samples": sum(1 for row in result.samples if not row.included),
        },
        "warnings": list(result.warnings),
        "plots": list(plots),
    }
    (directory / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def prepare_output(directory: Path, identity: str, *, overwrite: bool) -> None:
    prepare_directory(
        directory,
        identity=identity,
        identity_file=SUMMARY_FILE,
        identity_key=ANALYSIS_KEY,
        managed=MANAGED,
        overwrite=overwrite,
        label="statistical results",
    )


def _write_rows(path: Path, row_type: type, rows: Sequence[Any]) -> None:
    names = [field.name for field in fields(row_type)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(names)
        for row in rows:
            writer.writerow([_text(getattr(row, name)) for name in names])


def _text(value: Any) -> str:
    if isinstance(value, tuple):
        return ";".join(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "NA" if math.isnan(value) else repr(value)
    return str(value)
