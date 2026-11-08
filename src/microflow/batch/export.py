"""Export of a finished batch run to CSV and Parquet, with a data dictionary and provenance.

Every exported column has a name that states its unit and a row in the data dictionary that
states its type and meaning. The export directory's ``provenance.json`` is written last. If it is
missing while exported files exist, the export was interrupted and is refused unless overwritten.
"""

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from microflow.batch import store
from microflow.batch.records import BatchManifest
from microflow.provenance.guard import prepare_directory, remove_path
from microflow.provenance.record import OutputFile, make_record, result_set_identity, utc_now
from microflow.provenance.table import (
    ColumnSpec,
    Table,
    write_csv,
    write_dictionary,
    write_parquet,
)

MEASUREMENTS_CSV = "measurements.csv"
MEASUREMENTS_PARQUET = "measurements.parquet"
DICTIONARY = "data_dictionary.csv"
PROVENANCE = "provenance.json"
EXPORT_FORMATS = ("csv", "parquet")
_MANAGED = (MEASUREMENTS_CSV, MEASUREMENTS_PARQUET, DICTIONARY, PROVENANCE)
_LENGTH = "px|nm|um|mm|m"
_GEOMETRY_TEXT = {
    "perimeter": "length of the 0.5-level iso-contour of the object mask",
    "centroid_row": "centroid row measured from the top edge (pixel centre at index + 0.5)",
    "centroid_col": "centroid column measured from the left edge (pixel centre at index + 0.5)",
    "bbox_min_row": "bounding-box top edge",
    "bbox_min_col": "bounding-box left edge",
    "bbox_max_row": "bounding-box bottom edge (exclusive)",
    "bbox_max_col": "bounding-box right edge (exclusive)",
    "equivalent_diameter": "diameter of the disc with the same area",
}
_GEOMETRY_NAMES = "|".join(_GEOMETRY_TEXT)
_STAT_TEXT = {
    "mean": "mean",
    "median": "median",
    "std": "population standard deviation",
    "min": "minimum",
    "max": "maximum",
}


class ExportError(RuntimeError):
    """The run cannot be exported as requested."""


@dataclass(frozen=True)
class ExportResult:
    directory: Path
    files: tuple[str, ...]
    images: tuple[str, ...]
    rows: int
    result_set_id: str
    warnings: tuple[str, ...]


def describe_column(name: str) -> ColumnSpec:
    """Type, unit and meaning of one measurement column, from its name."""
    if name in ("item", "plane"):
        return ColumnSpec(name, "string", "", f"{name} that the row belongs to")
    if name == "label":
        return ColumnSpec(name, "int64", "1", "object label within its plane (0 is background)")
    if name == "eccentricity":
        return ColumnSpec(name, "float64", "1", "0 for a disc, approaching 1 for a line")
    match = re.fullmatch(rf"area_({_LENGTH})2", name)
    if match:
        unit = f"{match.group(1)}2"
        return ColumnSpec(name, "float64", unit, "object area (pixels x pixel area)")
    match = re.fullmatch(rf"({_GEOMETRY_NAMES})_({_LENGTH})", name)
    if match:
        return ColumnSpec(name, "float64", match.group(2), _GEOMETRY_TEXT[match.group(1)])
    match = re.fullmatch(rf"(raw|corrected)_(.+)_integrated_intensity_({_LENGTH})2", name)
    if match:
        source, channel, unit = match.groups()
        return ColumnSpec(
            name,
            "float64",
            f"intensity*{unit}2",
            f"{source} channel {channel}: sum of finite intensities x pixel area",
        )
    match = re.fullmatch(r"(raw|corrected)_(.+)_(valid_pixels|mean|median|std|min|max)", name)
    if match:
        source, channel, stat = match.groups()
        if stat == "valid_pixels":
            return ColumnSpec(
                name, "int64", "count", f"{source} channel {channel}: finite pixels in the object"
            )
        return ColumnSpec(
            name,
            "float64",
            "intensity",
            f"{source} channel {channel}: {_STAT_TEXT[stat]} of finite intensities "
            "(input image units, not converted)",
        )
    raise ExportError(f"unrecognised measurement column {name!r}")


def export_run(
    run_dir: Path,
    output_dir: Path,
    *,
    formats: Sequence[str] = EXPORT_FORMATS,
    allow_partial: bool = False,
    overwrite: bool = False,
) -> ExportResult:
    chosen = tuple(dict.fromkeys(formats))
    if not chosen or any(item not in EXPORT_FORMATS for item in chosen):
        raise ExportError(f"formats must be chosen from {', '.join(EXPORT_FORMATS)}")
    started = utc_now()
    manifest = _read_manifest(run_dir)
    if manifest.status != "complete" and not allow_partial:
        raise ExportError(
            f"the run status is {manifest.status}; pass allow_partial to export only the "
            "successful images (the export records what was excluded)"
        )
    successful = [item for item in manifest.items if item.status == "ok"]
    if not successful:
        raise ExportError("the run has no successful images to export")

    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for item in successful:
        item_dir = run_dir / store.ITEMS_DIR / store.item_directory_name(item.name)
        output = item.outputs.get("measurements")
        if output is None:
            raise ExportError(f"{item.name}: no measurements output was recorded")
        problem = store.validate_outputs(item_dir, item)
        if problem is not None:
            raise ExportError(f"{item.name}: {problem}; re-run the batch before exporting")
        columns, item_rows = _read_measurements(item_dir / output.path)
        if header is None:
            header = columns
        elif columns != header:
            raise ExportError("measurement columns differ between images; the run is inconsistent")
        rows.extend({"item": item.name, **row} for row in item_rows)
    if header is None:
        raise ExportError("the run has no measurement columns")

    specs = tuple(describe_column(name) for name in ["item", *header])
    table = Table(
        columns=specs,
        data={spec.name: [_convert(spec, row[spec.name]) for row in rows] for spec in specs},
    )
    by_name = {record.name: record for record in manifest.provenance.inputs}
    inputs = tuple(by_name[item.name] for item in successful if item.name in by_name)
    excluded = [item.name for item in manifest.items if item.status != "ok"]
    warnings: list[str] = []
    if excluded:
        warnings.append(
            f"partial export: {len(successful)} of {len(manifest.items)} images; "
            f"excluded: {', '.join(excluded)}"
        )
    if manifest.status != "complete":
        warnings.append(f"source run status: {manifest.status}")

    identity = result_set_identity(manifest.provenance.analysis_id, inputs)
    prepare_directory(
        output_dir,
        identity=identity,
        identity_file=PROVENANCE,
        identity_key="result_set_id",
        managed=_MANAGED,
        overwrite=overwrite,
        label="exported results",
    )
    remove_path(output_dir / PROVENANCE)

    base = make_record(
        analysis_id=manifest.provenance.analysis_id,
        resolved_config=manifest.provenance.resolved_config,
        inputs=inputs,
        warnings=warnings,
        outputs=(),
        started_utc=started,
        finished_utc=utc_now(),
        extra_algorithm_versions={
            key: value
            for key, value in manifest.provenance.algorithm_versions.items()
            if key.startswith("plugin:")
        },
    )
    metadata = {
        "microflow.provenance": base.model_dump_json(exclude={"outputs"}),
        "microflow.schema_version": str(base.schema_version),
    }
    written: list[OutputFile] = []
    if "csv" in chosen:
        written.append(
            store.write_output(output_dir, MEASUREMENTS_CSV, lambda path: write_csv(path, table))
        )
    if "parquet" in chosen:
        written.append(
            store.write_output(
                output_dir,
                MEASUREMENTS_PARQUET,
                lambda path: write_parquet(path, table, metadata),
            )
        )
    written.append(
        store.write_output(output_dir, DICTIONARY, lambda path: write_dictionary(path, table))
    )
    final = base.model_copy(update={"outputs": tuple(written)})
    store.write_text_atomic(output_dir / PROVENANCE, final.model_dump_json(indent=2))
    return ExportResult(
        directory=output_dir,
        files=tuple(output.path for output in written) + (PROVENANCE,),
        images=tuple(item.name for item in successful),
        rows=table.length,
        result_set_id=identity,
        warnings=tuple(warnings),
    )


def _read_manifest(run_dir: Path) -> BatchManifest:
    path = run_dir / store.MANIFEST_FILE
    if not path.is_file():
        raise ExportError(
            f"{run_dir} has no {store.MANIFEST_FILE}: the run is unfinished, aborted, "
            "or was never run"
        )
    try:
        return BatchManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ExportError(f"{path} is not a valid manifest: {exc}") from exc


def _read_measurements(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _convert(spec: ColumnSpec, text: str) -> str | int | float:
    if spec.dtype == "int64":
        return int(text)
    if spec.dtype == "float64":
        return float(text)
    return text
