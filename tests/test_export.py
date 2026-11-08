import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from microflow.batch import BatchConfig, BatchManifest, ExportError, export_run, run_batch
from microflow.batch.store import RUN_FILE, item_directory_name
from microflow.image_io import write_image
from microflow.provenance.guard import OutputConflictError
from microflow.provenance.record import result_set_identity
from microflow.sample_data import Disc, make_phantom

DISCS = [Disc(centre=(30, 30), radius=8), Disc(centre=(30, 95), radius=6)]

# The exported column contract. Changing any name, order or unit is a deliberate format change.
EXPECTED_HEADER = [
    "item",
    "plane",
    "label",
    "area_um2",
    "perimeter_um",
    "centroid_row_um",
    "centroid_col_um",
    "bbox_min_row_um",
    "bbox_min_col_um",
    "bbox_max_row_um",
    "bbox_max_col_um",
    "eccentricity",
    "equivalent_diameter_um",
    "raw_c0_valid_pixels",
    "raw_c0_mean",
    "raw_c0_median",
    "raw_c0_std",
    "raw_c0_min",
    "raw_c0_max",
    "raw_c0_integrated_intensity_um2",
    "corrected_c0_valid_pixels",
    "corrected_c0_mean",
    "corrected_c0_median",
    "corrected_c0_std",
    "corrected_c0_min",
    "corrected_c0_max",
    "corrected_c0_integrated_intensity_um2",
]


def _phantom(path: Path, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, _ = make_phantom(DISCS, seed=seed)
    write_image(path, image.astype(np.uint16))
    return path


def _config(root: Path, images: list[dict[str, Any]], **overrides: Any) -> BatchConfig:
    values: dict[str, Any] = {
        "name": "export-batch",
        "dataset": {"images": images},
        "segmentation": {"min_object_area": {"value": 9, "unit": "um"}},
        "output_directory": root / "out",
        "workers": 1,
    }
    values.update(overrides)
    return BatchConfig.model_validate(values)


def _images(root: Path) -> list[dict[str, Any]]:
    paths = [_phantom(root / "data" / f"img_{seed}.tif", seed) for seed in (1, 2)]
    return [
        {"name": f"img_{seed}", "path": path, "calibration": {"pixel_size": 1.0, "unit": "um"}}
        for seed, path in zip((1, 2), paths, strict=True)
    ]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


@pytest.fixture(scope="module")
def shared_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, BatchManifest]:
    root = tmp_path_factory.mktemp("shared-run")
    config = _config(root, _images(root))
    manifest = run_batch(config)
    assert manifest.status == "complete"
    return config.output_directory, manifest


# --- output stability ---------------------------------------------------------------------------


def test_csv_header_is_the_documented_column_contract(
    shared_run: tuple[Path, BatchManifest], tmp_path: Path
) -> None:
    run_dir, _ = shared_run
    result = export_run(run_dir, tmp_path / "exp", formats=["csv"])
    header, rows = _read_csv(tmp_path / "exp" / "measurements.csv")
    assert header == EXPECTED_HEADER
    assert result.rows == len(rows) == 4  # two discs in each of two images
    assert result.images == ("img_1", "img_2")


def test_data_dictionary_gives_type_unit_and_meaning_for_every_column(
    shared_run: tuple[Path, BatchManifest], tmp_path: Path
) -> None:
    run_dir, _ = shared_run
    export_run(run_dir, tmp_path / "exp", formats=["csv"])
    _, dictionary = _read_csv(tmp_path / "exp" / "data_dictionary.csv")
    by_name = {row["name"]: row for row in dictionary}
    assert [row["name"] for row in dictionary] == EXPECTED_HEADER
    assert by_name["area_um2"]["unit"] == "um2"
    assert by_name["perimeter_um"]["unit"] == "um"
    assert by_name["eccentricity"]["unit"] == "1"
    assert by_name["raw_c0_valid_pixels"]["dtype"] == "int64"
    assert by_name["raw_c0_valid_pixels"]["unit"] == "count"
    assert by_name["raw_c0_mean"]["dtype"] == "float64"
    assert by_name["raw_c0_mean"]["unit"] == "intensity"
    assert by_name["corrected_c0_integrated_intensity_um2"]["unit"] == "intensity*um2"
    assert all(row["description"] for row in dictionary)


def test_parquet_holds_the_same_values_units_and_provenance_as_the_csv(
    shared_run: tuple[Path, BatchManifest], tmp_path: Path
) -> None:
    run_dir, manifest = shared_run
    export_run(run_dir, tmp_path / "exp", formats=["csv", "parquet"])
    header, rows = _read_csv(tmp_path / "exp" / "measurements.csv")
    table = pq.read_table(tmp_path / "exp" / "measurements.parquet")

    assert table.column_names == header
    columns = table.to_pydict()
    for name in header:
        for text, value in zip((row[name] for row in rows), columns[name], strict=True):
            if isinstance(value, str):
                assert text == value
            elif isinstance(value, int):
                assert int(text) == value
            elif math.isnan(value):
                assert text == "nan"
            else:
                assert float(text) == value

    metadata = table.schema.metadata
    provenance = json.loads(metadata[b"microflow.provenance"].decode("utf-8"))
    assert provenance["analysis_id"] == manifest.provenance.analysis_id
    assert metadata[b"microflow.schema_version"] == b"1"
    assert table.schema.field("area_um2").metadata[b"unit"] == b"um2"
    assert table.schema.field("eccentricity").metadata[b"unit"] == b"1"


def test_repeated_exports_produce_identical_csv_and_dictionary_bytes(
    shared_run: tuple[Path, BatchManifest], tmp_path: Path
) -> None:
    run_dir, _ = shared_run
    first = export_run(run_dir, tmp_path / "first", formats=["csv"])
    second = export_run(run_dir, tmp_path / "second", formats=["csv"])
    for name in ("measurements.csv", "data_dictionary.csv"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    assert first.result_set_id == second.result_set_id

    before = json.loads((tmp_path / "first" / "provenance.json").read_text(encoding="utf-8"))
    after = json.loads((tmp_path / "second" / "provenance.json").read_text(encoding="utf-8"))
    for key in ("analysis_id", "result_set_id", "inputs", "resolved_config", "outputs"):
        assert before[key] == after[key]


def test_provenance_of_an_export_names_its_source_and_its_outputs(
    shared_run: tuple[Path, BatchManifest], tmp_path: Path
) -> None:
    run_dir, manifest = shared_run
    result = export_run(run_dir, tmp_path / "exp", formats=["csv", "parquet"])
    record = json.loads((tmp_path / "exp" / "provenance.json").read_text(encoding="utf-8"))
    assert record["analysis_id"] == manifest.provenance.analysis_id
    assert record["result_set_id"] == result_set_identity(
        manifest.provenance.analysis_id, manifest.provenance.inputs
    )
    assert record["schema_version"] == 1
    assert record["resolved_config"] == manifest.provenance.resolved_config
    assert {output["path"] for output in record["outputs"]} == {
        "measurements.csv",
        "measurements.parquet",
        "data_dictionary.csv",
    }
    for output in record["outputs"]:
        assert (tmp_path / "exp" / output["path"]).exists()
    assert result.files[-1] == "provenance.json"


def test_csv_only_export_writes_no_parquet_file(
    shared_run: tuple[Path, BatchManifest], tmp_path: Path
) -> None:
    run_dir, _ = shared_run
    export_run(run_dir, tmp_path / "exp", formats=["csv"])
    assert not (tmp_path / "exp" / "measurements.parquet").exists()
    assert (tmp_path / "exp" / "measurements.csv").exists()


def test_unknown_export_format_is_rejected(
    shared_run: tuple[Path, BatchManifest], tmp_path: Path
) -> None:
    run_dir, _ = shared_run
    with pytest.raises(ExportError, match="formats must be chosen from"):
        export_run(run_dir, tmp_path / "exp", formats=["xlsx"])


# --- overwrite protection -----------------------------------------------------------------------


def test_export_refuses_results_of_another_analysis_and_leaves_unrelated_files(
    tmp_path: Path,
) -> None:
    images = _images(tmp_path)
    first = run_batch(_config(tmp_path / "a", images))
    second = run_batch(
        _config(
            tmp_path / "b",
            images,
            segmentation={"min_object_area": {"value": 4, "unit": "um"}},
        )
    )
    target = tmp_path / "exports"
    export_run(tmp_path / "a" / "out", target)
    (target / "notes.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(OutputConflictError, match="different analysis or data set"):
        export_run(tmp_path / "b" / "out", target)
    assert (target / "notes.txt").read_text(encoding="utf-8") == "keep me"
    assert json.loads((target / "provenance.json").read_text())["analysis_id"] == (
        first.provenance.analysis_id
    )

    export_run(tmp_path / "b" / "out", target, overwrite=True)
    assert (target / "notes.txt").read_text(encoding="utf-8") == "keep me"
    record = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    assert record["analysis_id"] == second.provenance.analysis_id


def test_export_refuses_existing_outputs_that_have_no_record(
    shared_run: tuple[Path, BatchManifest], tmp_path: Path
) -> None:
    run_dir, _ = shared_run
    target = tmp_path / "exports"
    target.mkdir()
    (target / "measurements.csv").write_text("someone else's file\n", encoding="utf-8")
    (target / "notes.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(OutputConflictError, match="no recorded identity"):
        export_run(run_dir, target)
    assert (target / "measurements.csv").read_text(encoding="utf-8").startswith("someone")

    export_run(run_dir, target, overwrite=True)
    header, _ = _read_csv(target / "measurements.csv")
    assert header == EXPECTED_HEADER
    assert (target / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_interrupted_export_is_refused_until_overwritten(
    shared_run: tuple[Path, BatchManifest], tmp_path: Path
) -> None:
    run_dir, _ = shared_run
    target = tmp_path / "exports"
    export_run(run_dir, target)
    # The export was interrupted before its provenance record was written.
    (target / "provenance.json").unlink()

    with pytest.raises(OutputConflictError):
        export_run(run_dir, target)
    export_run(run_dir, target, overwrite=True)
    assert (target / "provenance.json").exists()


def test_same_result_set_can_be_refreshed_without_overwrite(
    shared_run: tuple[Path, BatchManifest], tmp_path: Path
) -> None:
    run_dir, _ = shared_run
    target = tmp_path / "exports"
    export_run(run_dir, target)
    export_run(run_dir, target)
    assert (target / "measurements.csv").exists()


# --- runs that did not complete -----------------------------------------------------------------


def test_run_with_failures_is_refused_unless_partial_export_is_requested(
    tmp_path: Path,
) -> None:
    images = _images(tmp_path)
    images.append({"name": "missing", "path": tmp_path / "data" / "absent.tif"})
    manifest = run_batch(_config(tmp_path, images))
    assert manifest.status == "completed_with_failures"
    run_dir = tmp_path / "out"

    with pytest.raises(ExportError, match="allow_partial"):
        export_run(run_dir, tmp_path / "exp")

    result = export_run(run_dir, tmp_path / "exp", allow_partial=True)
    assert result.images == ("img_1", "img_2")
    assert any(text.startswith("partial export") for text in result.warnings)
    assert any("missing" in text for text in result.warnings)
    record = json.loads((tmp_path / "exp" / "provenance.json").read_text(encoding="utf-8"))
    assert any(text.startswith("partial export") for text in record["warnings"])


def test_run_without_a_manifest_cannot_be_exported(tmp_path: Path) -> None:
    run_dir = tmp_path / "unfinished"
    run_dir.mkdir()
    (run_dir / RUN_FILE).write_text(json.dumps({"state": "aborted"}), encoding="utf-8")
    with pytest.raises(ExportError, match="has no batch_manifest.json"):
        export_run(run_dir, tmp_path / "exp")


def test_tampered_item_output_blocks_the_export(tmp_path: Path) -> None:
    config = _config(tmp_path, _images(tmp_path))
    manifest = run_batch(config)
    item = manifest.items[0]
    item_dir = config.output_directory / "items" / item_directory_name(item.name)
    with (item_dir / "measurements.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ExportError, match="was modified"):
        export_run(config.output_directory, tmp_path / "exp")
