import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile

from microflow.batch import BatchConfig, BatchError, BatchManifest, ItemRecord, run_batch
from microflow.batch.records import ProgressEvent
from microflow.batch.store import (
    CANCEL_FILE,
    ITEMS_DIR,
    LOCK_FILE,
    MANIFEST_FILE,
    RESULT_FILE,
    RUN_FILE,
    SUMMARY_FILE,
    item_directory_name,
)
from microflow.image_io import write_image
from microflow.sample_data import Disc, make_phantom

DISCS = [Disc(centre=(30, 30), radius=8), Disc(centre=(30, 95), radius=6)]


def _phantom(path: Path, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, _ = make_phantom(DISCS, seed=seed)
    write_image(path, image.astype(np.uint16))
    return path


def _image(path: Path, name: str, calibrated: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {"name": name, "path": path}
    if calibrated:
        record["calibration"] = {"pixel_size": 1.0, "unit": "um"}
    return record


def _config(root: Path, images: list[dict[str, Any]], **overrides: Any) -> BatchConfig:
    values: dict[str, Any] = {
        "name": "test-batch",
        "dataset": {"images": images},
        "segmentation": {"min_object_area": {"value": 9, "unit": "um"}},
        "output_directory": root / "out",
        "workers": 1,
    }
    values.update(overrides)
    return BatchConfig.model_validate(values)


def _items(manifest: BatchManifest) -> dict[str, ItemRecord]:
    return {item.name: item for item in manifest.items}


def _item_dir(config: BatchConfig, name: str) -> Path:
    return config.output_directory / ITEMS_DIR / item_directory_name(name)


@pytest.fixture
def three_images(tmp_path: Path) -> list[dict[str, Any]]:
    return [
        _image(_phantom(tmp_path / "data" / f"img_{seed}.tif", seed), f"img_{seed}")
        for seed in (1, 2, 3)
    ]


# --- successful batches -----------------------------------------------------------------------


def test_successful_batch_writes_outputs_and_a_complete_manifest(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    config = _config(tmp_path, three_images)
    manifest = run_batch(config)

    assert manifest.status == "complete"
    assert manifest.counts == {"ok": 3, "failed": 0, "cancelled": 0, "resumed": 0}
    for name in ("img_1", "img_2", "img_3"):
        item = _items(manifest)[name]
        assert item.status == "ok"
        assert item.objects == 2
        assert set(item.outputs) == {"labels", "measurements", "qc_report"}
        directory = _item_dir(config, name)
        for filename in ("labels.tif", "measurements.csv", "qc_report.html", RESULT_FILE):
            assert (directory / filename).exists(), filename

    labels = tifffile.imread(_item_dir(config, "img_1") / "labels.tif")
    assert labels.dtype == np.int32
    assert labels.max() == 2
    header = (_item_dir(config, "img_1") / "measurements.csv").read_text().splitlines()[0]
    assert header.split(",")[0] == "plane"
    assert "raw_c0_mean" in header.split(",")
    assert "corrected_c0_mean" in header.split(",")

    summary = (config.output_directory / SUMMARY_FILE).read_text(encoding="utf-8")
    assert summary.count("\n") == 4  # header and three rows
    assert not (config.output_directory / RUN_FILE).exists()
    assert not (config.output_directory / LOCK_FILE).exists()
    stored = BatchManifest.model_validate_json(
        (config.output_directory / MANIFEST_FILE).read_text(encoding="utf-8")
    )
    assert stored == manifest


def test_preprocessing_steps_are_applied_before_measurement(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    config = _config(
        tmp_path,
        three_images[:1],
        preprocessing=[{"kind": "subtract_constant", "value": 100}],
    )
    manifest = run_batch(config)
    assert manifest.status == "complete"
    with (_item_dir(config, "img_1") / "measurements.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    for row in rows:
        difference = float(row["raw_c0_mean"]) - float(row["corrected_c0_mean"])
        assert difference == pytest.approx(100.0)


def test_parallel_results_match_sequential_results(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    sequential = _config(tmp_path / "seq", three_images, workers=1)
    parallel = _config(tmp_path / "par", three_images, workers=2)
    run_batch(sequential)
    manifest = run_batch(parallel)

    assert manifest.status == "complete"
    for name in ("img_1", "img_2", "img_3"):
        seq_dir, par_dir = _item_dir(sequential, name), _item_dir(parallel, name)
        np.testing.assert_array_equal(
            tifffile.imread(seq_dir / "labels.tif"), tifffile.imread(par_dir / "labels.tif")
        )
        assert (seq_dir / "measurements.csv").read_text() == (
            par_dir / "measurements.csv"
        ).read_text()


def test_progress_reports_each_image_once_in_order(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    events: list[ProgressEvent] = []
    run_batch(_config(tmp_path, three_images), progress=events.append)
    assert [event.completed for event in events] == [1, 2, 3]
    assert all(event.total == 3 for event in events)
    assert {event.name for event in events} == {"img_1", "img_2", "img_3"}
    assert all(event.status == "ok" for event in events)


# --- individual failures ----------------------------------------------------------------------


def test_unreadable_inputs_are_recorded_and_the_rest_of_the_batch_completes(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    corrupt = tmp_path / "data" / "corrupt.tif"
    corrupt.write_bytes(b"not a tiff file")
    images = [
        three_images[0],
        _image(tmp_path / "data" / "absent.tif", "absent"),
        _image(corrupt, "corrupt"),
    ]
    config = _config(tmp_path, images)
    manifest = run_batch(config)
    items = _items(manifest)

    assert manifest.status == "completed_with_failures"
    assert manifest.counts["ok"] == 1 and manifest.counts["failed"] == 2
    assert items["img_1"].status == "ok"
    assert items["absent"].status == "failed"
    assert "cannot read input" in (items["absent"].error or "")
    assert items["corrupt"].status == "failed"
    assert "cannot read image header" in (items["corrupt"].error or "")
    assert items["absent"].outputs == {} and items["corrupt"].outputs == {}
    assert not (_item_dir(config, "absent") / "labels.tif").exists()
    summary = (config.output_directory / SUMMARY_FILE).read_text(encoding="utf-8")
    assert summary.count("\n") == 4


def test_error_inside_processing_is_recorded_with_its_message(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    images = [_image(three_images[0]["path"], "img_1", calibrated=False)]
    config = _config(tmp_path, images)
    manifest = run_batch(config)
    item = _items(manifest)["img_1"]

    assert manifest.status == "completed_with_failures"
    assert item.status == "failed"
    assert "needs pixel calibration" in (item.error or "")
    assert item.detail and "Traceback" in item.detail
    assert not (_item_dir(config, "img_1") / "labels.tif").exists()


def test_unknown_segmentation_channel_fails_only_that_image(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    config = _config(tmp_path, three_images[:1], segmentation_channel="DAPI")
    manifest = run_batch(config)
    assert manifest.status == "completed_with_failures"
    assert "segmentation channel 'DAPI'" in (_items(manifest)["img_1"].error or "")


# --- resume and validated cache ---------------------------------------------------------------


def test_cancelled_run_resumes_from_valid_results(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    config = _config(tmp_path, three_images)
    events: list[ProgressEvent] = []
    first = run_batch(config, progress=events.append, cancel=lambda: len(events) >= 1)
    assert first.status == "cancelled"
    assert [item.status for item in first.items] == ["ok", "cancelled", "cancelled"]
    kept = _items(first)["img_1"].outputs["labels"].sha256

    second = run_batch(config)
    assert second.status == "complete"
    assert [item.resumed for item in second.items] == [True, False, False]
    assert second.counts["resumed"] == 1
    assert _items(second)["img_1"].outputs["labels"].sha256 == kept


def test_changed_input_reruns_only_that_image(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    config = _config(tmp_path, three_images)
    run_batch(config)
    _phantom(Path(three_images[1]["path"]), seed=99)
    items = _items(run_batch(config))
    assert items["img_1"].resumed and items["img_3"].resumed
    assert not items["img_2"].resumed
    assert items["img_2"].status == "ok"


def test_modified_output_triggers_a_rerun_of_that_image(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    config = _config(tmp_path, three_images[:2])
    run_batch(config)
    with (_item_dir(config, "img_1") / "measurements.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    manifest = run_batch(config)
    assert not _items(manifest)["img_1"].resumed
    assert _items(manifest)["img_2"].resumed
    assert manifest.status == "complete"


def test_configuration_change_is_refused_until_overwrite_is_requested(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    first = run_batch(_config(tmp_path, three_images))
    changed = _config(
        tmp_path,
        three_images,
        segmentation={"min_object_area": {"value": 4, "unit": "um"}},
    )
    with pytest.raises(BatchError, match="different analysis"):
        run_batch(changed)
    kept = (tmp_path / "out" / MANIFEST_FILE).read_text(encoding="utf-8")
    assert kept == first.model_dump_json(indent=2)

    manifest = run_batch(changed, overwrite=True)
    assert manifest.status == "complete"
    assert manifest.counts["resumed"] == 0


def test_failed_images_are_retried_on_resume(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    images = [three_images[0], _image(tmp_path / "data" / "later.tif", "later")]
    config = _config(tmp_path, images)
    assert run_batch(config).status == "completed_with_failures"
    _phantom(tmp_path / "data" / "later.tif", seed=7)
    manifest = run_batch(config)
    assert manifest.status == "complete"
    assert _items(manifest)["img_1"].resumed
    assert not _items(manifest)["later"].resumed


# --- incomplete runs are never mistaken for finished ones -------------------------------------


def test_run_that_raises_leaves_no_manifest_and_is_marked_aborted(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    config = _config(tmp_path, three_images)

    def crash(event: ProgressEvent) -> None:
        raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_batch(config, progress=crash)

    output = config.output_directory
    assert not (output / MANIFEST_FILE).exists()
    assert not (output / SUMMARY_FILE).exists()
    assert not (output / LOCK_FILE).exists()
    state = json.loads((output / RUN_FILE).read_text(encoding="utf-8"))
    assert state["state"] == "aborted"

    manifest = run_batch(config)
    assert manifest.status == "complete"
    assert manifest.counts["resumed"] == 1
    assert not (output / RUN_FILE).exists()


def test_cancel_file_stops_new_work(tmp_path: Path, three_images: list[dict[str, Any]]) -> None:
    config = _config(tmp_path, three_images)

    def request_cancel(event: ProgressEvent) -> None:
        (config.output_directory / CANCEL_FILE).write_text("stop", encoding="utf-8")

    manifest = run_batch(config, progress=request_cancel)
    assert manifest.status == "cancelled"
    assert [item.status for item in manifest.items] == ["ok", "cancelled", "cancelled"]


def test_stale_partial_files_are_removed_and_a_fake_success_is_not_trusted(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    config = _config(tmp_path, three_images[:1])
    assert run_batch(config).status == "complete"
    item_dir = _item_dir(config, "img_1")
    (item_dir / "labels.tif.partial").write_bytes(b"half written")
    fake = ItemRecord(name="img_1", status="ok", input_sha256="bogus")
    (item_dir / RESULT_FILE).write_text(fake.model_dump_json(), encoding="utf-8")

    manifest = run_batch(config)
    assert manifest.status == "complete"
    assert not _items(manifest)["img_1"].resumed
    assert not list(config.output_directory.rglob("*.partial"))


def test_live_lock_refuses_a_second_run(tmp_path: Path, three_images: list[dict[str, Any]]) -> None:
    config = _config(tmp_path, three_images[:1])
    config.output_directory.mkdir(parents=True)
    lock = config.output_directory / LOCK_FILE
    lock.write_text(str(os.getpid()), encoding="utf-8")
    with pytest.raises(BatchError, match="another batch run"):
        run_batch(config)
    assert lock.exists()


def test_stale_lock_from_a_dead_process_does_not_block_a_run(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    config = _config(tmp_path, three_images[:1])
    config.output_directory.mkdir(parents=True)
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    (config.output_directory / LOCK_FILE).write_text(str(process.pid), encoding="utf-8")
    assert run_batch(config).status == "complete"
    assert not (config.output_directory / LOCK_FILE).exists()


def test_worker_count_must_be_positive(tmp_path: Path, three_images: list[dict[str, Any]]) -> None:
    with pytest.raises(BatchError, match="at least 1"):
        run_batch(_config(tmp_path, three_images), workers=0)


# --- constrained memory -----------------------------------------------------------------------


def test_images_over_the_memory_budget_are_rejected_without_running(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    config = _config(tmp_path, three_images[:2], memory_budget_mb=0.1)
    manifest = run_batch(config)
    assert manifest.status == "completed_with_failures"
    assert manifest.peak_estimated_mb == 0
    for item in manifest.items:
        assert item.status == "failed"
        assert "exceeds memory budget" in (item.error or "")
        assert item.outputs == {}


def test_memory_budget_serialises_work_that_would_exceed_it(
    tmp_path: Path, three_images: list[dict[str, Any]]
) -> None:
    # Each 128 x 128 image is estimated at 0.625 MB. Two at once would need 1.25 MB, so a
    # 0.8 MB budget admits one at a time even though two workers were requested.
    config = _config(tmp_path, three_images, memory_budget_mb=0.8, workers=2)
    manifest = run_batch(config)
    assert manifest.status == "complete"
    assert manifest.workers == 2
    assert 0.6 < manifest.peak_estimated_mb <= 0.8
