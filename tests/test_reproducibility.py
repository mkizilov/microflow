from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile

from microflow.batch import BatchConfig, BatchError, BatchManifest, analysis_id, run_batch
from microflow.batch.store import ITEMS_DIR, item_directory_name
from microflow.image_io import write_image
from microflow.sample_data import Disc, make_phantom

DISCS = [Disc(centre=(30, 30), radius=8), Disc(centre=(30, 95), radius=6)]


def _phantom(path: Path, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, _ = make_phantom(DISCS, seed=seed)
    write_image(path, image.astype(np.uint16))
    return path


def _config(root: Path, images: list[dict[str, Any]], **overrides: Any) -> BatchConfig:
    values: dict[str, Any] = {
        "name": "reproducible-batch",
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


def _comparable(manifest: BatchManifest) -> dict[str, Any]:
    """The parts of a run that must not depend on when or how it was executed."""
    return {
        "analysis_id": manifest.provenance.analysis_id,
        "result_set_id": manifest.provenance.result_set_id,
        "status": manifest.status,
        "items": [
            (
                item.name,
                item.status,
                item.objects,
                item.flags,
                item.warnings,
                item.input_sha256,
                item.config_fingerprint,
            )
            for item in manifest.items
        ],
    }


def _assert_same_item_outputs(first: Path, second: Path, names: list[str]) -> None:
    for name in names:
        left = first / ITEMS_DIR / item_directory_name(name)
        right = second / ITEMS_DIR / item_directory_name(name)
        assert (left / "measurements.csv").read_bytes() == (right / "measurements.csv").read_bytes()
        np.testing.assert_array_equal(
            tifffile.imread(left / "labels.tif"), tifffile.imread(right / "labels.tif")
        )


def test_repeated_runs_of_one_configuration_produce_identical_results(tmp_path: Path) -> None:
    images = _images(tmp_path)
    first = run_batch(_config(tmp_path / "a", images))
    second = run_batch(_config(tmp_path / "b", images))
    assert first.status == second.status == "complete"
    assert _comparable(first) == _comparable(second)
    _assert_same_item_outputs(tmp_path / "a" / "out", tmp_path / "b" / "out", ["img_1", "img_2"])


def test_resumed_run_reproduces_the_outputs_of_an_uninterrupted_run(tmp_path: Path) -> None:
    images = _images(tmp_path)
    fresh = run_batch(_config(tmp_path / "fresh", images))

    events: list[object] = []
    interrupted = _config(tmp_path / "resumed", images)
    partial = run_batch(interrupted, progress=events.append, cancel=lambda: len(events) >= 1)
    assert partial.status == "cancelled"
    resumed = run_batch(interrupted)

    assert resumed.status == "complete"
    assert resumed.counts["resumed"] == 1
    assert _comparable(resumed) == _comparable(fresh)
    _assert_same_item_outputs(
        tmp_path / "fresh" / "out", tmp_path / "resumed" / "out", ["img_1", "img_2"]
    )


def test_a_different_configuration_is_refused_in_a_populated_directory(tmp_path: Path) -> None:
    images = _images(tmp_path)
    config = _config(tmp_path, images)
    original = run_batch(config)
    (tmp_path / "out" / "notes.txt").write_text("keep me", encoding="utf-8")

    changed = _config(
        tmp_path,
        images,
        segmentation={"min_object_area": {"value": 4, "unit": "um"}},
    )
    with pytest.raises(BatchError, match="different analysis"):
        run_batch(changed)
    assert (tmp_path / "out" / "notes.txt").read_text(encoding="utf-8") == "keep me"
    assert _comparable(run_batch(config)) == _comparable(original)


def test_overwrite_replaces_managed_files_and_keeps_unrelated_files(tmp_path: Path) -> None:
    images = _images(tmp_path)
    config = _config(tmp_path, images)
    run_batch(config)
    (tmp_path / "out" / "notes.txt").write_text("keep me", encoding="utf-8")

    changed = _config(
        tmp_path,
        images,
        segmentation={"min_object_area": {"value": 4, "unit": "um"}},
    )
    manifest = run_batch(changed, overwrite=True)
    assert manifest.status == "complete"
    assert manifest.counts["resumed"] == 0
    assert manifest.provenance.analysis_id == analysis_id(changed)
    assert analysis_id(changed) != analysis_id(config)
    assert (tmp_path / "out" / "notes.txt").read_text(encoding="utf-8") == "keep me"
