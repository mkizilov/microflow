from pathlib import Path
from typing import Any

import numpy as np

from microflow.batch import BatchConfig, analysis_id, inspect_cache, run_batch
from microflow.batch.engine import ANALYSIS_SECTIONS
from microflow.batch.store import NO_PREVIOUS
from microflow.image_io import write_image
from microflow.provenance.compare import ABSENT, diff_configs, flatten
from microflow.sample_data import Disc, make_phantom

DISCS = [Disc(centre=(30, 30), radius=8), Disc(centre=(30, 95), radius=6)]


def _phantom(path: Path, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, _ = make_phantom(DISCS, seed=seed)
    write_image(path, image.astype(np.uint16))
    return path


def _config(root: Path, images: list[dict[str, Any]], **overrides: Any) -> BatchConfig:
    values: dict[str, Any] = {
        "name": "compare-batch",
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


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


# --- configuration differences ------------------------------------------------------------------


def test_flatten_gives_a_dotted_path_for_every_leaf() -> None:
    assert flatten({"a": {"b": [1, {"c": 2}]}}) == {"a.b[0]": 1, "a.b[1].c": 2}


def test_flatten_keeps_empty_containers() -> None:
    assert flatten({"a": [], "b": {}}) == {"a": [], "b": {}}


def test_identical_settings_have_no_differences() -> None:
    settings = {"segmentation": {"min_object_area": {"value": 9.0, "unit": "um"}}}
    assert diff_configs(settings, dict(settings), analysis_sections=ANALYSIS_SECTIONS) == ()


def test_differences_are_listed_with_paths_and_marked_by_their_effect() -> None:
    old = {
        "segmentation": {"min_object_area": {"value": 9, "unit": "um"}},
        "workers": 1,
        "output_directory": "/one",
        "preprocessing": [],
    }
    new = {
        "segmentation": {"min_object_area": {"value": 4, "unit": "um"}},
        "workers": 3,
        "output_directory": "/two",
        "preprocessing": [{"kind": "gaussian"}],
    }
    changes = {
        change.path: change
        for change in diff_configs(old, new, analysis_sections=ANALYSIS_SECTIONS)
    }
    value = changes["segmentation.min_object_area.value"]
    assert (value.old, value.new, value.affects_analysis) == (9, 4, True)
    assert changes["workers"].affects_analysis is False
    assert changes["output_directory"].affects_analysis is False
    kind = changes["preprocessing[0].kind"]
    assert (kind.old, kind.new, kind.affects_analysis) == (ABSENT, "gaussian", True)


def test_configurations_built_from_the_same_settings_share_an_identity(tmp_path: Path) -> None:
    images = _images(tmp_path)
    first = _config(tmp_path / "x", images, workers=1)
    second = _config(tmp_path / "y", images, workers=2, memory_budget_mb=256)
    assert analysis_id(first) == analysis_id(second)
    differences = diff_configs(
        first.model_dump(mode="json"),
        second.model_dump(mode="json"),
        analysis_sections=ANALYSIS_SECTIONS,
    )
    assert differences
    assert not any(change.affects_analysis for change in differences)


# --- cache validity ----------------------------------------------------------------------------


def test_cache_status_reports_every_image_reusable_after_a_run(tmp_path: Path) -> None:
    config = _config(tmp_path, _images(tmp_path))
    run_batch(config)
    report = inspect_cache(config)
    assert report.analysis_matches is True
    assert report.directory_analysis == report.analysis_id == analysis_id(config)
    assert [(status.name, status.reusable, status.reason) for status in report.images] == [
        ("img_1", True, "valid"),
        ("img_2", True, "valid"),
    ]


def test_cache_status_names_the_image_whose_input_changed(tmp_path: Path) -> None:
    images = _images(tmp_path)
    config = _config(tmp_path, images)
    run_batch(config)
    _phantom(Path(images[1]["path"]), seed=9)
    statuses = {status.name: status for status in inspect_cache(config).images}
    assert statuses["img_1"].reusable is True
    assert statuses["img_2"].reusable is False
    assert statuses["img_2"].reason == "input changed"


def test_cache_status_reports_a_different_analysis_for_every_image(tmp_path: Path) -> None:
    images = _images(tmp_path)
    config = _config(tmp_path, images)
    run_batch(config)
    changed = _config(
        tmp_path,
        images,
        segmentation={"min_object_area": {"value": 4, "unit": "um"}},
    )
    report = inspect_cache(changed)
    assert report.analysis_matches is False
    assert report.directory_analysis == analysis_id(config)
    assert all(
        not status.reusable and status.reason == "configuration changed" for status in report.images
    )


def test_cache_status_for_a_directory_that_was_never_run(tmp_path: Path) -> None:
    config = _config(tmp_path / "fresh", _images(tmp_path))
    report = inspect_cache(config)
    assert report.directory_analysis is None
    assert report.analysis_matches is False
    assert all(status.reason == NO_PREVIOUS for status in report.images)


def test_cache_status_changes_nothing(tmp_path: Path) -> None:
    images = _images(tmp_path)
    config = _config(tmp_path, images)
    run_batch(config)
    before = _snapshot(config.output_directory)
    inspect_cache(config)
    changed = _config(
        tmp_path,
        images,
        workers=2,
        segmentation={"min_object_area": {"value": 4, "unit": "um"}},
    )
    inspect_cache(changed)
    assert _snapshot(config.output_directory) == before
