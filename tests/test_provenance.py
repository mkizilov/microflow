import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from microflow import __version__
from microflow.batch import BatchConfig, BatchManifest, analysis_id, run_batch
from microflow.batch.store import ANALYSIS_FILE
from microflow.image_io import write_image
from microflow.provenance import versions
from microflow.provenance.hashing import digest, sha256_file
from microflow.provenance.record import (
    OUTPUT_SCHEMA_VERSION,
    InputFile,
    analysis_identity,
    make_record,
    result_set_identity,
)
from microflow.provenance.versions import (
    ALGORITHM_VERSIONS,
    REPORTED_LIBRARIES,
    library_versions,
    runtime_info,
)
from microflow.sample_data import Disc, make_phantom

DISCS = [Disc(centre=(30, 30), radius=8), Disc(centre=(30, 95), radius=6)]
RUNTIME_KEYS = {"python", "implementation", "platform", "machine"}


def _phantom(path: Path, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, _ = make_phantom(DISCS, seed=seed)
    write_image(path, image.astype(np.uint16))
    return path


def _config(root: Path, images: list[dict[str, Any]], **overrides: Any) -> BatchConfig:
    values: dict[str, Any] = {
        "name": "provenance-batch",
        "dataset": {"images": images},
        "segmentation": {"min_object_area": {"value": 9, "unit": "um"}},
        "output_directory": root / "out",
        "workers": 1,
    }
    values.update(overrides)
    return BatchConfig.model_validate(values)


def _image(path: Path, name: str) -> dict[str, Any]:
    return {"name": name, "path": path, "calibration": {"pixel_size": 1.0, "unit": "um"}}


@pytest.fixture
def batch_run(tmp_path: Path) -> tuple[BatchConfig, BatchManifest, list[Path]]:
    paths = [_phantom(tmp_path / "data" / f"img_{seed}.tif", seed) for seed in (1, 2)]
    images = [_image(paths[0], "img_1"), _image(paths[1], "img_2")]
    images.append(_image(tmp_path / "data" / "missing.tif", "missing"))
    config = _config(tmp_path, images)
    return config, run_batch(config), paths


# --- hashing ------------------------------------------------------------------------------------


def test_digest_matches_a_pinned_value() -> None:
    expected = "efbd0040190fb0871831e606c581f8a66db79d8e2bb836745a70051306956070"
    assert digest({"a": 1, "b": [2, 3]}) == expected


def test_digest_ignores_key_order_and_separates_different_content() -> None:
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    assert digest({"a": 1}) != digest({"a": 2})


def test_file_hash_is_the_sha256_of_the_bytes(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"microflow")
    assert sha256_file(path) == hashlib.sha256(b"microflow").hexdigest()


# --- versions and runtime -----------------------------------------------------------------------


def test_library_versions_report_every_reported_library() -> None:
    reported = library_versions()
    assert set(reported) == set(REPORTED_LIBRARIES)
    assert all(isinstance(value, str) and value for value in reported.values())


def test_a_missing_library_is_reported_rather_than_raised() -> None:
    assert library_versions(("microflow-no-such-package",)) == {
        "microflow-no-such-package": "not installed"
    }


def test_runtime_info_describes_the_interpreter_without_host_identity() -> None:
    info = runtime_info()
    assert set(info) == RUNTIME_KEYS
    assert info["python"].count(".") >= 1


# --- identities ---------------------------------------------------------------------------------


def test_analysis_identity_depends_on_settings_and_algorithm_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = analysis_identity({"segmentation": {"method": "otsu"}})
    assert analysis_identity({"segmentation": {"method": "otsu"}}) == base
    assert analysis_identity({"segmentation": {"method": "adaptive"}}) != base
    monkeypatch.setitem(versions.ALGORITHM_VERSIONS, "segmentation", "99")
    assert analysis_identity({"segmentation": {"method": "otsu"}}) != base


def test_result_set_identity_depends_on_the_input_hashes() -> None:
    first = InputFile(name="x", path="x.tif", sha256="1", size_bytes=1)
    second = InputFile(name="x", path="x.tif", sha256="2", size_bytes=1)
    assert result_set_identity("id", [first]) == result_set_identity("id", [first])
    assert result_set_identity("id", [first]) != result_set_identity("id", [second])


def test_record_carries_every_provenance_field() -> None:
    record = make_record(
        analysis_id="abc",
        resolved_config={"name": "demo"},
        inputs=[],
        warnings=["something to check"],
        outputs=[],
        started_utc="2026-01-01T00:00:00+00:00",
        finished_utc="2026-01-01T00:00:01+00:00",
    )
    assert record.schema_version == OUTPUT_SCHEMA_VERSION
    assert record.microflow_version == __version__
    assert record.analysis_id == "abc"
    assert record.resolved_config == {"name": "demo"}
    assert record.algorithm_versions == ALGORITHM_VERSIONS
    assert set(record.library_versions) == set(REPORTED_LIBRARIES)
    assert set(record.runtime) == RUNTIME_KEYS
    assert record.warnings == ("something to check",)
    assert record.inputs == () and record.outputs == ()


# --- provenance of a real run -------------------------------------------------------------------


def test_manifest_provenance_is_complete_and_matches_the_inputs(
    batch_run: tuple[BatchConfig, BatchManifest, list[Path]],
) -> None:
    config, manifest, _paths = batch_run
    record = manifest.provenance

    assert record.microflow_version == __version__
    assert record.schema_version == OUTPUT_SCHEMA_VERSION
    assert record.analysis_id == manifest.config_fingerprint == analysis_id(config)
    assert record.resolved_config == config.model_dump(mode="json")
    assert record.algorithm_versions == ALGORITHM_VERSIONS
    assert set(record.library_versions) == set(REPORTED_LIBRARIES)
    assert set(record.runtime) == RUNTIME_KEYS
    started = datetime.fromisoformat(record.started_utc)
    finished = datetime.fromisoformat(record.finished_utc)
    assert started <= finished


def test_manifest_records_each_input_hash_and_size(
    batch_run: tuple[BatchConfig, BatchManifest, list[Path]],
) -> None:
    _, manifest, paths = batch_run
    inputs = {item.name: item for item in manifest.provenance.inputs}
    assert set(inputs) == {"img_1", "img_2"}  # the missing file has no hash to record
    for name, path in zip(("img_1", "img_2"), paths, strict=True):
        assert inputs[name].sha256 == sha256_file(path)
        assert inputs[name].size_bytes == path.stat().st_size
    assert manifest.provenance.result_set_id == result_set_identity(
        manifest.provenance.analysis_id, manifest.provenance.inputs
    )


def test_manifest_warnings_record_failures_and_the_run_status(
    batch_run: tuple[BatchConfig, BatchManifest, list[Path]],
) -> None:
    _, manifest, _ = batch_run
    assert manifest.status == "completed_with_failures"
    assert any(text.startswith("missing: failed:") for text in manifest.provenance.warnings)
    assert "run status: completed_with_failures" in manifest.provenance.warnings


def test_run_directory_records_the_analysis_it_belongs_to(
    batch_run: tuple[BatchConfig, BatchManifest, list[Path]],
) -> None:
    config, manifest, _ = batch_run
    recorded = json.loads((config.output_directory / ANALYSIS_FILE).read_text(encoding="utf-8"))
    assert recorded["analysis_id"] == manifest.provenance.analysis_id
    assert recorded["microflow_version"] == __version__


def test_identity_ignores_workers_memory_budget_and_dataset_membership(tmp_path: Path) -> None:
    paths = [_phantom(tmp_path / "data" / f"img_{seed}.tif", seed) for seed in (1, 2)]
    images = [_image(path, f"img_{seed}") for seed, path in zip((1, 2), paths, strict=True)]
    single = _config(tmp_path, images[:1], workers=1)
    many = _config(tmp_path, images, workers=2, memory_budget_mb=512)
    assert analysis_id(single) == analysis_id(many)

    other = _config(
        tmp_path,
        images[:1],
        segmentation={"min_object_area": {"value": 4, "unit": "um"}},
    )
    assert analysis_id(other) != analysis_id(single)


def test_changing_an_algorithm_version_changes_the_batch_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, [_image(_phantom(tmp_path / "d.tif", 1), "a")])
    before = analysis_id(config)
    monkeypatch.setitem(versions.ALGORITHM_VERSIONS, "quantification", "2")
    assert analysis_id(config) != before
