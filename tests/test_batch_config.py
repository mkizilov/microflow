from pathlib import Path

import pytest

from microflow.batch import load_batch_config
from microflow.config import ConfigError


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "batch.yaml"
    path.write_text(text, encoding="utf-8")
    return path


MINIMAL = "dataset:\n  images:\n    - name: a\n      path: data/a.tif\n"


def test_relative_paths_resolve_against_the_config_file(tmp_path: Path) -> None:
    config = load_batch_config(_write(tmp_path, MINIMAL + "output_directory: out\n"))
    assert config.dataset.images[0].path == tmp_path / "data" / "a.tif"
    assert config.output_directory == tmp_path / "out"


def test_defaults_bound_parallelism_and_leave_memory_unlimited(tmp_path: Path) -> None:
    config = load_batch_config(_write(tmp_path, MINIMAL))
    assert config.name == "microflow-batch"
    assert config.workers == 2
    assert config.memory_budget_mb is None
    assert config.segmentation.method == "otsu"
    assert config.output_directory == tmp_path / "batch-results"


@pytest.mark.parametrize("workers", ["0", "9"])
def test_workers_are_bounded(tmp_path: Path, workers: str) -> None:
    with pytest.raises(ConfigError, match="workers"):
        load_batch_config(_write(tmp_path, MINIMAL + f"workers: {workers}\n"))


def test_memory_budget_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="memory_budget_mb"):
        load_batch_config(_write(tmp_path, MINIMAL + "memory_budget_mb: 0\n"))


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="worker"):
        load_batch_config(_write(tmp_path, MINIMAL + "worker: 4\n"))


def test_invalid_preprocessing_step_is_located(tmp_path: Path) -> None:
    text = MINIMAL + "preprocessing:\n  - kind: sharpen\n"
    with pytest.raises(ConfigError, match=r"preprocessing.0") as info:
        load_batch_config(_write(tmp_path, text))
    assert "sharpen" in str(info.value)


def test_empty_file_is_rejected_because_a_mapping_is_required(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_batch_config(_write(tmp_path, ""))


def test_missing_dataset_is_reported_by_field(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="dataset"):
        load_batch_config(_write(tmp_path, "name: nothing\n"))


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read file"):
        load_batch_config(tmp_path / "absent.yaml")


def test_segmentation_channel_is_accepted_by_name(tmp_path: Path) -> None:
    config = load_batch_config(_write(tmp_path, MINIMAL + "segmentation_channel: DAPI\n"))
    assert config.segmentation_channel == "DAPI"
