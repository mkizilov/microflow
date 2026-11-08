import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from microflow.config import ConfigError
from microflow.ml.config import TrainingConfig, load_training_config


def test_minimal_configuration_gets_documented_defaults() -> None:
    config = TrainingConfig.model_validate({"manifest": "manifest.yaml"})
    assert config.seed == 0
    assert config.split.validation == 0.2
    assert config.split.test == 0.2
    assert config.device == "cpu"
    assert config.num_workers == 0
    assert config.architecture() == {"in_channels": 1, "base_channels": 8, "depth": 2}


def test_relative_paths_resolve_against_the_configuration_directory(tmp_path: Path) -> None:
    path = tmp_path / "configs" / "train.yaml"
    path.parent.mkdir()
    path.write_text("manifest: ../data/manifest.yaml\noutput_directory: runs\n", encoding="utf-8")
    config = load_training_config(path)
    assert config.manifest == path.parent / "../data/manifest.yaml"
    assert config.output_directory == path.parent / "runs"


def test_held_out_fractions_must_leave_room_for_training() -> None:
    with pytest.raises(ValidationError, match="must be less than 1"):
        TrainingConfig.model_validate({"manifest": "m", "split": {"validation": 0.6, "test": 0.4}})


def test_each_fraction_must_be_strictly_between_zero_and_one() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        TrainingConfig.model_validate({"manifest": "m", "split": {"validation": 0.0}})


def test_patch_size_must_be_a_multiple_of_the_downsampling_factor() -> None:
    with pytest.raises(ValidationError, match=r"multiple of 2\*\*model.depth \(8\)"):
        TrainingConfig.model_validate(
            {"manifest": "m", "model": {"depth": 3}, "patches": {"size": 36}}
        )


def test_unknown_keys_are_rejected_with_their_location(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text(
        textwrap.dedent(
            """
            manifest: manifest.yaml
            training:
              epochs: 2
              momentum: 0.9
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"training.momentum: Extra inputs are not permitted"):
        load_training_config(path)


def test_unknown_device_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text("manifest: manifest.yaml\ndevice: tpu\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="device"):
        load_training_config(path)


def test_augmentation_ranges_are_checked() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        TrainingConfig.model_validate({"manifest": "m", "augment": {"intensity": 2.0}})


def test_invalid_yaml_is_reported_with_its_file(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text("manifest: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_training_config(path)
