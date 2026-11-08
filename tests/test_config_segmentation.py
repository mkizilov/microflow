from pathlib import Path

import pytest
from pydantic import ValidationError

from microflow.config import ConfigError, SegmentationConfig, load_segmentation_config
from microflow.units import Area, Length


def test_segmentation_defaults() -> None:
    settings = SegmentationConfig()
    assert settings.method == "otsu"
    assert settings.threshold is None
    assert settings.adaptive_block_size == Length(value=10.0, unit="um")
    assert settings.adaptive_offset == 0.0
    assert settings.opening_radius is None
    assert settings.closing_radius is None
    assert settings.watershed is False
    assert settings.watershed_min_distance == Length(value=2.0, unit="um")
    assert settings.min_object_area == Area(value=4.0, unit="um")
    assert settings.exclude_border is False


def test_adaptive_method_is_accepted() -> None:
    settings = SegmentationConfig.model_validate(
        {"method": "adaptive", "adaptive_block_size": {"value": 31, "unit": "um"}}
    )
    assert settings.method == "adaptive"


def test_threshold_is_rejected_for_adaptive() -> None:
    with pytest.raises(ValidationError, match="only used when method is 'threshold'"):
        SegmentationConfig.model_validate({"method": "adaptive", "threshold": 10})


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValidationError, match="method"):
        SegmentationConfig.model_validate({"method": "kmeans"})


@pytest.mark.parametrize("field", ["adaptive_block_size", "watershed_min_distance"])
def test_non_positive_lengths_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match="greater than zero"):
        SegmentationConfig.model_validate({field: {"value": 0, "unit": "um"}})


def test_negative_morphology_radius_is_rejected() -> None:
    with pytest.raises(ValidationError, match="opening_radius"):
        SegmentationConfig.model_validate({"opening_radius": {"value": -1, "unit": "um"}})


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="methodd"):
        SegmentationConfig.model_validate({"methodd": "otsu"})


def test_load_segmentation_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "seg.yaml"
    path.write_text(
        "method: adaptive\n"
        "adaptive_block_size: {value: 25, unit: um}\n"
        "opening_radius: {value: 0.5, unit: um}\n"
        "watershed: true\n"
        "exclude_border: true\n",
        encoding="utf-8",
    )
    settings = load_segmentation_config(path)
    assert settings.method == "adaptive"
    assert settings.opening_radius == Length(value=0.5, unit="um")
    assert settings.watershed is True
    assert settings.exclude_border is True


def test_empty_segmentation_file_means_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_segmentation_config(path) == SegmentationConfig()


def test_load_segmentation_config_reports_field_location(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("method: kmeans\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="method"):
        load_segmentation_config(path)


def test_load_segmentation_config_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read file"):
        load_segmentation_config(tmp_path / "absent.yaml")
