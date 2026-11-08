from pathlib import Path

import pytest
from pydantic import ValidationError

from microflow.config import ConfigError, QualityControlConfig, load_quality_config
from microflow.units import Area


def test_quality_defaults() -> None:
    settings = QualityControlConfig()
    assert settings.max_foreground_fraction == 0.6
    assert settings.max_saturated_fraction == 0.001
    assert settings.saturation_level is None
    assert settings.max_invalid_fraction == 0.01
    assert settings.min_object_area is None
    assert settings.max_object_area is None


@pytest.mark.parametrize(
    "field", ["max_foreground_fraction", "max_saturated_fraction", "max_invalid_fraction"]
)
@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_fractions_must_lie_in_unit_interval(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match=field):
        QualityControlConfig.model_validate({field: value})


def test_area_range_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        QualityControlConfig(
            min_object_area=Area(value=50, unit="um"),
            max_object_area=Area(value=5, unit="um"),
        )


def test_area_range_is_compared_in_a_common_unit() -> None:
    # 5 um2 is 5e6 nm2, which exceeds 1000 nm2, so this range is inverted.
    with pytest.raises(ValidationError, match="must not exceed"):
        QualityControlConfig(
            min_object_area=Area(value=5, unit="um"),
            max_object_area=Area(value=1000, unit="nm"),
        )


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="max_foreground"):
        QualityControlConfig.model_validate({"max_foreground": 0.5})


def test_load_quality_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "qc.yaml"
    path.write_text(
        "max_foreground_fraction: 0.5\n"
        "saturation_level: 4095\n"
        "min_object_area: {value: 5, unit: um}\n",
        encoding="utf-8",
    )
    settings = load_quality_config(path)
    assert settings.max_foreground_fraction == 0.5
    assert settings.saturation_level == 4095
    assert settings.min_object_area == Area(value=5, unit="um")


def test_empty_quality_file_means_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_quality_config(path) == QualityControlConfig()


def test_quality_config_reports_field_location(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("max_invalid_fraction: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="max_invalid_fraction"):
        load_quality_config(path)
