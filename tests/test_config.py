from pathlib import Path

import pytest
from pydantic import ValidationError

from microflow.config import ConfigError, MicroFlowConfig, load_config
from microflow.units import Area, Length

REPO_ROOT = Path(__file__).resolve().parents[1]

MINIMAL = {
    "dataset": {
        "images": [
            {
                "name": "img",
                "path": "img.tif",
                "calibration": {"pixel_size": 0.5, "unit": "um"},
            }
        ]
    }
}


def _config(**overrides: object) -> dict[str, object]:
    return {**MINIMAL, **overrides}


def test_defaults_apply_to_minimal_configuration() -> None:
    config = MicroFlowConfig.model_validate(MINIMAL)
    assert config.name == "microflow-run"
    assert config.processing.smoothing_sigma == Length(value=1.0, unit="um")
    assert config.processing.subtract_background is False
    assert config.segmentation.method == "otsu"
    assert config.segmentation.threshold is None
    assert config.segmentation.min_object_area == Area(value=4.0, unit="um")
    assert config.quantification.measures == ["area", "mean_intensity"]
    assert config.quantification.require_calibration is True
    assert config.output.formats == ["csv"]
    assert config.output.overwrite is False


def test_image_defaults_to_yx_axes_without_channels() -> None:
    config = MicroFlowConfig.model_validate(
        {
            "dataset": {"images": [{"name": "a", "path": "a.tif"}]},
            "quantification": {"require_calibration": False},
        }
    )
    image = config.dataset.images[0]
    assert image.axes == "YX"
    assert image.channels is None
    assert image.calibration is None


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="processing.smoothing_sgima"):
        MicroFlowConfig.model_validate(
            _config(processing={"smoothing_sgima": {"value": 1.0, "unit": "um"}})
        )


@pytest.mark.parametrize(
    ("axes", "message"),
    [
        ("", "must not be empty"),
        ("YXQ", "unknown axis letters"),
        ("YYX", "must not repeat"),
        ("CX", "must include Y and X"),
    ],
)
def test_invalid_axes_are_rejected(axes: str, message: str) -> None:
    images = [{"name": "a", "path": "a.tif", "axes": axes}]
    with pytest.raises(ValidationError, match=message):
        MicroFlowConfig.model_validate(
            {"dataset": {"images": images}, "quantification": {"require_calibration": False}}
        )


def test_duplicate_channel_names_are_rejected() -> None:
    images = [{"name": "a", "path": "a.tif", "axes": "CYX", "channels": ["DAPI", "DAPI"]}]
    with pytest.raises(ValidationError, match="must be unique"):
        MicroFlowConfig.model_validate({"dataset": {"images": images}})


def test_duplicate_image_names_are_rejected() -> None:
    images = [
        {"name": "a", "path": "a.tif", "calibration": {"pixel_size": 1, "unit": "um"}},
        {"name": "a", "path": "b.tif", "calibration": {"pixel_size": 1, "unit": "um"}},
    ]
    with pytest.raises(ValidationError, match="duplicate image names: a"):
        MicroFlowConfig.model_validate({"dataset": {"images": images}})


def test_empty_manifest_is_rejected() -> None:
    with pytest.raises(ValidationError, match="images"):
        MicroFlowConfig.model_validate({"dataset": {"images": []}})


@pytest.mark.parametrize("pixel_size", [0, -0.5])
def test_non_positive_pixel_size_is_rejected(pixel_size: float) -> None:
    calibration = {"pixel_size": pixel_size, "unit": "um"}
    images = [{"name": "a", "path": "a.tif", "calibration": calibration}]
    with pytest.raises(ValidationError, match="pixel_size"):
        MicroFlowConfig.model_validate({"dataset": {"images": images}})


def test_calibration_without_unit_is_rejected() -> None:
    images = [{"name": "a", "path": "a.tif", "calibration": {"pixel_size": 0.5}}]
    with pytest.raises(ValidationError, match="unit"):
        MicroFlowConfig.model_validate({"dataset": {"images": images}})


def test_unsupported_unit_is_rejected() -> None:
    images = [{"name": "a", "path": "a.tif", "calibration": {"pixel_size": 0.5, "unit": "pixels"}}]
    with pytest.raises(ValidationError, match="unit"):
        MicroFlowConfig.model_validate({"dataset": {"images": images}})


def test_negative_smoothing_sigma_is_rejected() -> None:
    with pytest.raises(ValidationError, match="smoothing_sigma"):
        MicroFlowConfig.model_validate(
            _config(processing={"smoothing_sigma": {"value": -1.0, "unit": "um"}})
        )


def test_threshold_method_requires_threshold() -> None:
    with pytest.raises(ValidationError, match="threshold is required"):
        MicroFlowConfig.model_validate(_config(segmentation={"method": "threshold"}))


def test_threshold_is_rejected_for_otsu() -> None:
    with pytest.raises(ValidationError, match="only used when method is 'threshold'"):
        MicroFlowConfig.model_validate(_config(segmentation={"method": "otsu", "threshold": 10}))


def test_missing_calibration_fails_when_quantifying() -> None:
    images = [{"name": "a", "path": "a.tif"}, {"name": "b", "path": "b.tif"}]
    with pytest.raises(ValidationError, match="missing for: a, b"):
        MicroFlowConfig.model_validate({"dataset": {"images": images}})


def test_missing_calibration_allowed_when_not_quantifying() -> None:
    images = [{"name": "a", "path": "a.tif"}]
    config = MicroFlowConfig.model_validate(
        {"dataset": {"images": images}, "quantification": {"require_calibration": False}}
    )
    assert config.dataset.images[0].calibration is None


def test_load_config_resolves_paths_relative_to_file(tmp_path: Path) -> None:
    config_file = tmp_path / "configs" / "run.yaml"
    config_file.parent.mkdir()
    config_file.write_text(
        "dataset:\n"
        "  images:\n"
        "    - name: a\n"
        "      path: ../data/a.tif\n"
        "      calibration: {pixel_size: 0.5, unit: um}\n"
        "output:\n"
        "  directory: ../results\n",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.dataset.images[0].path == (tmp_path / "configs" / "../data/a.tif")
    assert config.output.directory == (tmp_path / "configs" / "../results")


def test_load_config_reports_file_and_field_for_invalid_value(tmp_path: Path) -> None:
    config_file = tmp_path / "run.yaml"
    config_file.write_text(
        "dataset:\n  images:\n    - name: a\n      path: a.tif\n      axes: XYZQ\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as info:
        load_config(config_file)
    message = str(info.value)
    assert str(config_file) in message
    assert "dataset.images.0.axes" in message
    assert "unknown axis letters" in message


def test_load_config_reports_yaml_syntax_errors(tmp_path: Path) -> None:
    config_file = tmp_path / "broken.yaml"
    config_file.write_text("dataset: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(config_file)


def test_load_config_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_config(config_file)


def test_load_config_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read file"):
        load_config(tmp_path / "absent.yaml")


def test_example_configuration_is_valid() -> None:
    config = load_config(REPO_ROOT / "examples" / "synthetic.yaml")
    assert [image.name for image in config.dataset.images] == [
        "synthetic_00",
        "synthetic_01",
        "synthetic_02",
    ]
    expected = REPO_ROOT / "examples" / "../data/synthetic/synthetic_00.tif"
    assert config.dataset.images[0].path == expected
