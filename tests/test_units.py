from pathlib import Path

import pytest
from pydantic import ValidationError

from microflow.image_io import read_calibration, write_calibration
from microflow.units import Area, Calibration, Length


def test_calibration_round_trips_through_json(tmp_path: Path) -> None:
    calibration = Calibration(pixel_size=0.25, unit="nm")
    path = tmp_path / "calibration.json"
    write_calibration(path, calibration)
    assert read_calibration(path) == calibration


@pytest.mark.parametrize("pixel_size", [0.0, -1.0])
def test_calibration_rejects_non_positive_pixel_size(pixel_size: float) -> None:
    with pytest.raises(ValidationError):
        Calibration(pixel_size=pixel_size, unit="um")


def test_calibration_requires_known_unit() -> None:
    with pytest.raises(ValidationError):
        Calibration(pixel_size=1.0, unit="pixels")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "from_unit", "to_unit", "expected"),
    [
        (0.5, "um", "nm", 500.0),
        (1500.0, "nm", "um", 1.5),
        (2.0, "mm", "um", 2000.0),
        (1.0, "m", "mm", 1000.0),
    ],
)
def test_length_conversion(value: float, from_unit: str, to_unit: str, expected: float) -> None:
    converted = Length(value=value, unit=from_unit).to(to_unit)  # type: ignore[arg-type]
    assert converted.unit == to_unit
    assert converted.value == pytest.approx(expected)


def test_area_conversion_squares_the_length_factor() -> None:
    converted = Area(value=4.0, unit="um").to("nm")
    assert converted.unit == "nm"
    assert converted.value == pytest.approx(4.0e6)


def test_length_rejects_negative_value() -> None:
    with pytest.raises(ValidationError):
        Length(value=-1.0, unit="um")
