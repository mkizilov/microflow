from pathlib import Path

import pytest
from pydantic import ValidationError

from microflow.image_io import read_calibration, write_calibration
from microflow.units import Calibration


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
