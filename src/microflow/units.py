"""Physical units attached to calibrated measurements."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat

LengthUnit = Literal["nm", "um", "mm", "m"]

_NANOMETRES_PER_UNIT: dict[LengthUnit, float] = {"nm": 1.0, "um": 1e3, "mm": 1e6, "m": 1e9}


class Calibration(BaseModel):
    """Pixel-size calibration. The unit is required: quantities are never unitless."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pixel_size: PositiveFloat
    unit: LengthUnit


class Length(BaseModel):
    """A physical length, for example a smoothing sigma."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: NonNegativeFloat
    unit: LengthUnit

    def to(self, unit: LengthUnit) -> "Length":
        factor = _NANOMETRES_PER_UNIT[self.unit] / _NANOMETRES_PER_UNIT[unit]
        return Length(value=self.value * factor, unit=unit)


class Area(BaseModel):
    """A physical area, expressed as the square of a length unit (for example µm²)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: NonNegativeFloat
    unit: LengthUnit

    def to(self, unit: LengthUnit) -> "Area":
        factor = (_NANOMETRES_PER_UNIT[self.unit] / _NANOMETRES_PER_UNIT[unit]) ** 2
        return Area(value=self.value * factor, unit=unit)


def _require_calibration(calibration: Calibration | None, quantity: str) -> Calibration:
    if calibration is None:
        raise ValueError(
            f"{quantity} needs pixel calibration; the image has none, "
            "so give its pixel size and unit"
        )
    return calibration


def to_pixels(length: Length, calibration: Calibration | None) -> float:
    """Convert a physical length to a number of pixels using the image calibration.

    Zero is zero pixels in any unit, so it needs no calibration: a zero minimum area means no
    filter, and an uncalibrated image can still use it.
    """
    if length.value == 0:
        return 0.0
    calibration = _require_calibration(calibration, f"a length of {length.value} {length.unit}")
    return length.to(calibration.unit).value / calibration.pixel_size


def area_to_pixels(area: Area, calibration: Calibration | None) -> float:
    """Convert a physical area to a number of pixels using the image calibration.

    Zero needs no calibration, for the same reason as :func:`to_pixels`.
    """
    if area.value == 0:
        return 0.0
    calibration = _require_calibration(calibration, f"an area of {area.value} {area.unit}²")
    return area.to(calibration.unit).value / calibration.pixel_size**2
