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
