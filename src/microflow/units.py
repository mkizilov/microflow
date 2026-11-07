"""Physical units attached to calibrated measurements."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveFloat

LengthUnit = Literal["nm", "um", "mm", "m"]


class Calibration(BaseModel):
    """Pixel-size calibration. The unit is required: quantities are never unitless."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pixel_size: PositiveFloat
    unit: LengthUnit
