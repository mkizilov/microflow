"""Ordered, recorded preprocessing of a loaded image.

Every run returns float64 data together with one :class:`ProcessingRecord` per stage. Records
hold the parameters as given and the derived quantities (for example pixel radii and the
intensity scale applied by normalisation), so any output can be traced back to its input.
Normalisation is affine and recorded; the raw image held by the caller is never modified.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from microflow.loading import LoadedImage
from microflow.plugins.api import PluginSpec
from microflow.plugins.contracts import call, check_image
from microflow.plugins.registry import configure
from microflow.preprocessing import operations as ops
from microflow.preprocessing.operations import PreprocessingError
from microflow.preprocessing.steps import (
    GaussianFilter,
    MaskArtifacts,
    MedianFilter,
    Normalize,
    PluginStep,
    Step,
    SubtractBackground,
    SubtractConstant,
)
from microflow.units import Calibration, Length

_EXACT_INTEGER_LIMIT = 2**53


class ProcessingRecord(BaseModel):
    stage: str
    parameters: dict[str, Any]
    effect: dict[str, float | int]


@dataclass(frozen=True)
class PreprocessResult:
    data: npt.NDArray[np.float64]
    axes: str
    source_dtype: str
    calibration: Calibration | None
    records: tuple[ProcessingRecord, ...]

    def provenance(self, source: str) -> dict[str, Any]:
        return {
            "source": source,
            "axes": self.axes,
            "source_dtype": self.source_dtype,
            "output_dtype": str(self.data.dtype),
            "calibration": self.calibration.model_dump() if self.calibration else None,
            "stages": [record.model_dump() for record in self.records],
        }


def preprocess(
    image: LoadedImage,
    steps: Sequence[Step] = (),
    *,
    flatfield: LoadedImage | None = None,
) -> PreprocessResult:
    """Run the optional flat-field correction, then ``steps`` in order, on a copy of ``image``."""
    _check_precision(image.data)
    data = np.array(image.data, dtype=np.float64)
    non_finite = ~np.isfinite(data)
    data[non_finite] = np.nan
    saturation = float(np.iinfo(image.data.dtype).max) if image.data.dtype.kind in "iu" else None
    records = [
        _record(
            "input",
            {"source_dtype": str(image.data.dtype)},
            {"nonfinite_pixels": int(non_finite.sum())},
        )
    ]

    if flatfield is not None:
        flat = _flatfield_plane(image, flatfield)
        data, flat_mean = ops.flat_field_correct(data, image.axes, flat)
        records.append(
            _record("flat_field", {"source": flatfield.source}, {"flat_mean": flat_mean})
        )

    for step in steps:
        data, record = _apply(step, data, image, saturation)
        records.append(record)

    invalid = int(np.isnan(data).sum())
    records.append(_record("output", {"dtype": "float64"}, {"invalid_pixels": invalid}))
    return PreprocessResult(
        data=data,
        axes=image.axes,
        source_dtype=str(image.data.dtype),
        calibration=image.calibration,
        records=tuple(records),
    )


def _apply(
    step: Step, data: npt.NDArray[np.float64], image: LoadedImage, saturation: float | None
) -> tuple[npt.NDArray[np.float64], ProcessingRecord]:
    stage = step.kind
    parameters = step.model_dump(mode="json", exclude={"kind"})
    if isinstance(step, SubtractConstant):
        return ops.subtract_constant(data, step.value), _record(stage, parameters, {})
    if isinstance(step, SubtractBackground):
        radius_px = _pixels(step.radius, image.calibration, stage)
        out = ops.subtract_morphological_background(data, image.axes, radius_px)
        return out, _record(stage, parameters, {"radius_px": radius_px})
    if isinstance(step, GaussianFilter):
        sigma_px = _pixels(step.sigma, image.calibration, stage)
        return ops.gaussian_filter(data, image.axes, sigma_px), _record(
            stage, parameters, {"sigma_px": sigma_px}
        )
    if isinstance(step, MedianFilter):
        radius_px = _pixels(step.radius, image.calibration, stage)
        return ops.median_filter(data, image.axes, radius_px), _record(
            stage, parameters, {"radius_px": radius_px}
        )
    if isinstance(step, MaskArtifacts):
        out, masked = ops.mask_artifacts(data, saturation if step.saturated else None, step.zero)
        return out, _record(stage, parameters, {"masked_pixels": masked})
    if isinstance(step, PluginStep):
        spec = PluginSpec(name=step.plugin, options=step.options)
        configured = configure("preprocessor", spec)
        label = f"preprocessor plugin {step.plugin!r}"
        output = call(
            "preprocessor",
            step.plugin,
            "process",
            configured.plugin.process,
            data.copy(),
            image.calibration,
            configured.options,
        )
        result = check_image(output, data.shape, label)
        return result, _record(
            f"plugin:{step.plugin}",
            {
                "plugin": step.plugin,
                "version": configured.version,
                "options": configured.options.model_dump(mode="json"),
            },
            {},
        )
    if not isinstance(step, Normalize):
        raise PreprocessingError(f"unknown preprocessing step {stage!r}")
    out, scale, offset, low, high = ops.normalize_percentile(
        data, step.lower_percentile, step.upper_percentile, step.target_min, step.target_max
    )
    return out, _record(
        stage,
        parameters,
        {"scale": scale, "offset": offset, "lower_value": low, "upper_value": high},
    )


def _pixels(length: Length, calibration: Calibration | None, stage: str) -> float:
    if calibration is None:
        raise PreprocessingError(
            f"{stage}: a length ({length.value} {length.unit}) needs pixel calibration; "
            "the image has none, so give its pixel size and unit"
        )
    return length.to(calibration.unit).value / calibration.pixel_size


def _flatfield_plane(image: LoadedImage, flatfield: LoadedImage) -> npt.NDArray[np.float64]:
    if flatfield.axes != "YX":
        raise PreprocessingError(
            f"flat-field must be a single YX plane, got axes {flatfield.axes!r}"
        )
    expected = (image.axis_size("Y"), image.axis_size("X"))
    if flatfield.shape != expected:
        raise PreprocessingError(
            f"flat-field shape {flatfield.shape} does not match the image plane {expected}"
        )
    return np.asarray(flatfield.data, dtype=np.float64)


def _check_precision(data: npt.NDArray[Any]) -> None:
    if data.dtype.kind in "iu" and data.size:
        largest = max(abs(int(data.max())), abs(int(data.min())))
        if largest > _EXACT_INTEGER_LIMIT:
            raise PreprocessingError(
                f"integer values up to {largest} cannot be represented exactly as float64"
            )


def _record(
    stage: str, parameters: dict[str, Any], effect: dict[str, float | int]
) -> ProcessingRecord:
    return ProcessingRecord(stage=stage, parameters=parameters, effect=effect)
