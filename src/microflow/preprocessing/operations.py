"""Array operations for preprocessing.

Every operation takes a float64 array in which NaN marks an invalid pixel, and returns a new array
with the same shape. Invalid pixels stay invalid, so filters never average them in and
normalisation never counts them. Spatial operations act on the Y and X axes only; other axes
(channels, time, depth) are processed independently.
"""

import warnings

import numpy as np
import numpy.typing as npt
from scipy import ndimage

Array = npt.NDArray[np.float64]

_WEIGHT_EPSILON = 1e-12


class PreprocessingError(ValueError):
    """A preprocessing request cannot be carried out faithfully."""


def _spatial_sigma(axes: str, sigma_px: float) -> tuple[float, ...]:
    return tuple(sigma_px if axis in "YX" else 0.0 for axis in axes)


def _window(axes: str, size: int) -> tuple[int, ...]:
    return tuple(size if axis in "YX" else 1 for axis in axes)


def _window_size(radius_px: float) -> int:
    return 2 * round(radius_px) + 1


def gaussian_filter(data: Array, axes: str, sigma_px: float) -> Array:
    """Gaussian smoothing by normalised convolution, so NaN pixels do not bias their neighbours."""
    valid = np.isfinite(data)
    numerator = ndimage.gaussian_filter(np.where(valid, data, 0.0), _spatial_sigma(axes, sigma_px))
    weight = ndimage.gaussian_filter(valid.astype(np.float64), _spatial_sigma(axes, sigma_px))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = numerator / weight
    out[(weight <= _WEIGHT_EPSILON) | ~valid] = np.nan
    return out


def median_filter(data: Array, axes: str, radius_px: float) -> Array:
    """Median filter over a square window. Windows that contain NaN use the NaN-aware median."""
    footprint = np.ones(_window(axes, _window_size(radius_px)), dtype=bool)
    valid = np.isfinite(data)
    if valid.all():
        return ndimage.median_filter(data, footprint=footprint, mode="reflect")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = ndimage.generic_filter(data, np.nanmedian, footprint=footprint, mode="reflect")
    out[~valid] = np.nan
    return out


def subtract_morphological_background(data: Array, axes: str, radius_px: float) -> Array:
    """Subtract a grey-level opening estimate of the background (square structuring element).

    Structures narrower than the window are treated as foreground and kept. Invalid pixels are
    filled with the largest valid value so they cannot lower the erosion step.
    """
    footprint = np.ones(_window(axes, _window_size(radius_px)), dtype=bool)
    valid = np.isfinite(data)
    fill = float(data[valid].max()) if valid.any() else 0.0
    background = ndimage.grey_opening(
        np.where(valid, data, fill), footprint=footprint, mode="reflect"
    )
    out = data - background
    out[~valid] = np.nan
    return out


def subtract_constant(data: Array, value: float) -> Array:
    return data - value


def flat_field_correct(data: Array, axes: str, flat: Array) -> tuple[Array, float]:
    """Divide by the flat-field gain (flat / mean(flat)) so the mean intensity is preserved.

    Returns the corrected array and the mean of the flat-field reference.
    """
    if not np.isfinite(flat).all():
        raise PreprocessingError("flat-field reference contains NaN or infinite values")
    if (flat <= 0).any():
        raise PreprocessingError(
            "flat-field reference must be strictly positive; zero or negative pixels cannot be "
            "divided out"
        )
    mean = float(flat.mean())
    gain = flat / mean
    if axes.index("Y") > axes.index("X"):
        gain = gain.T
    shape = tuple(
        size if axis in "YX" else 1
        for axis, size in zip(axes, data.shape, strict=True)
    )
    return data / gain.reshape(shape), mean


def mask_artifacts(data: Array, saturation: float | None, zero: bool) -> tuple[Array, int]:
    """Mark saturated (and optionally zero) pixels invalid; also returns the newly masked count."""
    bad = np.zeros(data.shape, dtype=bool)
    if saturation is not None:
        bad |= data >= saturation
    if zero:
        bad |= data == 0
    newly_masked = bad & np.isfinite(data)
    return np.where(bad, np.nan, data), int(newly_masked.sum())


def normalize_percentile(
    data: Array, lower: float, upper: float, target_min: float, target_max: float
) -> tuple[Array, float, float, float, float]:
    """Affine rescale so the lower and upper percentiles map to the target range.

    Returns the output, the scale and offset (output = scale * input + offset), and the input
    values at the two percentiles. Values outside the percentiles are not clipped.
    """
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        raise PreprocessingError("cannot normalise: the image has no finite pixels")
    low, high = (float(value) for value in np.percentile(valid, [lower, upper]))
    if high <= low:
        raise PreprocessingError(
            f"cannot normalise: the {lower}th and {upper}th percentiles are both {low}, "
            "so the image has no intensity range"
        )
    scale = (target_max - target_min) / (high - low)
    offset = target_min - low * scale
    return data * scale + offset, scale, offset, low, high
