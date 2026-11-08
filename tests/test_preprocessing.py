from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from microflow.config import ConfigError
from microflow.loading import LoadedImage, load_array, load_image
from microflow.preprocessing import PreprocessingError, preprocess
from microflow.preprocessing.operations import (
    flat_field_correct,
    gaussian_filter,
    mask_artifacts,
    median_filter,
    normalize_percentile,
    subtract_morphological_background,
)
from microflow.preprocessing.steps import (
    GaussianFilter,
    MaskArtifacts,
    MedianFilter,
    Normalize,
    Step,
    SubtractBackground,
    SubtractConstant,
    load_steps,
)
from microflow.units import Calibration, Length

UM_CALIBRATION = Calibration(pixel_size=1.0, unit="um")


def _image(
    data: np.ndarray, axes: str = "YX", calibration: Calibration | None = None
) -> LoadedImage:
    return load_array(data, axes=axes, calibration=calibration)


# --- dtype and precision handling -------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32, np.float64])
def test_identity_pipeline_preserves_values_and_outputs_float64(dtype: type[np.generic]) -> None:
    data = (np.arange(36).reshape(6, 6) % 7).astype(dtype)
    result = preprocess(_image(data))
    assert result.data.dtype == np.float64
    np.testing.assert_array_equal(result.data, data.astype(np.float64))
    assert result.source_dtype == str(np.dtype(dtype))


def test_full_range_integers_are_not_rescaled() -> None:
    data = np.array([[0, 1], [32767, 65535]], dtype=np.uint16)
    assert preprocess(_image(data)).data.tolist() == [[0.0, 1.0], [32767.0, 65535.0]]


def test_integers_beyond_exact_float_range_are_rejected() -> None:
    with pytest.raises(PreprocessingError, match="cannot be represented exactly"):
        preprocess(_image(np.array([[2**60]], dtype=np.int64)))


def test_nan_and_infinity_become_invalid_and_are_counted() -> None:
    data = np.array([[1.0, np.nan], [np.inf, -np.inf]], dtype=np.float32)
    result = preprocess(_image(data))
    assert np.isnan(result.data[0, 1]) and np.isnan(result.data[1, 0])
    assert np.isnan(result.data[1, 1])
    assert result.data[0, 0] == 1.0
    assert result.records[0].effect == {"nonfinite_pixels": 3}
    assert result.records[-1].effect == {"invalid_pixels": 3}


def test_zero_intensity_is_kept_unless_masked() -> None:
    data = np.array([[0, 5], [0, 9]], dtype=np.uint16)
    assert preprocess(_image(data)).data[0, 0] == 0.0
    masked = preprocess(_image(data), [MaskArtifacts(kind="mask_artifacts", zero=True)])
    assert np.isnan(masked.data[0, 0]) and masked.data[0, 1] == 5.0
    assert masked.records[1].effect == {"masked_pixels": 2}


def test_saturated_integer_pixels_are_masked_by_default() -> None:
    data = np.array([[255, 10], [254, 0]], dtype=np.uint8)
    result = preprocess(_image(data), [MaskArtifacts(kind="mask_artifacts")])
    assert np.isnan(result.data[0, 0])
    assert result.data[1, 0] == 254.0
    assert result.records[1].effect == {"masked_pixels": 1}


def test_saturation_masking_can_be_disabled() -> None:
    data = np.array([[255, 10]], dtype=np.uint8)
    step = MaskArtifacts(kind="mask_artifacts", saturated=False)
    assert preprocess(_image(data), [step]).data[0, 0] == 255.0


def test_float_images_have_no_saturation_level() -> None:
    data = np.array([[1e30, 2.0]], dtype=np.float32)
    result = preprocess(_image(data), [MaskArtifacts(kind="mask_artifacts")])
    assert result.data[0, 0] == np.float32(1e30)


# --- background subtraction -------------------------------------------------------------------


def test_morphological_background_removes_ramp_and_keeps_narrow_spot() -> None:
    columns = np.arange(40, dtype=np.float64)
    data = np.tile(10.0 * columns, (40, 1))
    data[19:22, 19:22] += 200.0
    residual = subtract_morphological_background(data, "YX", radius_px=3.0)
    interior = np.ones_like(residual, dtype=bool)
    interior[:10, :] = interior[-10:, :] = False
    interior[:, :10] = interior[:, -10:] = False
    interior[13:28, 13:28] = False
    np.testing.assert_allclose(residual[interior], 0.0, atol=1e-9)
    np.testing.assert_allclose(residual[20, 20], 200.0, rtol=1e-9)


def test_subtract_constant_can_go_negative_and_is_recorded() -> None:
    step = SubtractConstant(kind="subtract_constant", value=8.0)
    result = preprocess(_image(np.full((3, 3), 5.0)), [step])
    assert result.data.max() == -3.0
    assert result.records[1].parameters == {"value": 8.0}


def test_background_subtraction_converts_units_to_pixels() -> None:
    calibration = Calibration(pixel_size=0.5, unit="um")
    data = np.zeros((32, 32))
    step = SubtractBackground(kind="subtract_background", radius=Length(value=1000, unit="nm"))
    result = preprocess(_image(data, calibration=calibration), [step])
    assert result.records[1].effect == {"radius_px": pytest.approx(2.0)}


# --- filtering ----------------------------------------------------------------------------------


def test_gaussian_reduces_white_noise_by_theoretical_factor() -> None:
    sigma = 2.0
    rng = np.random.default_rng(12345)
    noise = rng.normal(0.0, 1.0, size=(256, 256))
    smoothed = gaussian_filter(noise, "YX", sigma)
    expected_std_ratio = 1.0 / np.sqrt(4.0 * np.pi * sigma**2)
    assert np.std(smoothed[10:-10, 10:-10]) / np.std(noise) == pytest.approx(
        expected_std_ratio, rel=0.10
    )


def test_gaussian_of_constant_image_is_unchanged() -> None:
    data = np.full((16, 16), 7.25)
    np.testing.assert_allclose(gaussian_filter(data, "YX", 1.5), data, rtol=1e-12)


def test_gaussian_does_not_bias_neighbours_of_nan_pixels() -> None:
    data = np.full((20, 20), 5.0)
    data[8:12, 8:12] = np.nan
    out = gaussian_filter(data, "YX", 1.0)
    assert np.isnan(out[9, 9])
    np.testing.assert_allclose(out[7, 7], 5.0, rtol=1e-12)
    np.testing.assert_allclose(out[13, 13], 5.0, rtol=1e-12)


def test_median_removes_isolated_spikes_exactly() -> None:
    data = np.full((12, 12), 10.0)
    data[3, 3] = 1000.0
    data[8, 6] = 0.0
    out = median_filter(data, "YX", radius_px=1.0)
    np.testing.assert_array_equal(out, np.full((12, 12), 10.0))


def test_median_nan_path_removes_spikes_and_keeps_nan() -> None:
    data = np.full((12, 12), 10.0)
    data[3, 3] = 1000.0
    data[7, 7] = np.nan
    out = median_filter(data, "YX", radius_px=1.0)
    assert out[3, 3] == 10.0
    assert np.isnan(out[7, 7])


def test_channels_are_filtered_independently() -> None:
    data = np.stack([np.full((10, 10), 10.0), np.full((10, 10), 500.0)])
    steps: list[Step] = [
        GaussianFilter(kind="gaussian", sigma=Length(value=1.0, unit="um")),
        MedianFilter(kind="median", radius=Length(value=1.0, unit="um")),
    ]
    result = preprocess(_image(data, axes="CYX", calibration=UM_CALIBRATION), steps)
    np.testing.assert_allclose(result.data[0], 10.0)
    np.testing.assert_allclose(result.data[1], 500.0)


# --- flat-field correction ----------------------------------------------------------------------


def test_flat_field_removes_gain_and_preserves_mean_intensity() -> None:
    columns = np.linspace(1.0, 1.5, 32)
    gain = np.tile(columns, (24, 1))
    scene = np.full((24, 32), 50.0)
    corrected, flat_mean = flat_field_correct(scene * gain, "YX", gain)
    np.testing.assert_allclose(corrected, 50.0 * gain.mean(), rtol=1e-12)
    assert flat_mean == pytest.approx(gain.mean())


def test_flat_field_is_applied_per_channel_plane() -> None:
    gain = np.linspace(1.0, 2.0, 16).reshape(4, 4)
    data = np.stack([gain * 3.0, gain * 9.0])
    flat = load_array(gain, axes="YX")
    result = preprocess(_image(data, axes="CYX"), flatfield=flat)
    np.testing.assert_allclose(result.data[0], result.data[0].mean(), rtol=1e-12)
    np.testing.assert_allclose(result.data[0] * 3, result.data[1], rtol=1e-12)


def test_flat_field_with_transposed_image_axes_is_applied_correctly() -> None:
    gain = np.linspace(1.0, 2.0, 12).reshape(3, 4)
    data = gain.T * 5.0
    result = preprocess(_image(data, axes="XY"), flatfield=load_array(gain, axes="YX"))
    np.testing.assert_allclose(result.data, np.full((4, 3), result.data.mean()), rtol=1e-12)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf])
def test_flat_field_rejects_zero_negative_and_non_finite_references(bad_value: float) -> None:
    flat = np.ones((4, 4))
    flat[1, 1] = bad_value
    with pytest.raises(PreprocessingError, match="flat-field"):
        flat_field_correct(np.ones((4, 4)), "YX", flat)


def test_flat_field_shape_must_match_image_plane() -> None:
    flat = load_array(np.ones((4, 5)), axes="YX")
    with pytest.raises(PreprocessingError, match="does not match the image plane"):
        preprocess(_image(np.ones((4, 4))), flatfield=flat)


def test_flat_field_must_be_a_single_plane() -> None:
    flat = load_array(np.ones((2, 4, 4)), axes="CYX")
    with pytest.raises(PreprocessingError, match="single YX plane"):
        preprocess(_image(np.ones((4, 4))), flatfield=flat)


# --- normalisation ------------------------------------------------------------------------------


def test_normalize_maps_percentiles_to_target_range_and_is_invertible() -> None:
    rng = np.random.default_rng(3)
    data = rng.uniform(100.0, 4000.0, size=(64, 64))
    out, scale, offset, low, high = normalize_percentile(data, 1.0, 99.0, 0.0, 1.0)
    np.testing.assert_allclose(np.percentile(out, [1.0, 99.0]), [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose((out - offset) / scale, data, rtol=1e-12)
    assert low == pytest.approx(np.percentile(data, 1.0))
    assert high == pytest.approx(np.percentile(data, 99.0))


def test_normalize_is_explicit_and_recorded_with_scale_and_offset() -> None:
    data = np.linspace(0.0, 100.0, 101).reshape(101, 1)
    step = Normalize(kind="normalize", lower_percentile=0, upper_percentile=100)
    result = preprocess(_image(data), [step])
    assert result.records[1].stage == "normalize"
    assert result.records[1].effect["scale"] == pytest.approx(0.01)
    assert result.records[1].effect["offset"] == pytest.approx(0.0)
    assert result.data[-1, 0] == pytest.approx(1.0)


def test_normalize_does_not_alter_input_image() -> None:
    data = np.linspace(0.0, 100.0, 101).reshape(101, 1)
    original = data.copy()
    preprocess(_image(data), [Normalize(kind="normalize")])
    np.testing.assert_array_equal(data, original)


def test_normalize_ignores_nan_pixels_when_computing_range() -> None:
    data = np.array([[0.0, 10.0, np.nan, 20.0]])
    out, *_ = normalize_percentile(data, 0.0, 100.0, 0.0, 1.0)
    assert out[0, 0] == 0.0 and out[0, 3] == 1.0 and np.isnan(out[0, 2])


def test_normalize_rejects_constant_image() -> None:
    with pytest.raises(PreprocessingError, match="no intensity range"):
        normalize_percentile(np.full((4, 4), 3.0), 1.0, 99.0, 0.0, 1.0)


def test_normalize_rejects_all_nan_image() -> None:
    with pytest.raises(PreprocessingError, match="no finite pixels"):
        normalize_percentile(np.full((4, 4), np.nan), 1.0, 99.0, 0.0, 1.0)


def test_mask_artifacts_helper_counts_only_newly_masked_pixels() -> None:
    data = np.array([255.0, np.nan, 0.0, 3.0])
    out, count = mask_artifacts(data, saturation=255.0, zero=True)
    assert count == 2
    assert np.isnan(out[:3]).all() and out[3] == 3.0


# --- units, calibration and provenance ----------------------------------------------------------


def test_length_based_step_without_calibration_is_rejected() -> None:
    step = GaussianFilter(kind="gaussian", sigma=Length(value=1.0, unit="um"))
    with pytest.raises(PreprocessingError, match="needs pixel calibration"):
        preprocess(_image(np.ones((8, 8))), [step])


def test_gaussian_sigma_converts_from_physical_units() -> None:
    calibration = Calibration(pixel_size=0.5, unit="um")
    step = GaussianFilter(kind="gaussian", sigma=Length(value=1.0, unit="um"))
    result = preprocess(_image(np.ones((8, 8)), calibration=calibration), [step])
    assert result.records[1].effect == {"sigma_px": pytest.approx(2.0)}


def test_sigma_in_nanometres_converts_to_pixels() -> None:
    calibration = Calibration(pixel_size=0.5, unit="um")
    step = GaussianFilter(kind="gaussian", sigma=Length(value=500, unit="nm"))
    result = preprocess(_image(np.ones((8, 8)), calibration=calibration), [step])
    assert result.records[1].effect == {"sigma_px": pytest.approx(1.0)}


def test_provenance_lists_stages_in_order_with_parameters() -> None:
    steps: list[Step] = [
        SubtractConstant(kind="subtract_constant", value=10.0),
        GaussianFilter(kind="gaussian", sigma=Length(value=1.0, unit="um")),
        Normalize(kind="normalize"),
    ]
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    result = preprocess(_image(data, calibration=UM_CALIBRATION), steps)
    provenance = result.provenance("input.tif")
    assert [stage["stage"] for stage in provenance["stages"]] == [
        "input",
        "subtract_constant",
        "gaussian",
        "normalize",
        "output",
    ]
    assert provenance["source_dtype"] == "uint16"
    assert provenance["output_dtype"] == "float64"
    assert provenance["stages"][2]["parameters"] == {"sigma": {"value": 1.0, "unit": "um"}}
    assert provenance["calibration"] == {"pixel_size": 1.0, "unit": "um"}


def test_stage_order_is_respected() -> None:
    # A flat image has no intensity range, so normalisation correctly refuses it. This image spans a
    # range, and subtracting first leaves a range to normalise.
    data = np.arange(64, dtype=np.float64).reshape(8, 8) + 10.0
    steps: list[Step] = [
        SubtractConstant(kind="subtract_constant", value=4.0),
        Normalize(kind="normalize"),
    ]
    subtract_then_normalize = preprocess(_image(data), steps)
    assert [record.stage for record in subtract_then_normalize.records][1:3] == [
        "subtract_constant",
        "normalize",
    ]


# --- steps file ---------------------------------------------------------------------------------


def _write_steps(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "steps.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_steps_file_parses_each_kind(tmp_path: Path) -> None:
    path = _write_steps(
        tmp_path,
        "steps:\n"
        "  - kind: subtract_constant\n    value: 3\n"
        "  - kind: subtract_background\n    radius: {value: 10, unit: um}\n"
        "  - kind: gaussian\n    sigma: {value: 1, unit: um}\n"
        "  - kind: median\n    radius: {value: 0.5, unit: um}\n"
        "  - kind: mask_artifacts\n    zero: true\n"
        "  - kind: normalize\n    lower_percentile: 1\n    upper_percentile: 99.5\n",
    )
    steps = load_steps(path)
    assert [step.kind for step in steps] == [
        "subtract_constant",
        "subtract_background",
        "gaussian",
        "median",
        "mask_artifacts",
        "normalize",
    ]
    assert isinstance(steps[1], SubtractBackground)
    assert isinstance(steps[4], MaskArtifacts) and steps[4].zero is True


def test_empty_steps_file_means_no_processing(tmp_path: Path) -> None:
    assert load_steps(_write_steps(tmp_path, "")) == []
    assert load_steps(_write_steps(tmp_path, "steps: []\n")) == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("steps:\n  - kind: sharpen\n", "Input tag 'sharpen'"),
        ("steps:\n  - kind: gaussian\n    sigma: {value: 0, unit: um}\n", "sigma must be greater"),
        (
            "steps:\n  - kind: normalize\n    lower_percentile: 99\n    upper_percentile: 1\n",
            "lower_percentile must be less",
        ),
        (
            "steps:\n  - kind: median\n    radius: {value: 1, unit: um}\n    size: 3\n",
            "Extra inputs",
        ),
        ("steps:\n  - kind: gaussian\n    sigma: {value: 1, unit: parsec}\n", "sigma.unit"),
        ("steps:\n  - kind: normalize\n    upper_percentile: 120\n", "upper_percentile"),
    ],
)
def test_invalid_steps_are_rejected_with_location(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_steps(_write_steps(tmp_path, text))


def test_steps_file_reports_yaml_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_steps(_write_steps(tmp_path, "steps: [unclosed\n"))


# --- end-to-end on files ------------------------------------------------------------------------


def test_preprocess_tiff_round_trip_keeps_calibration_and_channels(
    write_tiff: Callable[..., Path], stack_cyx: np.ndarray
) -> None:
    path = write_tiff(
        "stack.tif",
        stack_cyx,
        ome=True,
        metadata={
            "axes": "CYX",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeXUnit": "\u00b5m",
            "PhysicalSizeY": 0.5,
            "PhysicalSizeYUnit": "\u00b5m",
            "Channel": {"Name": ["DAPI", "GFP"]},
        },
    )
    loaded = load_image(path)
    result = preprocess(loaded, [GaussianFilter(kind="gaussian", sigma=Length(value=1, unit="um"))])
    assert result.axes == "CYX"
    assert result.calibration == Calibration(pixel_size=0.5, unit="um")
    assert result.data.shape == stack_cyx.shape
    assert result.records[1].effect == {"sigma_px": pytest.approx(2.0)}
