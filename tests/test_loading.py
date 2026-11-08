from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from microflow.image_io import read_shape_and_dtype
from microflow.loading import ImageLoadError, load_array, load_image, parse_ome_pixels
from microflow.units import Calibration

INTEGER_AND_FLOAT_DTYPES = [np.uint8, np.uint16, np.uint32, np.int16, np.float32, np.float64]


def _sample(dtype: type[np.generic]) -> np.ndarray:
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min, info.max, size=(6, 7), dtype=dtype, endpoint=True)
    return rng.normal(size=(6, 7)).astype(dtype)


@pytest.mark.parametrize("dtype", INTEGER_AND_FLOAT_DTYPES)
def test_tiff_preserves_dtype_and_values(
    write_tiff: Callable[..., Path], dtype: type[np.generic]
) -> None:
    data = _sample(dtype)
    loaded = load_image(write_tiff("sample.tif", data))
    assert loaded.dtype == data.dtype
    assert loaded.axes == "YX"
    np.testing.assert_array_equal(loaded.data, data)


def test_float_special_values_are_not_altered(write_tiff: Callable[..., Path]) -> None:
    data = np.array([[np.nan, np.inf], [-np.inf, 1e-30]], dtype=np.float32)
    loaded = load_image(write_tiff("special.tif", data))
    np.testing.assert_array_equal(loaded.data, data)


def test_full_integer_range_is_not_rescaled(write_tiff: Callable[..., Path]) -> None:
    data = np.array([[0, 1], [32767, 65535]], dtype=np.uint16)
    loaded = load_image(write_tiff("range.tif", data))
    assert loaded.data.tolist() == [[0, 1], [32767, 65535]]


def test_grayscale_tiff_has_no_calibration_or_channels(write_tiff: Callable[..., Path]) -> None:
    loaded = load_image(write_tiff("plane.tif", np.zeros((8, 8), dtype=np.uint8)))
    assert loaded.calibration is None
    assert loaded.channels is None
    assert loaded.shape == (8, 8)


def test_multichannel_tiff_axes_and_metadata_agree_with_header(
    write_tiff: Callable[..., Path], stack_cyx: np.ndarray
) -> None:
    path = write_tiff("stack.tif", stack_cyx, metadata={"axes": "CYX"})
    loaded = load_image(path)
    assert loaded.axes == "CYX"
    assert loaded.axis_size("C") == 2
    assert loaded.channels is None
    assert read_shape_and_dtype(path) == (loaded.shape, str(loaded.dtype))
    np.testing.assert_array_equal(loaded.data, stack_cyx)


def test_axis_order_is_preserved_for_higher_dimensional_stacks(
    write_tiff: Callable[..., Path],
) -> None:
    data = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(3, 2, 4, 5)
    loaded = load_image(write_tiff("zcyx.tif", data, metadata={"axes": "ZCYX"}))
    assert loaded.axes == "ZCYX"
    assert loaded.shape == (3, 2, 4, 5)
    assert loaded.axis_size("Z") == 3
    np.testing.assert_array_equal(loaded.data, data)


def test_requested_axes_must_match_the_file(
    write_tiff: Callable[..., Path], stack_cyx: np.ndarray
) -> None:
    path = write_tiff("stack.tif", stack_cyx, metadata={"axes": "CYX"})
    with pytest.raises(ImageLoadError, match="declares 'CYX'"):
        load_image(path, axes="YXC")
    assert load_image(path, axes="CYX").axes == "CYX"


def test_resolution_tags_provide_calibration(write_tiff: Callable[..., Path]) -> None:
    # 2000 pixels per centimetre is a pixel size of 5 micrometres.
    path = write_tiff(
        "resolution.tif",
        np.zeros((8, 8), dtype=np.uint8),
        resolution=(2000, 2000),
        resolutionunit="CENTIMETER",
    )
    calibration = load_image(path).calibration
    assert calibration is not None
    assert calibration.unit == "um"
    assert calibration.pixel_size == pytest.approx(5.0)


def test_tiff_without_resolution_unit_has_no_calibration(
    write_tiff: Callable[..., Path],
) -> None:
    path = write_tiff(
        "no-unit.tif",
        np.zeros((8, 8), dtype=np.uint8),
        resolution=(1, 1),
        resolutionunit="NONE",
    )
    assert load_image(path).calibration is None


def test_ome_tiff_preserves_calibration_channels_and_axes(
    write_tiff: Callable[..., Path], stack_cyx: np.ndarray
) -> None:
    path = write_tiff(
        "sample.ome.tif",
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
    assert loaded.axes == "CYX"
    assert loaded.calibration == Calibration(pixel_size=0.5, unit="um")
    assert loaded.channels == ("DAPI", "GFP")
    np.testing.assert_array_equal(loaded.data, stack_cyx)


def test_explicit_calibration_overrides_file_calibration(
    write_tiff: Callable[..., Path],
) -> None:
    path = write_tiff(
        "resolution.tif",
        np.zeros((8, 8), dtype=np.uint8),
        resolution=(2000, 2000),
        resolutionunit="CENTIMETER",
    )
    override = Calibration(pixel_size=0.25, unit="nm")
    assert load_image(path, calibration=override).calibration == override


def test_lazy_uncompressed_tiff_is_memory_mapped(
    write_tiff: Callable[..., Path], stack_cyx: np.ndarray
) -> None:
    loaded = load_image(write_tiff("plain.tif", stack_cyx, metadata={"axes": "CYX"}), lazy=True)
    assert loaded.storage == "memmap"
    np.testing.assert_array_equal(loaded.data, stack_cyx)


def test_lazy_compressed_tiff_falls_back_to_memory(
    write_tiff: Callable[..., Path], stack_cyx: np.ndarray
) -> None:
    path = write_tiff("compressed.tif", stack_cyx, metadata={"axes": "CYX"}, compression="zlib")
    loaded = load_image(path, lazy=True)
    assert loaded.storage == "memory"
    np.testing.assert_array_equal(loaded.data, stack_cyx)


def test_npy_requires_axis_labels(tmp_path: Path) -> None:
    path = tmp_path / "plane.npy"
    np.save(path, np.zeros((4, 4), dtype=np.uint16))
    with pytest.raises(ImageLoadError, match="no axis labels"):
        load_image(path)


def test_npy_round_trip_preserves_dtype(tmp_path: Path) -> None:
    data = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "float.npy"
    np.save(path, data)
    loaded = load_image(path, axes="YX")
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded.data, data)


def test_npy_lazy_is_memory_mapped(tmp_path: Path) -> None:
    path = tmp_path / "lazy.npy"
    np.save(path, np.arange(16, dtype=np.uint16).reshape(4, 4))
    loaded = load_image(path, axes="YX", lazy=True)
    assert loaded.storage == "memmap"


def test_npy_pickled_object_arrays_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "objects.npy"
    np.save(path, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ImageLoadError, match="cannot read NumPy file"):
        load_image(path, axes="X")


def test_array_loading_does_not_copy() -> None:
    array = np.zeros((4, 4), dtype=np.uint16)
    loaded = load_array(array, axes="YX")
    assert np.shares_memory(loaded.data, array)


def test_array_channel_names_must_match_channel_axis() -> None:
    with pytest.raises(ImageLoadError, match="3 channel name"):
        load_array(np.zeros((2, 4, 4), dtype=np.uint8), axes="CYX", channels=["a", "b", "c"])


def test_array_channel_names_must_be_unique() -> None:
    with pytest.raises(ImageLoadError, match="must be unique"):
        load_array(np.zeros((2, 4, 4), dtype=np.uint8), axes="CYX", channels=["a", "a"])


def test_array_axes_must_match_dimensions() -> None:
    with pytest.raises(ImageLoadError, match="describe 3 dimensions"):
        load_array(np.zeros((4, 4), dtype=np.uint8), axes="CYX")


@pytest.mark.parametrize("axes", ["ZYQ", "YXS", "YY", "CX", ""])
def test_array_axes_must_be_supported_labels(axes: str) -> None:
    with pytest.raises(ImageLoadError):
        load_array(np.zeros((2, 4, 4), dtype=np.uint8), axes=axes)


def test_array_rejects_bool_and_complex_dtypes() -> None:
    with pytest.raises(ImageLoadError, match="not an intensity type"):
        load_array(np.zeros((4, 4), dtype=bool), axes="YX")
    with pytest.raises(ImageLoadError, match="not an intensity type"):
        load_array(np.zeros((4, 4), dtype=np.complex64), axes="YX")


def test_rgb_tiff_is_rejected_with_hint(write_tiff: Callable[..., Path]) -> None:
    path = write_tiff("rgb.tif", np.zeros((8, 8, 3), dtype=np.uint8), photometric="rgb")
    with pytest.raises(ImageLoadError, match="sample axes are not supported"):
        load_image(path)


def test_non_tiff_bytes_with_tiff_suffix_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fake.tif"
    path.write_bytes(b"this is not a tiff file")
    with pytest.raises(ImageLoadError, match="cannot read TIFF"):
        load_image(path)


def test_truncated_tiff_is_rejected(write_tiff: Callable[..., Path]) -> None:
    path = write_tiff("truncated.tif", np.arange(4096, dtype=np.uint16).reshape(64, 64))
    size = path.stat().st_size
    with path.open("r+b") as handle:
        handle.truncate(size // 2)
    with pytest.raises(ImageLoadError):
        load_image(path)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ImageLoadError, match="cannot read TIFF"):
        load_image(tmp_path / "absent.tif")


def test_unsupported_suffix_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ImageLoadError, match="unsupported image format '.png'"):
        load_image(tmp_path / "picture.png")


def test_series_index_out_of_range_is_rejected(write_tiff: Callable[..., Path]) -> None:
    path = write_tiff("plane.tif", np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ImageLoadError, match="series 3 is out of range"):
        load_image(path, series=3)


def test_series_argument_is_rejected_for_npy(tmp_path: Path) -> None:
    path = tmp_path / "plane.npy"
    np.save(path, np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ImageLoadError, match="applies only to TIFF"):
        load_image(path, axes="YX", series=0)


OME_TEMPLATE = (
    '<?xml version="1.0"?>'
    '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
    '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" SizeC="2" '
    '{pixel_attributes}>'
    '<TiffData><UUID FileName="{filename}">urn:uuid:0</UUID></TiffData>'
    '<Channel ID="Channel:0:0" Name="DAPI"/>'
    '<Channel ID="Channel:0:1" Name="GFP"/>'
    "</Pixels></Image></OME>"
)


def _ome(pixel_attributes: str = "", filename: str = "img.ome.tif") -> str:
    return OME_TEMPLATE.format(pixel_attributes=pixel_attributes, filename=filename)


def test_ome_default_unit_is_micrometre() -> None:
    calibration, _ = parse_ome_pixels(
        _ome('PhysicalSizeX="0.25" PhysicalSizeY="0.25"'), 0, "img.ome.tif"
    )
    assert calibration == Calibration(pixel_size=0.25, unit="um")


@pytest.mark.parametrize(
    ("unit", "expected"),
    [("nm", "nm"), ("mm", "mm"), ("m", "m"), ("micron", "um"), ("\u03bcm", "um")],
)
def test_ome_units_are_mapped(unit: str, expected: str) -> None:
    attributes = (
        f'PhysicalSizeX="2" PhysicalSizeXUnit="{unit}" '
        f'PhysicalSizeY="2" PhysicalSizeYUnit="{unit}"'
    )
    calibration, _ = parse_ome_pixels(_ome(attributes), 0, "img.ome.tif")
    assert calibration == Calibration(pixel_size=2.0, unit=expected)  # type: ignore[arg-type]


def test_ome_channel_names_are_read() -> None:
    _, channels = parse_ome_pixels(_ome(), 0, "img.ome.tif")
    assert channels == ("DAPI", "GFP")


def test_ome_without_physical_size_has_no_calibration() -> None:
    calibration, _ = parse_ome_pixels(_ome(), 0, "img.ome.tif")
    assert calibration is None


def test_ome_anisotropic_pixels_are_rejected() -> None:
    with pytest.raises(ImageLoadError, match="anisotropic"):
        parse_ome_pixels(_ome('PhysicalSizeX="0.5" PhysicalSizeY="0.25"'), 0, "img.ome.tif")


def test_ome_unsupported_unit_is_rejected() -> None:
    attributes = 'PhysicalSizeX="1" PhysicalSizeXUnit="pixel" PhysicalSizeY="1"'
    with pytest.raises(ImageLoadError, match="unsupported physical unit 'pixel'"):
        parse_ome_pixels(_ome(attributes), 0, "img.ome.tif")


def test_ome_mixed_units_are_rejected() -> None:
    attributes = (
        'PhysicalSizeX="1" PhysicalSizeXUnit="um" PhysicalSizeY="1" PhysicalSizeYUnit="nm"'
    )
    with pytest.raises(ImageLoadError, match="different units"):
        parse_ome_pixels(_ome(attributes), 0, "img.ome.tif")


def test_ome_only_one_physical_size_is_rejected() -> None:
    with pytest.raises(ImageLoadError, match="must both be set"):
        parse_ome_pixels(_ome('PhysicalSizeX="1"'), 0, "img.ome.tif")


def test_ome_non_positive_pixel_size_is_rejected() -> None:
    with pytest.raises(ImageLoadError, match="invalid physical pixel size"):
        parse_ome_pixels(_ome('PhysicalSizeX="0" PhysicalSizeY="0"'), 0, "img.ome.tif")


def test_ome_multi_file_reference_is_rejected() -> None:
    with pytest.raises(ImageLoadError, match="multi-file OME-TIFF"):
        parse_ome_pixels(_ome(filename="other.ome.tif"), 0, "img.ome.tif")


def test_malformed_ome_xml_is_rejected() -> None:
    with pytest.raises(ImageLoadError, match="malformed OME-XML"):
        parse_ome_pixels("<OME><Image>", 0, "img.ome.tif")


def test_ome_series_without_pixels_is_rejected() -> None:
    with pytest.raises(ImageLoadError, match="no matching OME Pixels"):
        parse_ome_pixels(_ome(), 1, "img.ome.tif")
