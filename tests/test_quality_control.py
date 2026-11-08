import numpy as np
import pytest

from microflow.config import QualityControlConfig, SegmentationConfig
from microflow.loading import LoadedImage, load_array
from microflow.reporting.qc import QualityReport, build_quality_report, inputs_from_images
from microflow.sample_data import Disc, make_phantom
from microflow.segmentation import segment_plane
from microflow.units import Area, Calibration

PIXEL = Calibration(pixel_size=0.5, unit="um")
DISCS = [
    Disc(centre=(30, 30), radius=8),
    Disc(centre=(30, 95), radius=6),
    Disc(centre=(90, 40), radius=10),
]


def _report(
    raw: np.ndarray,
    labels: np.ndarray,
    settings: QualityControlConfig | None = None,
    *,
    axes: str = "YX",
    calibration: Calibration | None = None,
    preprocessed: np.ndarray | None = None,
) -> QualityReport:
    raw_image = load_array(raw, axes=axes, calibration=calibration)
    label_image = load_array(labels, axes=axes)
    pre_image: LoadedImage | None = (
        load_array(preprocessed, axes=axes) if preprocessed is not None else None
    )
    inputs = inputs_from_images(
        raw_image, label_image, settings or QualityControlConfig(), preprocessed=pre_image
    )
    return build_quality_report(inputs)


def _codes(report: QualityReport) -> list[str]:
    return [flag.code for flag in report.flags]


def _square_labels(shape: tuple[int, int] = (100, 100)) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    labels[20:30, 40:50] = 1
    return labels


def test_clean_phantom_has_no_flags() -> None:
    image, _ = make_phantom(DISCS, seed=1)
    labels, _ = segment_plane(image, SegmentationConfig(min_object_area=Area(9, "um")), PIXEL)
    report = _report(image, labels, calibration=PIXEL)
    assert report.flags == ()
    assert report.planes[0].object_count == 3


def test_summary_statistics_are_exact_for_a_known_square() -> None:
    report = _report(np.full((100, 100), 10.0), _square_labels(), calibration=PIXEL)
    plane = report.planes[0]
    assert plane.total_pixels == 10_000
    assert plane.foreground_pixels == 100
    assert plane.foreground_fraction == pytest.approx(0.01)
    assert plane.object_count == 1
    assert plane.area_median == pytest.approx(100 * 0.25)
    assert plane.area_min == plane.area_max == plane.area_mean == pytest.approx(25.0)
    assert report.area_unit == "um2"
    assert report.length_unit == "um"


def test_uncalibrated_summary_reports_pixel_areas() -> None:
    report = _report(np.full((100, 100), 10.0), _square_labels())
    assert report.area_unit == "px2"
    assert report.planes[0].area_median == pytest.approx(100.0)


def test_empty_mask_is_flagged() -> None:
    report = _report(np.ones((20, 20)), np.zeros((20, 20), dtype=np.int32))
    assert _codes(report) == ["empty_mask"]
    assert report.planes[0].area_median is None
    assert report.planes[0].object_count == 0


def test_excessive_foreground_coverage_is_flagged_with_both_values() -> None:
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[:, :40] = 1
    report = _report(np.ones((50, 50)), labels)
    flag = next(f for f in report.flags if f.code == "excessive_coverage")
    assert "80.0%" in flag.message
    assert "60.0%" in flag.message


def test_coverage_at_the_limit_is_not_flagged() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[:6, :] = 1
    report = _report(np.ones((10, 10)), labels)
    assert "excessive_coverage" not in _codes(report)


def test_saturation_is_flagged_for_integer_images_at_their_maximum() -> None:
    raw = np.full((40, 40), 1000, dtype=np.uint16)
    raw[:4, :] = 65535  # 160 of 1600 pixels: 10%
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[10:20, 10:20] = 1
    report = _report(raw, labels)
    assert "saturation" in _codes(report)
    assert report.planes[0].saturated_fraction == pytest.approx(0.1)


def test_float_images_are_not_saturation_checked_unless_a_level_is_configured() -> None:
    raw = np.full((40, 40), 65535.0)
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[10:20, 10:20] = 1
    unconfigured = _report(raw, labels)
    assert unconfigured.planes[0].saturated_fraction is None
    assert "saturation" not in _codes(unconfigured)
    configured = _report(raw, labels, QualityControlConfig(saturation_level=65535.0))
    assert "saturation" in _codes(configured)


def test_invalid_pixels_above_the_limit_are_flagged() -> None:
    preprocessed = np.ones((100, 100))
    preprocessed[:2, :] = np.nan  # 2%
    report = _report(np.ones((100, 100)), _square_labels(), preprocessed=preprocessed)
    assert "invalid_pixels" in _codes(report)
    assert report.planes[0].invalid_fraction == pytest.approx(0.02)


def test_invalid_pixels_at_the_limit_are_not_flagged() -> None:
    preprocessed = np.ones((100, 100))
    preprocessed[:1, :] = np.inf  # exactly 1%
    report = _report(np.ones((100, 100)), _square_labels(), preprocessed=preprocessed)
    assert "invalid_pixels" not in _codes(report)


def test_objects_outside_the_configured_area_range_are_flagged_with_their_labels() -> None:
    labels = np.zeros((60, 60), dtype=np.int32)
    labels[5:9, 5:9] = 1  # 16 px = 4 um2
    labels[20:40, 20:40] = 2  # 400 px = 100 um2
    settings = QualityControlConfig(
        min_object_area=Area(value=5, unit="um"),
        max_object_area=Area(value=50, unit="um"),
    )
    report = _report(np.ones((60, 60)), labels, settings, calibration=PIXEL)
    below = next(f for f in report.flags if f.code == "object_area_below")
    above = next(f for f in report.flags if f.code == "object_area_above")
    assert "labels 1" in below.message and "5 um2" in below.message
    assert "labels 2" in above.message and "50 um2" in above.message
    assert report.area_min_limit == 5 and report.area_max_limit == 50


def test_objects_within_the_area_range_are_not_flagged() -> None:
    labels = np.zeros((60, 60), dtype=np.int32)
    labels[5:15, 5:15] = 1  # 100 px = 25 um2
    settings = QualityControlConfig(
        min_object_area=Area(value=5, unit="um"),
        max_object_area=Area(value=50, unit="um"),
    )
    report = _report(np.ones((60, 60)), labels, settings, calibration=PIXEL)
    assert report.flags == ()


def test_area_limits_need_calibration() -> None:
    settings = QualityControlConfig(min_object_area=Area(value=5, unit="um"))
    with pytest.raises(ValueError, match="needs pixel calibration"):
        _report(np.ones((10, 10)), np.zeros((10, 10), dtype=np.int32), settings)


def test_each_plane_is_summarised_and_flagged_separately() -> None:
    raw = np.ones((2, 40, 40))
    labels = np.zeros((2, 40, 40), dtype=np.int32)
    labels[0, 10:20, 10:20] = 1
    report = _report(raw, labels, axes="CYX")
    assert [plane.plane for plane in report.planes] == ["C0", "C1"]
    assert [flag.plane for flag in report.flags if flag.code == "empty_mask"] == ["C1"]


def test_figure_plane_is_the_plane_with_most_objects() -> None:
    raw = np.ones((2, 40, 40))
    labels = np.zeros((2, 40, 40), dtype=np.int32)
    labels[0, 2:4, 2:4] = 1
    labels[1, 2:4, 2:4] = 1
    labels[1, 10:12, 10:12] = 2
    report = _report(raw, labels, axes="CYX")
    assert report.figure_plane == "C1"


def test_labels_must_share_the_raw_image_grid() -> None:
    raw_image = load_array(np.ones((10, 10)), axes="YX")
    labels = load_array(np.zeros((12, 12), dtype=np.int32), axes="YX")
    with pytest.raises(ValueError, match="share"):
        inputs_from_images(raw_image, labels, QualityControlConfig())


def test_preprocessed_image_must_share_the_raw_grid() -> None:
    raw_image = load_array(np.ones((10, 10)), axes="YX")
    labels = load_array(np.zeros((10, 10), dtype=np.int32), axes="YX")
    preprocessed = load_array(np.ones((8, 8)), axes="YX")
    with pytest.raises(ValueError, match="preprocessed image must share"):
        inputs_from_images(raw_image, labels, QualityControlConfig(), preprocessed=preprocessed)
