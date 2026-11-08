import csv
from pathlib import Path

import numpy as np
import pytest

from microflow.quantification import IntensitySource, MeasurementError, measure_plane
from microflow.quantification.records import GeometryRecord, MeasurementTable, ObjectRecord
from microflow.units import Calibration

PIXEL = Calibration(pixel_size=0.5, unit="um")
SQUARE_PERIMETER_PX = 4 * (10 - 1) + 4 * np.sqrt(0.5)


def _raw(values: np.ndarray, channels: tuple[str, ...] = ("DAPI",)) -> IntensitySource:
    planes = values[np.newaxis] if values.ndim == 2 else values
    return IntensitySource(name="raw", planes=planes.astype(np.float64), channels=channels)


def _square_labels(shape: tuple[int, int] = (40, 40)) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    labels[5:15, 3:13] = 1  # 10 x 10 pixels, rows 5..14, columns 3..12
    return labels


def _only_object(table: MeasurementTable) -> ObjectRecord:
    assert len(table.objects) == 1
    return table.objects[0]


def _disc(shape: tuple[int, int], centre: tuple[int, int], radius: float) -> np.ndarray:
    rows, cols = np.indices(shape)
    return ((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius**2).astype(np.int32)


# --- analytic geometry, pixel units -----------------------------------------------------------


def test_square_geometry_is_exact_in_pixels() -> None:
    labels = _square_labels()
    table = measure_plane(labels, [_raw(np.ones((40, 40)))], None)
    geometry = _only_object(table).geometry
    assert table.length_unit == "px"
    assert geometry.label == 1
    assert geometry.area == 100
    assert geometry.perimeter == pytest.approx(SQUARE_PERIMETER_PX)
    assert geometry.centroid_row == pytest.approx(10.0)  # index 9.5 plus half a pixel
    assert geometry.centroid_col == pytest.approx(8.0)  # index 7.5 plus half a pixel
    assert (geometry.bbox_min_row, geometry.bbox_min_col) == (5, 3)
    assert (geometry.bbox_max_row, geometry.bbox_max_col) == (15, 13)
    assert geometry.eccentricity == pytest.approx(0.0, abs=1e-6)
    assert geometry.equivalent_diameter == pytest.approx(np.sqrt(400 / np.pi))


def test_calibrated_geometry_scales_lengths_and_areas() -> None:
    table = measure_plane(_square_labels(), [_raw(np.ones((40, 40)))], PIXEL)
    geometry = _only_object(table).geometry
    assert table.length_unit == "um"
    assert geometry.area == pytest.approx(100 * 0.25)
    assert geometry.perimeter == pytest.approx(SQUARE_PERIMETER_PX * 0.5)
    assert geometry.centroid_row == pytest.approx(5.0)
    assert geometry.centroid_col == pytest.approx(4.0)
    assert (geometry.bbox_min_row, geometry.bbox_min_col) == (2.5, 1.5)
    assert (geometry.bbox_max_row, geometry.bbox_max_col) == (7.5, 6.5)
    assert geometry.equivalent_diameter == pytest.approx(np.sqrt(4 * 25.0 / np.pi))


def test_rectangle_area_and_bounding_box_are_exact() -> None:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[4:10, 6:15] = 1  # 6 rows x 9 columns
    geometry = _only_object(measure_plane(labels, [_raw(np.ones((20, 20)))], PIXEL)).geometry
    assert geometry.area == pytest.approx(54 * 0.25)
    assert (geometry.bbox_max_row - geometry.bbox_min_row) * 2 == pytest.approx(6)
    assert (geometry.bbox_max_col - geometry.bbox_min_col) * 2 == pytest.approx(9)


def test_disc_geometry_approximates_analytic_values() -> None:
    radius = 20.0
    labels = _disc((101, 101), (50, 50), radius)
    geometry = _only_object(measure_plane(labels, [_raw(np.ones((101, 101)))], None)).geometry
    assert geometry.area == pytest.approx(np.pi * radius**2, rel=0.01)
    # The 0.5-level iso-contour of a digital disc is longer than the circle. Measured ratios are
    # documented in docs/measurements.md; see test_disc_perimeter_bias_shrinks_with_radius.
    assert 1.0 < geometry.perimeter / (2 * np.pi * radius) < 1.10
    assert geometry.equivalent_diameter == pytest.approx(2 * radius, rel=0.01)
    assert geometry.centroid_row == pytest.approx(50.5)
    assert geometry.centroid_col == pytest.approx(50.5)
    assert geometry.eccentricity == pytest.approx(0.0, abs=0.02)


def test_disc_perimeter_bias_shrinks_with_radius() -> None:
    ratios = []
    for radius in (10.0, 20.0, 40.0):
        size = int(2 * radius + 21)
        labels = _disc((size, size), (size // 2, size // 2), radius)
        raw = _raw(np.ones((size, size)))
        perimeter = _only_object(measure_plane(labels, [raw], None)).geometry.perimeter
        ratios.append(perimeter / (2 * np.pi * radius))
    # Measured ratios are about 1.09, 1.07 and 1.06: always above the circle, and closer to it as
    # the disc grows. This pins the documented behaviour, not an exact estimate.
    assert all(ratio > 1.0 for ratio in ratios)
    assert ratios[0] > ratios[1] > ratios[2]


def test_ellipse_eccentricity_matches_axis_ratio() -> None:
    rows, cols = np.indices((101, 101))
    labels = (((rows - 50) / 10.0) ** 2 + ((cols - 50) / 20.0) ** 2 <= 1).astype(np.int32)
    geometry = _only_object(measure_plane(labels, [_raw(np.ones((101, 101)))], None)).geometry
    expected = np.sqrt(1 - (10.0 / 20.0) ** 2)
    assert geometry.eccentricity == pytest.approx(expected, abs=0.01)


def test_single_pixel_perimeter_is_its_diamond() -> None:
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[2, 2] = 1
    geometry = _only_object(measure_plane(labels, [_raw(np.ones((5, 5)))], None)).geometry
    assert geometry.area == 1
    assert geometry.perimeter == pytest.approx(2 * np.sqrt(2))


def test_equivalent_diameter_is_consistent_with_area_in_any_unit() -> None:
    labels = _disc((41, 41), (20, 20), 7.0)
    for calibration in (None, PIXEL, Calibration(pixel_size=2.0, unit="nm")):
        table = measure_plane(labels, [_raw(np.ones((41, 41)))], calibration)
        geometry = _only_object(table).geometry
        assert geometry.equivalent_diameter == pytest.approx(np.sqrt(4 * geometry.area / np.pi))


# --- intensity statistics ---------------------------------------------------------------------


def _row_object(values: list[float]) -> tuple[np.ndarray, IntensitySource]:
    labels = np.zeros((10, 20), dtype=np.int32)
    labels[5, 5 : 5 + len(values)] = 1
    planes = np.zeros((10, 20))
    planes[5, 5 : 5 + len(values)] = values
    return labels, _raw(planes)


def test_intensity_statistics_match_known_values() -> None:
    labels, raw = _row_object(list(range(1, 11)))
    record = _only_object(measure_plane(labels, [raw], PIXEL)).intensities[0]
    assert record.source == "raw"
    assert record.channel == "DAPI"
    assert record.valid_pixels == 10
    assert record.mean == pytest.approx(5.5)
    assert record.median == pytest.approx(5.5)
    assert record.std == pytest.approx(np.sqrt(99 / 12))  # population std of 1..10
    assert record.minimum == 1
    assert record.maximum == 10
    assert record.integrated == pytest.approx(55 * 0.25)


def test_integrated_intensity_is_plain_sum_without_calibration() -> None:
    labels, raw = _row_object(list(range(1, 11)))
    record = _only_object(measure_plane(labels, [raw], None)).intensities[0]
    assert record.integrated == pytest.approx(55.0)


def test_nan_pixels_are_excluded_and_counted() -> None:
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[2, 2:6] = 1
    planes = np.zeros((6, 6))
    planes[2, 2:6] = [1.0, 2.0, np.nan, 4.0]
    record = _only_object(measure_plane(labels, [_raw(planes)], PIXEL)).intensities[0]
    assert record.valid_pixels == 3
    assert record.mean == pytest.approx(7 / 3)
    assert record.median == pytest.approx(2.0)
    assert record.std == pytest.approx(np.std([1.0, 2.0, 4.0]))
    assert record.integrated == pytest.approx(7 * 0.25)


def test_object_without_finite_pixels_reports_nan() -> None:
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1, 1:3] = 1
    planes = np.full((6, 6), np.nan)
    record = _only_object(measure_plane(labels, [_raw(planes)], PIXEL)).intensities[0]
    assert record.valid_pixels == 0
    assert np.isnan(record.mean) and np.isnan(record.integrated)


def test_zero_intensity_pixels_are_real_values() -> None:
    labels, raw = _row_object([0.0, 0.0, 4.0])
    record = _only_object(measure_plane(labels, [raw], None)).intensities[0]
    assert record.valid_pixels == 3
    assert record.minimum == 0.0
    assert record.mean == pytest.approx(4 / 3)


# --- channels, sources and tables -------------------------------------------------------------


def test_raw_and_corrected_sources_are_kept_separate() -> None:
    labels = _square_labels()
    raw_planes = np.stack([np.full((40, 40), 10.0), np.full((40, 40), 100.0)])
    raw = IntensitySource(name="raw", planes=raw_planes, channels=("DAPI", "GFP"))
    corrected = IntensitySource(name="corrected", planes=raw_planes * 0.5, channels=("DAPI", "GFP"))
    table = measure_plane(labels, [raw, corrected], PIXEL)
    obj = _only_object(table)
    assert table.sources == ("raw", "corrected")
    assert [(i.source, i.channel) for i in obj.intensities] == [
        ("raw", "DAPI"),
        ("raw", "GFP"),
        ("corrected", "DAPI"),
        ("corrected", "GFP"),
    ]
    means = {(i.source, i.channel): i.mean for i in obj.intensities}
    assert means[("raw", "GFP")] == 100.0
    assert means[("corrected", "GFP")] == 50.0


def test_geometry_is_not_repeated_per_intensity_source() -> None:
    labels = _square_labels()
    raw = _raw(np.ones((40, 40)))
    corrected = IntensitySource(name="corrected", planes=raw.planes, channels=raw.channels)
    table = measure_plane(labels, [raw, corrected], None)
    columns = table.columns()
    geometry_columns = [c for c in columns if not c.startswith(("raw_", "corrected_"))]
    assert len(geometry_columns) == len(GeometryRecord.model_fields)
    assert len(table.objects) == 1
    assert isinstance(table.objects[0].geometry, GeometryRecord)


def test_one_row_per_object_keeping_label_values() -> None:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[1:3, 1:3] = 3
    labels[10:14, 10:12] = 7
    table = measure_plane(labels, [_raw(np.ones((20, 20)))], None)
    assert [obj.geometry.label for obj in table.objects] == [3, 7]
    assert [obj.geometry.area for obj in table.objects] == [4, 8]


def test_background_is_not_an_object() -> None:
    table = measure_plane(np.zeros((8, 8), dtype=np.int32), [_raw(np.ones((8, 8)))], PIXEL)
    assert table.objects == ()


def test_column_names_carry_the_length_unit() -> None:
    labels = _square_labels()
    raw = _raw(np.ones((40, 40)))
    pixels = measure_plane(labels, [raw], None).columns()
    physical = measure_plane(labels, [raw], PIXEL).columns()
    assert "area_px2" in pixels and "perimeter_px" in pixels
    assert "area_um2" in physical and "perimeter_um" in physical
    assert "raw_DAPI_integrated_intensity_um2" in physical
    assert "eccentricity" in physical
    assert physical[0] == "label"


def test_csv_round_trip_matches_the_typed_table(tmp_path: Path) -> None:
    labels = _square_labels()
    labels[30, 30:33] = 2
    table = measure_plane(labels, [_raw(np.full((40, 40), 200.0))], PIXEL)
    path = tmp_path / "measurements.csv"
    table.write_csv(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == table.columns()
        rows = list(reader)
    assert len(rows) == 2
    assert float(rows[0]["area_um2"]) == pytest.approx(25.0)
    assert float(rows[1]["area_um2"]) == pytest.approx(0.75)
    assert float(rows[0]["raw_DAPI_mean"]) == pytest.approx(200.0)


def test_empty_label_image_writes_only_a_header(tmp_path: Path) -> None:
    table = measure_plane(np.zeros((4, 4), dtype=np.int32), [_raw(np.ones((4, 4)))], None)
    path = tmp_path / "empty.csv"
    table.write_csv(path)
    assert path.read_text(encoding="utf-8").splitlines() == [",".join(table.columns())]


# --- input validation -------------------------------------------------------------------------


def test_labels_must_be_a_single_plane() -> None:
    with pytest.raises(MeasurementError, match="one Y-X label plane"):
        measure_plane(np.zeros((2, 4, 4), dtype=np.int32), [_raw(np.ones((4, 4)))], None)


def test_labels_must_be_integers() -> None:
    with pytest.raises(MeasurementError, match="integer dtype"):
        measure_plane(np.zeros((4, 4)), [_raw(np.ones((4, 4)))], None)


def test_intensity_planes_must_match_the_label_grid() -> None:
    with pytest.raises(MeasurementError, match="expected"):
        measure_plane(np.zeros((4, 4), dtype=np.int32), [_raw(np.ones((5, 5)))], None)


def test_at_least_one_source_is_required() -> None:
    with pytest.raises(MeasurementError, match="at least one"):
        measure_plane(np.zeros((4, 4), dtype=np.int32), [], None)


def test_sources_must_share_channel_names() -> None:
    raw = _raw(np.ones((4, 4)), channels=("DAPI",))
    other = IntensitySource(name="corrected", planes=raw.planes, channels=("GFP",))
    with pytest.raises(MeasurementError, match="same channel names"):
        measure_plane(np.zeros((4, 4), dtype=np.int32), [raw, other], None)


def test_channel_name_count_must_match_planes() -> None:
    bad = IntensitySource(name="raw", planes=np.ones((2, 4, 4)), channels=("DAPI",))
    with pytest.raises(MeasurementError, match="channel name"):
        measure_plane(np.zeros((4, 4), dtype=np.int32), [bad], None)


def test_duplicate_source_names_are_rejected() -> None:
    raw = _raw(np.ones((4, 4)))
    with pytest.raises(MeasurementError, match="distinct"):
        measure_plane(np.zeros((4, 4), dtype=np.int32), [raw, raw], None)
