import struct
import subprocess
import sys

import numpy as np

from microflow.config import QualityControlConfig
from microflow.loading import load_array
from microflow.reporting.figures import FIGURE_TITLES, render_figures
from microflow.reporting.html import render_html
from microflow.reporting.qc import (
    QualityInputs,
    QualityReport,
    build_quality_report,
    inputs_from_images,
)
from microflow.units import Calibration

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_END = b"IEND\xaeB`\x82"
PIXEL = Calibration(pixel_size=0.5, unit="um")


def _build(
    raw: np.ndarray,
    labels: np.ndarray,
    *,
    preprocessed: np.ndarray | None = None,
    settings: QualityControlConfig | None = None,
    calibration: Calibration | None = None,
    title: str = "Test run",
    axes: str = "YX",
) -> tuple[QualityInputs, QualityReport]:
    raw_image = load_array(raw, axes=axes, calibration=calibration)
    label_image = load_array(labels, axes=axes)
    pre_image = load_array(preprocessed, axes=axes) if preprocessed is not None else None
    inputs = inputs_from_images(
        raw_image,
        label_image,
        settings or QualityControlConfig(),
        preprocessed=pre_image,
        title=title,
    )
    return inputs, build_quality_report(inputs)


def _objects() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    raw = rng.normal(100.0, 5.0, (64, 64))
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[10:20, 10:22] = 1
    labels[40:50, 30:36] = 2
    raw[labels > 0] += 300.0
    return raw, labels


def _png_size(data: bytes) -> tuple[int, int]:
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def test_figures_are_valid_png_files_in_title_order() -> None:
    raw, labels = _objects()
    inputs, report = _build(raw, labels, preprocessed=raw * 0.5, calibration=PIXEL)
    figures = render_figures(inputs, report)
    assert tuple(figures) == tuple(FIGURE_TITLES)
    for name, data in figures.items():
        assert data.startswith(PNG_SIGNATURE), name
        assert data.endswith(PNG_END), name
        assert _png_size(data)[0] > 0 and _png_size(data)[1] > 0


def test_preprocessed_figure_is_omitted_without_a_preprocessed_image() -> None:
    raw, labels = _objects()
    inputs, report = _build(raw, labels)
    figures = render_figures(inputs, report)
    assert "preprocessed" not in figures
    assert "original" in figures and "area" in figures


def test_figures_render_for_an_empty_mask_and_non_finite_values() -> None:
    raw = np.full((32, 32), 50.0)
    raw[:4, :] = np.nan
    labels = np.zeros((32, 32), dtype=np.int32)
    inputs, report = _build(raw, labels, calibration=PIXEL)
    figures = render_figures(inputs, report)
    assert all(data.startswith(PNG_SIGNATURE) for data in figures.values())


def test_rendering_is_deterministic() -> None:
    raw, labels = _objects()
    inputs, report = _build(raw, labels, calibration=PIXEL)
    assert render_figures(inputs, report) == render_figures(inputs, report)


def test_headless_import_does_not_load_pyplot() -> None:
    code = (
        "import sys\n"
        "import microflow.reporting.figures\n"
        "import microflow.reporting.html\n"
        "print('pyplot' in ' '.join(sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_html_is_self_contained_and_embeds_every_figure() -> None:
    raw, labels = _objects()
    inputs, report = _build(raw, labels, preprocessed=raw * 0.5, calibration=PIXEL)
    figures = render_figures(inputs, report)
    page = render_html(report, figures)
    assert page.count("data:image/png;base64,") == len(figures)
    assert 'src="http' not in page
    assert "<link" not in page
    assert "<script" not in page
    assert "Segmentation summary" in page
    assert "Flags" in page


def test_html_escapes_the_title_and_text_fields() -> None:
    raw, labels = _objects()
    inputs, report = _build(raw, labels, title="<script>alert(1)</script>")
    page = render_html(report, render_figures(inputs, report))
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_html_lists_flags_and_reports_when_none_were_raised() -> None:
    raw, labels = _objects()
    clean_inputs, clean_report = _build(raw, labels, calibration=PIXEL)
    clean_page = render_html(clean_report, render_figures(clean_inputs, clean_report))
    assert "No suspicious results were flagged." in clean_page

    empty_inputs, empty_report = _build(raw, np.zeros_like(labels))
    empty_page = render_html(empty_report, render_figures(empty_inputs, empty_report))
    assert "empty_mask" in empty_page
    assert "No suspicious results" not in empty_page


def test_html_summary_has_a_row_per_plane() -> None:
    raw = np.stack([np.full((16, 16), 10.0), np.full((16, 16), 20.0)])
    labels = np.zeros((2, 16, 16), dtype=np.int32)
    labels[0, 2:6, 2:6] = 1
    inputs, report = _build(raw, labels, axes="CYX")
    page = render_html(report, render_figures(inputs, report))
    assert "<td>C0</td>" in page and "<td>C1</td>" in page


def test_html_documents_the_configured_area_range() -> None:
    from microflow.units import Area

    raw, labels = _objects()
    settings = QualityControlConfig(min_object_area=Area(value=5, unit="um"))
    inputs, report = _build(raw, labels, settings=settings, calibration=PIXEL)
    page = render_html(report, render_figures(inputs, report))
    assert "Configured object area range: 5 to none um2" in page
