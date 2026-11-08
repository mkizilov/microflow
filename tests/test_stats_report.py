import base64
import json
from pathlib import Path

from microflow.stats.report import render_study_report

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-image-body"
DESCRIPTIVE_HEADER = (
    "measure,unit,group,samples,items,objects,sample_mean,sample_sd,sem,mean_ci_low,mean_ci_high,"
    "sample_median,sample_q25,sample_q75,object_mean,object_sd,object_median,flags\n"
)
COMPARISON_HEADER = (
    "measure,unit,group_a,group_b,samples_a,samples_b,objects_a,objects_b,mean_a,mean_b,difference,"
    "difference_ci_low,difference_ci_high,hedges_g,hedges_g_ci_low,hedges_g_ci_high,rank_biserial,"
    "p_welch,p_welch_adjusted,p_mannwhitney,p_mannwhitney_adjusted,p_permutation,"
    "p_permutation_adjusted,permutation_method,flags\n"
)


def _results(root: Path) -> Path:
    results = root / "stats"
    (results / "plots").mkdir(parents=True)
    summary = {
        "name": "Area <comparison> & check",
        "analysis_id": "f" * 64,
        "microflow_version": "0.1.0",
        "generated_utc": "2026-01-01T00:00:00+00:00",
        "correction": "bh",
        "confidence": 0.95,
        "test_families": {"welch": 1, "mannwhitney": 1, "permutation": 1},
        "counts": {
            "measures": 1,
            "groups": 2,
            "comparisons": 1,
            "samples": 4,
            "excluded_samples": 0,
        },
        "warnings": ["area: one sample <excluded>"],
    }
    (results / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (results / "descriptive.csv").write_text(
        DESCRIPTIVE_HEADER
        + "area_um2,um2,control,2,4,6,11.0,1.414,1.0,9.0,13.0,11.0,10.5,11.5,11.0,1.5,11.0,\n",
        encoding="utf-8",
    )
    (results / "comparisons.csv").write_text(
        COMPARISON_HEADER + "area_um2,um2,control,treated,2,2,6,6,11.0,17.0,-6.0,-8.0,-4.0,"
        "-3.1,-5.0,-1.0,-1.0,0.0333,0.0333,0.1,0.1,0.02,0.02,exact,\n",
        encoding="utf-8",
    )
    (results / "plots" / "area_um2.png").write_bytes(PNG)
    return results


def _qc(root: Path) -> Path:
    batch = root / "batch"
    item = batch / "items" / "sample-1234abcd"
    item.mkdir(parents=True)
    (item / "qc_report.html").write_text("<html>qc</html>", encoding="utf-8")
    (item / "result.json").write_text(
        json.dumps(
            {
                "name": "sample",
                "outputs": {
                    "qc_report": {"path": "qc_report.html", "sha256": "x", "size_bytes": 1}
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "items": [{"name": "sample", "status": "ok", "objects": 6, "flags": ["excessive_coverage"]}]
    }
    (batch / "batch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return batch


def test_the_report_is_one_file_with_the_figure_embedded(tmp_path: Path) -> None:
    results = _results(tmp_path)
    output = render_study_report(results, tmp_path / "report.html")
    page = output.read_text(encoding="utf-8")
    assert page.startswith("<!DOCTYPE html>")
    assert "data:image/png;base64," + base64.b64encode(PNG).decode("ascii") in page


def test_values_from_files_are_escaped_not_injected(tmp_path: Path) -> None:
    results = _results(tmp_path)
    page = render_study_report(results, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Area &lt;comparison&gt; &amp; check" in page
    assert "<comparison>" not in page
    assert "one sample &lt;excluded&gt;" in page


def test_the_tables_show_counts_uncertainty_and_adjusted_p_values(tmp_path: Path) -> None:
    results = _results(tmp_path)
    page = render_study_report(results, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<th>samples</th>" in page
    assert "9 to 13" in page
    assert "-3.1 [-5, -1]" in page
    assert "exact" in page
    assert "Adjusted p-values use bh" in page


def test_missing_values_are_shown_as_na(tmp_path: Path) -> None:
    results = _results(tmp_path)
    (results / "comparisons.csv").write_text(
        COMPARISON_HEADER
        + "area_um2,um2,control,treated,1,2,1,6,11.0,17.0,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,"
        "not_applicable,insufficient_replication\n",
        encoding="utf-8",
    )
    page = render_study_report(results, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "NA" in page
    assert "insufficient_replication" in page


def test_quality_control_links_are_relative_to_the_report(tmp_path: Path) -> None:
    results = _results(tmp_path)
    batch = _qc(tmp_path)
    report_dir = tmp_path / "reports"
    page = render_study_report(results, report_dir / "report.html", batch_dir=batch).read_text(
        encoding="utf-8"
    )
    assert "<h2>Quality control (per image)</h2>" in page
    assert "../batch/items/sample-1234abcd/qc_report.html" in page
    assert "excessive_coverage" in page


def test_a_run_without_figures_still_produces_a_report(tmp_path: Path) -> None:
    results = _results(tmp_path)
    (results / "plots" / "area_um2.png").unlink()
    page = render_study_report(results, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "No figures were written for this run." in page
