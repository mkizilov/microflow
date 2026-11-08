"""A self-contained HTML report for a statistics run. Standard library only.

It reads what ``microflow stats run`` writes (summary.json, descriptive.csv, comparisons.csv and the
PNG figures in plots/). Optionally it reads the batch run that produced the measurements, so each
image's quality-control report can be linked. Figures are embedded as base64 PNGs, so the page is a
single file. Every value read from a file is HTML-escaped, and missing values are shown as NA.
"""

from __future__ import annotations

import base64
import csv
import html
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PLOT_DIR = "plots"
MANIFEST_FILE = "batch_manifest.json"
ITEMS_DIR = "items"
ITEM_RESULT = "result.json"

_STYLE = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; color: #222; max-width: 1100px;
       margin: 2em auto; padding: 0 1em; line-height: 1.45; }
h1 { font-size: 1.6em; } h2 { font-size: 1.25em; margin-top: 1.8em; border-bottom: 1px solid #ddd; }
table { border-collapse: collapse; width: 100%; font-size: 0.88em; margin: 0.6em 0; }
th, td { border-bottom: 1px solid #e4e4e4; padding: 0.3em 0.5em; text-align: left;
         vertical-align: top; }
th { background: #f5f5f5; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.meta { color: #555; font-size: 0.9em; } .warn { color: #8a4b00; }
figure { margin: 1em 0; } figure img { max-width: 100%; height: auto; }
figcaption { font-size: 0.85em; color: #555; }
"""


def render_study_report(results_dir: Path, output: Path, *, batch_dir: Path | None = None) -> Path:
    summary = _load_json(results_dir / "summary.json")
    descriptive = _read_rows(results_dir / "descriptive.csv")
    comparisons = _read_rows(results_dir / "comparisons.csv")
    figures = _figures(results_dir / PLOT_DIR)
    quality = _quality(batch_dir, output.parent) if batch_dir is not None else None
    page = _page(summary, descriptive, comparisons, figures, quality)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


def _page(
    summary: dict[str, Any],
    descriptive: list[dict[str, str]],
    comparisons: list[dict[str, str]],
    figures: list[tuple[str, str]],
    quality: list[dict[str, Any]] | None,
) -> str:
    title = _text(summary.get("name", "statistics"))
    sections = [
        f"<h1>{title}</h1>",
        _meta(summary),
        _counts(summary),
        _warnings(summary),
        "<h2>Descriptive statistics (per biological sample)</h2>",
        _descriptive_table(descriptive),
        "<h2>Comparisons</h2>",
        _comparison_table(comparisons, summary),
        "<h2>Figures</h2>",
        _figure_list(figures),
    ]
    if quality is not None:
        sections += ["<h2>Quality control (per image)</h2>", _quality_table(quality)]
    sections.append(_methods(summary))
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{title}</title><style>{_STYLE}</style></head><body>\n"
        + "\n".join(sections)
        + "\n</body></html>\n"
    )


def _meta(summary: dict[str, Any]) -> str:
    families = summary.get("test_families", {})
    items = [
        ("analysis", _text(str(summary.get("analysis_id", ""))[:16])),
        ("microflow", _text(summary.get("microflow_version", ""))),
        ("generated (UTC)", _text(summary.get("generated_utc", ""))),
        ("multiple-comparison correction", _text(summary.get("correction", ""))),
        ("bootstrap confidence", f"{float(summary.get('confidence', 0.95)):.0%}"),
        (
            "tests per family",
            ", ".join(f"{_text(name)} {int(count)}" for name, count in families.items()),
        ),
    ]
    rows = "".join(f"<tr><th>{_text(k)}</th><td>{v}</td></tr>" for k, v in items)
    return f"<table class=\"meta\">{rows}</table>"


def _counts(summary: dict[str, Any]) -> str:
    counts = summary.get("counts", {})
    labels = {
        "measures": "measures",
        "groups": "groups",
        "comparisons": "comparisons",
        "samples": "samples included",
        "excluded_samples": "samples excluded",
    }
    parts = [f"{labels[key]}: {int(counts[key])}" for key in labels if key in counts]
    return f"<p>{' &middot; '.join(parts)}</p>"


def _warnings(summary: dict[str, Any]) -> str:
    warnings = summary.get("warnings", [])
    if not warnings:
        return ""
    items = "".join(f"<li class=\"warn\">{_text(w)}</li>" for w in warnings)
    return f"<h2>Warnings</h2><ul>{items}</ul>"


def _descriptive_table(rows: list[dict[str, str]]) -> str:
    header = (
        "<tr><th>measure</th><th>unit</th><th>group</th><th>samples</th><th>objects</th>"
        "<th>mean</th><th>SD</th><th>mean CI</th><th>median (IQR)</th><th>flags</th></tr>"
    )
    body = []
    for row in rows:
        cells = [
            _text(row["measure"]),
            _text(row["unit"]),
            _text(row["group"]),
            _num(row["samples"]),
            _num(row["objects"]),
            _num(row["sample_mean"]),
            _num(row["sample_sd"]),
            f"{_num(row['mean_ci_low'])} to {_num(row['mean_ci_high'])}",
            (
                f"{_num(row['sample_median'])} "
                f"({_num(row['sample_q25'])} to {_num(row['sample_q75'])})"
            ),
            _text(row["flags"].replace(";", ", ")),
        ]
        body.append(_row(cells))
    return f"<table>{header}{''.join(body)}</table>"


def _comparison_table(rows: list[dict[str, str]], summary: dict[str, Any]) -> str:
    header = (
        "<tr><th>measure</th><th>comparison</th><th>n</th><th>difference [CI]</th>"
        "<th>Hedges g [CI]</th><th>Welch p</th><th>adjusted</th><th>Mann-Whitney p</th>"
        "<th>permutation p</th><th>flags</th></tr>"
    )
    body = []
    for row in rows:
        cells = [
            _text(row["measure"]),
            f"{_text(row['group_a'])} vs {_text(row['group_b'])}",
            f"{_num(row['samples_a'])}, {_num(row['samples_b'])}",
            (
                f"{_num(row['difference'])} "
                f"[{_num(row['difference_ci_low'])}, {_num(row['difference_ci_high'])}]"
            ),
            (
                f"{_num(row['hedges_g'])} "
                f"[{_num(row['hedges_g_ci_low'])}, {_num(row['hedges_g_ci_high'])}]"
            ),
            _num(row["p_welch"]),
            _num(row["p_welch_adjusted"]),
            f"{_num(row['p_mannwhitney'])} (adj. {_num(row['p_mannwhitney_adjusted'])})",
            (
                f"{_num(row['p_permutation'])} ({_text(row['permutation_method'])}; "
                f"adj. {_num(row['p_permutation_adjusted'])})"
            ),
            _text(row["flags"].replace(";", ", ")),
        ]
        body.append(_row(cells))
    note = (
        f"<p class=\"meta\">Adjusted p-values use "
        f"{_text(summary.get('correction', 'bh'))} within each test family. "
        "A NA means the test could not be computed for that comparison.</p>"
    )
    return f"<table>{header}{''.join(body)}</table>{note}"


def _figure_list(figures: list[tuple[str, str]]) -> str:
    if not figures:
        return "<p class=\"meta\">No figures were written for this run.</p>"
    items = [
        f"<figure><img src=\"data:image/png;base64,{data}\" alt=\"{_text(name)}\">"
        f"<figcaption>{_text(name)}</figcaption></figure>"
        for name, data in figures
    ]
    return "\n".join(items)


def _quality_table(rows: list[dict[str, Any]]) -> str:
    header = "<tr><th>image</th><th>status</th><th>objects</th><th>flags</th><th>report</th></tr>"
    body = []
    for row in rows:
        link = (
            f"<a href=\"{_text(row['link'])}\">qc_report.html</a>" if row["link"] else "NA"
        )
        cells = [
            _text(row["name"]),
            _text(row["status"]),
            _num(str(row["objects"])),
            _text(", ".join(row["flags"]) or "none"),
            link,
        ]
        body.append(_row(cells))
    return f"<table>{header}{''.join(body)}</table>"


def _methods(summary: dict[str, Any]) -> str:
    return (
        "<h2>Method notes</h2><ul class=\"meta\">"
        "<li>Inference uses one value per biological sample. Objects are summarised within each "
        "sample and are never treated as independent replicates.</li>"
        "<li>Intervals are percentile bootstrap intervals that resample samples within groups.</li>"
        "<li>Hedges' g is the standardised mean difference of the first group minus the second, "
        "with the small-sample correction.</li>"
        "<li>Results depend on the design and assumptions in the study design file. "
        f"Design identity {_text(str(summary.get('analysis_id', ''))[:12])}.</li></ul>"
    )


def _quality(batch_dir: Path, base: Path) -> list[dict[str, Any]]:
    manifest = _load_json(batch_dir / MANIFEST_FILE)
    reports = _report_links(batch_dir, base)
    rows = []
    for item in manifest.get("items", []):
        name = str(item["name"])
        rows.append(
            {
                "name": name,
                "status": str(item["status"]),
                "objects": int(item.get("objects", 0)),
                "flags": [str(flag) for flag in item.get("flags", [])],
                "link": reports.get(name),
            }
        )
    return rows


def _report_links(batch_dir: Path, base: Path) -> dict[str, str]:
    links: dict[str, str] = {}
    for result in sorted((batch_dir / ITEMS_DIR).glob(f"*/{ITEM_RESULT}")):
        item: dict[str, Any] = json.loads(result.read_text(encoding="utf-8"))
        report = item.get("outputs", {}).get("qc_report")
        if report is not None:
            target = result.parent / report["path"]
            links[str(item["name"])] = os.path.relpath(target, base).replace(os.sep, "/")
    return links


def _figures(plots: Path) -> list[tuple[str, str]]:
    if not plots.is_dir():
        return []
    figures = []
    for path in sorted(plots.glob("*.png")):
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        figures.append((path.stem, data))
    return figures


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _row(cells: Sequence[str]) -> str:
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def _num(value: str | int) -> str:
    text = str(value)
    if text == "NA" or text == "":
        return "NA"
    try:
        return f"{float(text):.4g}"
    except ValueError:
        return _text(text)


def _text(value: object) -> str:
    return html.escape(str(value), quote=True)
