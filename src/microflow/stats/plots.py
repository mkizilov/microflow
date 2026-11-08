"""Publication figures for statistical results: sample-level comparisons and effect sizes.

Figures use the object-oriented matplotlib API on an Agg canvas and never import pyplot, so they
render without a display. Points are biological samples. Object-level values appear only as faint
background points, because the sample is the unit of inference. Each figure is written as PNG at 300
dpi for print and as SVG for editing. The SVG date is omitted and the SVG identifiers are fixed, so
identical inputs give identical files.
"""

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from microflow.stats.analysis import AnalysisResult, ComparisonRow, DescriptiveRow
from microflow.stats.data import ObjectTable, group_objects
from microflow.stats.design import StatsDesign

PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")
MAX_BACKGROUND_POINTS = 400
JITTER_WIDTH = 0.22
STYLE: dict[str, Any] = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "microflow",
    "savefig.dpi": 300,
}


def write_plots(
    directory: Path,
    result: AnalysisResult,
    table: ObjectTable,
    design: StatsDesign,
) -> tuple[str, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(design.bootstrap.seed)
    groups = list(design.groups)
    written: list[str] = []
    for measure in design.measures:
        unit = table.units.get(measure, "")
        sample_points = {
            group: np.asarray(
                [row.value for row in result.samples
                 if row.measure == measure and row.group == group and row.included],
                dtype=np.float64,
            )
            for group in groups
        }
        object_points = {
            group: group_objects(table, design, measure, group) for group in groups
        }
        means = {row.group: row for row in result.descriptive if row.measure == measure}
        figure = render_measure(
            measure=measure,
            unit=unit,
            groups=groups,
            sample_points=sample_points,
            object_points=object_points,
            descriptive=means,
            rng=rng,
            confidence=result.confidence,
        )
        written += write_figure(figure, directory / _slug(measure))
    testable = [row for row in result.comparisons if math.isfinite(row.hedges_g)]
    if testable:
        written += write_figure(
            render_effect_sizes(testable, confidence=result.confidence),
            directory / "effect_sizes",
        )
    return tuple(written)


def render_measure(
    *,
    measure: str,
    unit: str,
    groups: Sequence[str],
    sample_points: Mapping[str, NDArray[np.float64]],
    object_points: Mapping[str, NDArray[np.float64]],
    descriptive: Mapping[str, DescriptiveRow],
    rng: np.random.Generator,
    confidence: float,
) -> Figure:
    with rc_context(STYLE):
        figure = Figure(figsize=(3.3, 3.2), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.add_subplot()
        for position, group in enumerate(groups):
            colour = PALETTE[position % len(PALETTE)]
            background = _downsample(object_points[group], rng)
            axis.scatter(
                position + _jitter(rng, background.size),
                background,
                s=1.5,
                color="0.75",
                linewidths=0,
                zorder=1,
            )
            samples = sample_points[group]
            axis.scatter(
                position + _jitter(rng, samples.size),
                samples,
                s=16,
                color=colour,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
            row = descriptive.get(group)
            if row is None or not math.isfinite(row.sample_mean):
                continue
            if math.isfinite(row.mean_ci_low):
                below = row.sample_mean - row.mean_ci_low
                above = row.mean_ci_high - row.sample_mean
                axis.errorbar(
                    position,
                    row.sample_mean,
                    yerr=[[below], [above]],
                    fmt="_",
                    color="black",
                    markersize=10,
                    capsize=3,
                    linewidth=1,
                    zorder=4,
                )
            else:
                axis.plot([position], [row.sample_mean], marker="_", color="black", markersize=10)
        labels = [f"{group}\n(n = {descriptive[group].samples})" if group in descriptive else group
                  for group in groups]
        axis.set_xticks(range(len(groups)), labels=labels)
        axis.set_xlim(-0.6, len(groups) - 0.4)
        axis.set_ylabel(f"{measure} ({unit})" if unit else measure)
        figure.suptitle(
            f"Points: biological samples. Bars: mean and {confidence:.0%} bootstrap interval.",
            fontsize=6,
        )
    return figure


def render_effect_sizes(comparisons: Sequence[ComparisonRow], *, confidence: float) -> Figure:
    with rc_context(STYLE):
        height = max(2.0, 0.42 * len(comparisons) + 1.4)
        figure = Figure(figsize=(3.6, height), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.add_subplot()
        positions = list(range(len(comparisons)))[::-1]
        for position, row in zip(positions, comparisons, strict=True):
            colour = PALETTE[position % len(PALETTE)]
            low, high, g = row.hedges_g_ci_low, row.hedges_g_ci_high, row.hedges_g
            if math.isfinite(low) and math.isfinite(high):
                axis.errorbar(
                    g,
                    position,
                    xerr=[[g - low], [high - g]],
                    fmt="o",
                    color=colour,
                    capsize=2,
                    markersize=4,
                )
            else:
                axis.plot([g], [position], marker="o", color=colour, markersize=4)
        axis.axvline(0, color="0.5", linewidth=0.8, linestyle="--")
        axis.set_yticks(
            positions,
            labels=[
                f"{row.measure}\n{row.group_a} vs {row.group_b}" for row in comparisons
            ],
        )
        axis.set_xlabel(f"Hedges' g (mean of group a minus group b; {confidence:.0%} bootstrap CI)")
        axis.set_ylim(-0.7, len(comparisons) - 0.3)
    return figure


def write_figure(figure: Figure, base: Path) -> list[str]:
    """Save PNG and SVG under the same base name. Returns the file names written."""
    png = base.with_suffix(".png")
    svg = base.with_suffix(".svg")
    with rc_context(STYLE):
        figure.savefig(png, format="png")
        figure.savefig(svg, format="svg", metadata={"Date": None})
    return [png.name, svg.name]


def _downsample(values: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
    if values.size <= MAX_BACKGROUND_POINTS:
        return values
    chosen = np.sort(rng.choice(values.size, size=MAX_BACKGROUND_POINTS, replace=False))
    return values[chosen]


def _jitter(rng: np.random.Generator, count: int) -> NDArray[np.float64]:
    return rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=count)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)
