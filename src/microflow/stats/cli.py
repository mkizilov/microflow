"""The ``microflow stats`` command group. Needs no PyTorch."""

import math
from pathlib import Path
from typing import Annotated, NoReturn

import numpy as np
import typer

from microflow.config import ConfigError
from microflow.provenance.guard import OutputConflictError
from microflow.stats.analysis import ComparisonRow
from microflow.stats.data import check_assignments, read_measurements
from microflow.stats.design import load_design
from microflow.stats.errors import StatsError
from microflow.stats.example import write_example
from microflow.stats.study import run_study

stats_app = typer.Typer(
    help="Compare groups of biological samples: descriptive statistics, bootstrap intervals, "
    "effect sizes, tests with corrected p-values, and figures."
)

_FAILURES = (ConfigError, StatsError, OutputConflictError, ValueError, OSError)


@stats_app.command("run")
def run_command(
    design: Annotated[Path, typer.Argument(help="Study design YAML.")],
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace results of a different analysis."),
    ] = False,
    no_plots: Annotated[bool, typer.Option("--no-plots", help="Skip the figures.")] = False,
) -> None:
    """Run the analysis described by a design file and write its tables and figures."""
    try:
        output = run_study(design, overwrite=overwrite, plots=False if no_plots else None)
    except _FAILURES as exc:
        _fail(exc)
    result = output.result
    typer.echo(
        f"wrote {len(result.descriptive)} descriptive row(s) and {len(result.comparisons)} "
        f"comparison(s) to {output.directory}"
    )
    for comparison in result.comparisons:
        typer.echo(_comparison_line(comparison, result.confidence))
    for warning in result.warnings:
        typer.echo(f"warning: {warning}")
    typer.echo(f"analysis {output.identity[:12]}")


@stats_app.command("check")
def check_command(
    design: Annotated[Path, typer.Argument(help="Study design YAML.")],
) -> None:
    """Check a design against its measurements and show each group's sample sizes.

    No test is run.
    """
    try:
        loaded = load_design(design)
        table = read_measurements(
            loaded.measurements, measures=loaded.measures, dictionary=loaded.data_dictionary
        )
        check_assignments(table, loaded)
    except _FAILURES as exc:
        _fail(exc)
    typer.echo(f"{loaded.name}: {len(loaded.measures)} measure(s), {table.objects} object row(s)")
    for name, group in loaded.groups.items():
        items = [item for sample in group.samples for item in sample.items]
        objects = int(np.isin(table.items, items).sum())
        typer.echo(
            f"group {name:<12} {len(group.samples)} sample(s), {len(items)} image(s), "
            f"{objects} object row(s)"
        )


@stats_app.command("example")
def example_command(
    directory: Annotated[Path, typer.Argument(help="Directory to write the example data to.")],
    seed: Annotated[int, typer.Option(min=0, help="Seed for the generator.")] = 0,
) -> None:
    """Write the synthetic two-condition example measurements and data dictionary."""
    try:
        measurements, dictionary = write_example(directory, seed=seed)
    except OSError as exc:
        _fail(exc)
    typer.echo(f"wrote {measurements}")
    typer.echo(f"wrote {dictionary}")


def _comparison_line(row: ComparisonRow, confidence: float) -> str:
    effect = _number(row.hedges_g)
    low = _number(row.hedges_g_ci_low)
    high = _number(row.hedges_g_ci_high)
    adjusted = _number(row.p_welch_adjusted)
    return (
        f"{row.measure}: {row.group_a} vs {row.group_b} (n = {row.samples_a}, {row.samples_b}): "
        f"Hedges' g {effect} [{low}, {high}] ({confidence:.0%} CI), "
        f"Welch p adjusted {adjusted}"
    )


def _number(value: float) -> str:
    return "NA" if math.isnan(value) else f"{value:.3f}"


def _fail(exc: BaseException) -> NoReturn:
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code=1) from exc
