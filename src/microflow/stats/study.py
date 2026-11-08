"""Run a study from its design file: read, check, analyse, then write.

Everything that can fail on bad input (the design, the table, the assignments) happens before the
output directory is touched. A results directory is therefore never left half-written by invalid
data.
"""

from dataclasses import dataclass
from pathlib import Path

from microflow.provenance.hashing import sha256_file
from microflow.stats.analysis import AnalysisResult, run_analysis
from microflow.stats.data import read_measurements
from microflow.stats.design import load_design
from microflow.stats.output import (
    PLOT_DIR,
    analysis_identity,
    prepare_output,
    write_summary,
    write_tables,
)


@dataclass(frozen=True)
class StudyOutput:
    result: AnalysisResult
    directory: Path
    identity: str
    plots: tuple[str, ...]


def run_study(
    design_path: Path, *, overwrite: bool = False, plots: bool | None = None
) -> StudyOutput:
    design = load_design(design_path)
    table = read_measurements(
        design.measurements, measures=design.measures, dictionary=design.data_dictionary
    )
    result = run_analysis(design, table)
    inputs: dict[str, str | None] = {
        "measurements_sha256": sha256_file(design.measurements),
        "data_dictionary_sha256": (
            sha256_file(design.data_dictionary) if design.data_dictionary is not None else None
        ),
    }
    identity = analysis_identity(
        design, str(inputs["measurements_sha256"]), inputs["data_dictionary_sha256"]
    )
    directory = design.output_directory
    prepare_output(directory, identity, overwrite=overwrite)
    write_tables(directory, result)
    figures: tuple[str, ...] = ()
    if design.plots if plots is None else plots:
        from microflow.stats.plots import write_plots

        figures = write_plots(directory / PLOT_DIR, result, table, design)
    write_summary(
        directory,
        result,
        design=design,
        identity=identity,
        inputs=inputs,
        plots=figures,
    )
    return StudyOutput(result=result, directory=directory, identity=identity, plots=figures)
