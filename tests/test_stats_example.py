from pathlib import Path

import numpy as np
import pytest

from microflow.stats.analysis import run_analysis
from microflow.stats.data import check_assignments, read_measurements
from microflow.stats.design import load_design
from microflow.stats.example import write_example
from microflow.stats.study import run_study

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DESIGN = ROOT / "examples" / "stats" / "design.yaml"


def _example_design(tmp_path: Path) -> Path:
    """The shipped example design, with its data and output redirected into tmp_path."""
    write_example(tmp_path / "data", seed=0)
    text = EXAMPLE_DESIGN.read_text(encoding="utf-8")
    text = text.replace(
        "../../data/stats-example/measurements.csv",
        str((tmp_path / "data" / "measurements.csv").as_posix()),
    )
    text = text.replace(
        "../../data/stats-example/data_dictionary.csv",
        str((tmp_path / "data" / "data_dictionary.csv").as_posix()),
    )
    text = text.replace(
        "../../stats-results/synthetic-two-condition",
        str((tmp_path / "results").as_posix()),
    )
    text = text.replace("resamples: 10000", "resamples: 2000")
    path = tmp_path / "design.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_example_design_matches_the_generated_items_exactly(tmp_path: Path) -> None:
    design = load_design(_example_design(tmp_path))
    table = read_measurements(
        design.measurements, measures=design.measures, dictionary=design.data_dictionary
    )
    check_assignments(table, design)
    assigned = {item for group in design.groups.values() for s in group.samples for item in s.items}
    assert len(assigned) == 16


def test_the_designed_area_effect_is_recovered_at_the_donor_level(tmp_path: Path) -> None:
    design = load_design(_example_design(tmp_path))
    table = read_measurements(
        design.measurements, measures=design.measures, dictionary=design.data_dictionary
    )
    area = next(row for row in run_analysis(design, table).comparisons if row.measure == "area_um2")
    assert area.samples_a == area.samples_b == 4
    # Treated objects are 1.4 times larger, so control minus treated is negative. The effect is
    # large relative to donor-to-donor variation, so the sample-level estimate is strongly negative
    # and the Welch test is significant. These are predictions from the generator's parameters that
    # have not been run here.
    assert area.mean_a < area.mean_b
    assert area.hedges_g < -1.0
    assert area.p_welch < 0.05
    # Eccentricity and raw intensity carry no designed effect. Their p-values are not asserted: at
    # four donors per group a null measure can cross 0.05 by chance, and this example is not a
    # calibration study.


def test_the_example_run_writes_the_complete_result_set(tmp_path: Path) -> None:
    design_path = _example_design(tmp_path)
    output = run_study(design_path, plots=False)
    assert (output.directory / "summary.json").is_file()
    assert (output.directory / "descriptive.csv").is_file()
    assert (output.directory / "comparisons.csv").is_file()
    assert output.result.families["welch"] == 3
    assert np.isfinite(output.result.comparisons[0].hedges_g)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_generator_is_reproducible_for_a_seed(tmp_path: Path, seed: int) -> None:
    first = write_example(tmp_path / f"a{seed}", seed=seed)[0].read_bytes()
    second = write_example(tmp_path / f"b{seed}", seed=seed)[0].read_bytes()
    assert first == second
