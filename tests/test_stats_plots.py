import subprocess
import sys
from pathlib import Path

from microflow.stats.study import run_study
from stats_fixtures import design_mapping, write_design, write_dictionary, write_measurements

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _run(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_measurements(root)
    write_dictionary(root)
    design = write_design(root, design_mapping(root, plots=True))
    return run_study(design).directory


def test_each_measure_and_the_effect_sizes_get_a_png_and_an_svg(tmp_path: Path) -> None:
    directory = _run(tmp_path / "run")
    plots = directory / "plots"
    for name in ("area_um2.png", "area_um2.svg", "effect_sizes.png", "effect_sizes.svg"):
        assert (plots / name).is_file(), name
    assert (plots / "area_um2.png").read_bytes().startswith(PNG_SIGNATURE)
    assert (plots / "area_um2.svg").read_text(encoding="utf-8").lstrip().startswith("<?xml")


def test_the_svg_carries_no_creation_date(tmp_path: Path) -> None:
    directory = _run(tmp_path / "run")
    svg = (directory / "plots" / "area_um2.svg").read_text(encoding="utf-8")
    assert "<dc:date>" not in svg


def test_repeated_runs_give_identical_figure_bytes(tmp_path: Path) -> None:
    first = _run(tmp_path / "first") / "plots"
    second = _run(tmp_path / "second") / "plots"
    for name in ("area_um2.png", "area_um2.svg", "effect_sizes.svg"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_the_plot_module_does_not_load_pyplot() -> None:
    code = (
        "import sys, microflow.stats.plots; "
        "sys.exit(1 if 'matplotlib.pyplot' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
