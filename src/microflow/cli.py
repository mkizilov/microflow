"""Command-line interface for MicroFlow."""

from pathlib import Path
from typing import Annotated

import typer

from microflow import __version__
from microflow.sample_data import write_sample_dataset

app = typer.Typer(
    help="MicroFlow: scientific image processing and quantitative analysis.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the MicroFlow version."""
    typer.echo(__version__)


@app.command("make-sample-data")
def make_sample_data(
    out_dir: Annotated[Path, typer.Option(help="Destination directory.")] = Path(
        "data/synthetic"
    ),
    seed: Annotated[int, typer.Option(help="Base random seed.")] = 0,
    count: Annotated[int, typer.Option(min=1, help="Number of images to write.")] = 3,
) -> None:
    """Write deterministic synthetic TIFF images with calibration sidecars."""
    for path in write_sample_dataset(out_dir, seed=seed, count=count):
        typer.echo(f"wrote {path}")
