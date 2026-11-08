"""Command-line interface for MicroFlow."""

from pathlib import Path
from typing import Annotated

import typer

from microflow import __version__
from microflow.config import ConfigError, load_config
from microflow.dataset import inspect_dataset
from microflow.sample_data import write_sample_dataset

app = typer.Typer(
    help="MicroFlow: scientific image processing and quantitative analysis.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the MicroFlow version."""
    typer.echo(__version__)


@app.command("validate-config")
def validate_config(
    config_path: Annotated[Path, typer.Argument(help="Pipeline YAML configuration file.")],
    check_images: Annotated[
        bool,
        typer.Option("--check-images", help="Also read each image header and check it."),
    ] = False,
) -> None:
    """Validate a configuration file and, optionally, the images it references."""
    try:
        config = load_config(config_path)
        metadata = inspect_dataset(config) if check_images else []
    except ConfigError as exc:
        typer.echo(f"invalid: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"valid: {config_path} ({len(config.dataset.images)} images)")
    for image in metadata:
        typer.echo(f"  {image.name}: axes={image.axes} shape={image.shape} dtype={image.dtype}")


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
