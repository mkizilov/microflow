"""Command-line interface for MicroFlow."""

import json
from pathlib import Path
from typing import Annotated

import typer

from pydantic import ValidationError

from microflow import __version__
from microflow.config import (
    ConfigError,
    SegmentationConfig,
    load_config,
    load_segmentation_config,
)
from microflow.dataset import inspect_dataset
from microflow.image_io import write_image
from microflow.loading import ImageLoadError, load_image
from microflow.preprocessing import PreprocessingError, preprocess
from microflow.preprocessing.steps import load_steps
from microflow.reporting.preview import first_plane, write_before_after_png
from microflow.sample_data import write_sample_dataset
from microflow.segmentation import score_segmentation, segment_image
from microflow.units import Calibration

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


@app.command("preprocess")
def preprocess_image(
    input_path: Annotated[Path, typer.Argument(help="Input TIFF or .npy image.")],
    output: Annotated[Path, typer.Option(help="Output TIFF; written as float64.")],
    steps: Annotated[
        Path | None, typer.Option(help="YAML file listing preprocessing steps in order.")
    ] = None,
    flatfield: Annotated[
        Path | None, typer.Option(help="Flat-field reference image with axes YX.")
    ] = None,
    pixel_size: Annotated[
        float | None,
        typer.Option(help="Pixel size; overrides the file's calibration. Needs --unit."),
    ] = None,
    unit: Annotated[
        str | None, typer.Option(help="Length unit for --pixel-size: nm, um, mm or m.")
    ] = None,
    preview: Annotated[
        Path | None, typer.Option(help="Write a before/after PNG of the first plane.")
    ] = None,
) -> None:
    """Apply preprocessing steps and write the result with a provenance sidecar.

    The sidecar is OUTPUT with a .provenance.json suffix. It records each stage with its
    parameters, derived pixel sizes and normalisation scale.
    """
    try:
        image = load_image(input_path, calibration=_calibration_option(pixel_size, unit))
        flat = load_image(flatfield) if flatfield is not None else None
        step_list = load_steps(steps) if steps is not None else []
        result = preprocess(image, step_list, flatfield=flat)
    except (ImageLoadError, ConfigError, PreprocessingError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    write_image(output, result.data)
    provenance_path = output.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(result.provenance(str(input_path)), indent=2) + "\n", encoding="utf-8"
    )
    if preview is not None:
        write_before_after_png(
            preview,
            first_plane(image.data, image.axes),
            first_plane(result.data, result.axes),
        )
    typer.echo(f"wrote {output}")
    typer.echo(f"wrote {provenance_path}")


def _calibration_option(pixel_size: float | None, unit: str | None) -> Calibration | None:
    if pixel_size is None:
        return None
    if unit is None:
        typer.echo("error: --pixel-size needs --unit", err=True)
        raise typer.Exit(code=1)
    try:
        return Calibration.model_validate({"pixel_size": pixel_size, "unit": unit})
    except ValidationError as exc:
        typer.echo(f"error: invalid calibration: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("segment")
def segment_command(
    input_path: Annotated[Path, typer.Argument(help="Input TIFF or .npy image.")],
    output: Annotated[Path, typer.Option(help="Output label image (int32; 0 is background).")],
    config: Annotated[
        Path | None, typer.Option(help="YAML file with segmentation settings.")
    ] = None,
    reference: Annotated[
        Path | None,
        typer.Option(help="Ground-truth label image; reports Dice and IoU against it."),
    ] = None,
    pixel_size: Annotated[
        float | None,
        typer.Option(help="Pixel size; overrides the file's calibration. Needs --unit."),
    ] = None,
    unit: Annotated[
        str | None, typer.Option(help="Length unit for --pixel-size: nm, um, mm or m.")
    ] = None,
) -> None:
    """Segment objects and write a label image plus a JSON sidecar.

    The sidecar is OUTPUT with a .segmentation.json suffix. It records the settings, the
    threshold used on each plane, the object counts, and Dice and IoU if a reference is given.
    """
    try:
        image = load_image(input_path, calibration=_calibration_option(pixel_size, unit))
        settings = load_segmentation_config(config) if config is not None else SegmentationConfig()
        result = segment_image(image.data, image.axes, settings, image.calibration)
        scores = None
        if reference is not None:
            scores = score_segmentation(result.labels, load_image(reference).data)
    except (ImageLoadError, ConfigError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    write_image(output, result.labels)
    record = {
        "source": str(input_path),
        "axes": result.axes,
        "object_counts": list(result.object_counts),
        "thresholds": list(result.thresholds),
        "settings": settings.model_dump(mode="json"),
        "scores": scores.model_dump() if scores is not None else None,
    }
    sidecar = output.with_suffix(".segmentation.json")
    sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"wrote {output} ({sum(result.object_counts)} objects)")
    if scores is not None:
        typer.echo(f"Dice {scores.dice:.4f}  IoU {scores.iou:.4f}")
    typer.echo(f"wrote {sidecar}")


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
