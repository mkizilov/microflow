"""Command-line interface for MicroFlow."""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from pydantic import ValidationError

from microflow import __version__
from microflow.config import (
    ConfigError,
    SegmentationConfig,
    QualityControlConfig,
    load_config,
    load_quality_config,
    load_segmentation_config,
)
from microflow.dataset import inspect_dataset
from microflow.image_io import write_image
from microflow.loading import ImageLoadError, LoadedImage, load_image
from microflow.preprocessing import PreprocessingError, preprocess
from microflow.preprocessing.steps import load_steps
from microflow.reporting.figures import render_figures
from microflow.reporting.html import render_html
from microflow.reporting.preview import first_plane, write_before_after_png
from microflow.reporting.qc import build_quality_report, inputs_from_images
from microflow.sample_data import write_sample_dataset
from microflow.quantification import IntensitySource, MeasurementError, measure_plane
from microflow.quantification.records import Source
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

    write_image(output, result.data, result.axes)
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

    write_image(output, result.labels, result.axes)
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


@app.command("measure")
def measure_command(
    labels_path: Annotated[
        Path, typer.Argument(help="Label image from `microflow segment`, one Y-X plane.")
    ],
    raw: Annotated[
        Path, typer.Option(help="Raw intensity image: axes YX, or one channel axis C.")
    ],
    output: Annotated[Path, typer.Option(help="Output CSV with one row per object.")],
    corrected: Annotated[
        Path | None,
        typer.Option(
            help="Corrected image on the same grid, e.g. from `microflow preprocess` "
            "without normalize."
        ),
    ] = None,
    pixel_size: Annotated[
        float | None,
        typer.Option(help="Pixel size; overrides the file's calibration. Needs --unit."),
    ] = None,
    unit: Annotated[
        str | None, typer.Option(help="Length unit for --pixel-size: nm, um, mm or m.")
    ] = None,
    require_calibration: Annotated[
        bool,
        typer.Option(
            "--require-calibration",
            help="Fail rather than report pixel units when the image has no calibration.",
        ),
    ] = False,
) -> None:
    """Measure geometry and intensity of every labelled object and write a CSV table.

    Geometry appears once per object. Intensity columns are prefixed by source (raw or
    corrected) and channel. docs/measurements.md defines every column.
    """
    warning: str | None = None
    try:
        raw_image = load_image(raw, calibration=_calibration_option(pixel_size, unit))
        labels = load_image(labels_path)
        if labels.axes != "YX":
            raise MeasurementError(f"{labels_path}: labels have axes {labels.axes!r}; expected YX")
        if require_calibration and raw_image.calibration is None:
            raise MeasurementError(f"{raw}: no pixel calibration; give --pixel-size and --unit")
        raw_source = _intensity_source("raw", raw_image)
        sources = [raw_source]
        if corrected is not None:
            corrected_image = load_image(corrected)
            if corrected_image.axes != raw_image.axes or corrected_image.shape != raw_image.shape:
                raise MeasurementError(
                    f"{corrected}: axes {corrected_image.axes!r} and shape {corrected_image.shape} "
                    f"do not match the raw image ({raw_image.axes!r}, {raw_image.shape})"
                )
            warning = _check_corrected_provenance(corrected)
            sources.append(_intensity_source("corrected", corrected_image, raw_source.channels))
        table = measure_plane(labels.data, sources, raw_image.calibration)
    except (ImageLoadError, ConfigError, ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if warning is not None:
        typer.echo(warning, err=True)
    table.write_csv(output)
    typer.echo(f"wrote {output} ({len(table.objects)} objects, length unit: {table.length_unit})")


def _intensity_source(
    name: Source, image: LoadedImage, channels: tuple[str, ...] | None = None
) -> IntensitySource:
    if not set(image.axes) <= {"C", "Y", "X"}:
        raise MeasurementError(
            f"{name} image has axes {image.axes!r}; measurement supports YX or one channel axis C"
        )
    order = [image.axes.index("Y"), image.axes.index("X")]
    if "C" in image.axes:
        order.insert(0, image.axes.index("C"))
    planes = np.transpose(np.asarray(image.data, dtype=np.float64), order)
    if "C" not in image.axes:
        planes = planes[np.newaxis]
    names = channels or image.channels or tuple(f"c{index}" for index in range(planes.shape[0]))
    return IntensitySource(name=name, planes=planes, channels=names)


def _check_corrected_provenance(path: Path) -> str | None:
    """Refuse normalised images, which would distort intensities. Warn if unverifiable."""
    sidecar = path.with_suffix(".provenance.json")
    if not sidecar.exists():
        return (
            f"warning: {path.name} has no provenance sidecar, "
            "so it cannot be verified as un-normalised"
        )
    stages = json.loads(sidecar.read_text(encoding="utf-8")).get("stages", [])
    if any(stage.get("stage") == "normalize" for stage in stages):
        raise MeasurementError(
            f"{path.name} was normalised ({sidecar.name} lists a normalize stage). "
            "Normalisation rescales intensities, so measure before normalising."
        )
    return None


@app.command("qc-report")
def qc_report_command(
    input_path: Annotated[Path, typer.Argument(help="Raw image (TIFF or .npy).")],
    labels_path: Annotated[
        Path, typer.Option("--labels", help="Label image from `microflow segment`.")
    ],
    output: Annotated[Path, typer.Option(help="Output self-contained HTML report.")],
    preprocessed: Annotated[
        Path | None, typer.Option(help="Preprocessed image on the same grid (optional).")
    ] = None,
    qc_config: Annotated[
        Path | None, typer.Option("--qc-config", help="YAML with quality-control thresholds.")
    ] = None,
    png_dir: Annotated[
        Path | None, typer.Option(help="Also write each figure as a PNG into this directory.")
    ] = None,
    fail_on_flags: Annotated[
        bool,
        typer.Option(
            "--fail-on-flags", help="Exit with status 2 if any result is flagged (for CI)."
        ),
    ] = False,
    pixel_size: Annotated[
        float | None,
        typer.Option(help="Pixel size; overrides the file's calibration. Needs --unit."),
    ] = None,
    unit: Annotated[
        str | None, typer.Option(help="Length unit for --pixel-size: nm, um, mm or m.")
    ] = None,
) -> None:
    """Build a quality-control report: figures, segmentation summary and flagged results."""
    try:
        raw_image = load_image(input_path, calibration=_calibration_option(pixel_size, unit))
        labels = load_image(labels_path)
        pre = load_image(preprocessed) if preprocessed is not None else None
        settings = (
            load_quality_config(qc_config) if qc_config is not None else QualityControlConfig()
        )
        inputs = inputs_from_images(
            raw_image,
            labels,
            settings,
            preprocessed=pre,
            title=f"Quality control: {input_path.name}",
        )
        report = build_quality_report(inputs)
        figures = render_figures(inputs, report)
        output.write_text(render_html(report, figures), encoding="utf-8")
        if png_dir is not None:
            png_dir.mkdir(parents=True, exist_ok=True)
            for name, data in figures.items():
                (png_dir / f"{name}.png").write_bytes(data)
    except (ImageLoadError, ConfigError, ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    for flag in report.flags:
        typer.echo(f"flag {flag.code} [{flag.plane}]: {flag.message}")
    typer.echo(
        f"wrote {output} ({len(report.flags)} flag(s), {len(report.planes)} plane(s))"
    )
    if fail_on_flags and report.flags:
        raise typer.Exit(code=2)


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
