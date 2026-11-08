"""The ``microflow ml`` command group.

``make-dataset``, ``splits``, ``fingerprint`` and ``compare`` run without PyTorch. ``train``,
``evaluate``, ``inspect`` and ``segment`` require the optional ``ml`` extra and fail with an install
hint without it.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn, cast, get_args

import numpy as np
import typer
from numpy.typing import NDArray

from microflow.config import ConfigError
from microflow.image_io import write_image
from microflow.loading import ImageLoadError, load_image
from microflow.ml.compare import compare_label_stacks
from microflow.ml.config import TrainingConfig, load_training_config
from microflow.ml.manifest import read_manifest
from microflow.ml.pipeline import segment_stack
from microflow.ml.preparation import digests_for, plan_for
from microflow.ml.runtime import MLError, require_torch
from microflow.ml.splitting import SPLIT_NAMES
from microflow.ml.synthetic import make_synthetic_dataset
from microflow.provenance.guard import OutputConflictError
from microflow.provenance.hashing import sha256_file
from microflow.segmentation.metrics import score_segmentation
from microflow.units import Area, Calibration, LengthUnit, area_to_pixels


def _length_unit(text: str) -> LengthUnit:
    units: tuple[str, ...] = get_args(LengthUnit)
    if text not in units:
        raise typer.BadParameter(f"unit must be one of {', '.join(units)}; got {text!r}")
    return cast(LengthUnit, text)


if TYPE_CHECKING:
    from microflow.ml.train import EpochRecord

ml_app = typer.Typer(
    help="Supervised segmentation with a small U-Net. train, evaluate, inspect and segment "
    "need the 'ml' extra.",
)

DEVICES = ("cpu", "cuda", "auto")
_FAILURES = (ConfigError, ImageLoadError, MLError, OutputConflictError, ValueError, OSError)


@ml_app.command("make-dataset")
def make_dataset_command(
    out_dir: Annotated[Path, typer.Argument(help="Directory to write images, masks and manifest.")],
    samples: Annotated[int, typer.Option(min=3, help="Number of original samples.")] = 6,
    images_per_sample: Annotated[
        int, typer.Option("--images-per-sample", min=1, help="Fields of view per sample.")
    ] = 2,
    size: Annotated[int, typer.Option(min=24, help="Image side length in pixels.")] = 64,
    seed: Annotated[int, typer.Option(min=0, help="Seed for the generator.")] = 0,
) -> None:
    """Write a deterministic synthetic dataset (images, masks and a manifest)."""
    try:
        manifest = make_synthetic_dataset(
            out_dir,
            samples=samples,
            images_per_sample=images_per_sample,
            size=size,
            seed=seed,
        )
    except (ValueError, OSError) as exc:
        _fail(exc)
    typer.echo(f"wrote {manifest}")


@ml_app.command("splits")
def splits_command(
    config_path: Annotated[Path, typer.Argument(help="Training YAML configuration.")],
) -> None:
    """Show which original samples fall in each split, without loading any pixels."""
    try:
        config = load_training_config(config_path)
        records = read_manifest(config.manifest)
        plan = plan_for(config, records, digests_for(records))
    except _FAILURES as exc:
        _fail(exc)
    typer.echo(f"seed {plan.seed}: {len(plan.assignment)} sample(s), {len(records)} image(s)")
    for name in SPLIT_NAMES:
        members = plan.members(name)
        images = sum(1 for record in records if plan.split_of(record.sample) == name)
        typer.echo(f"{name:<10} {len(members)} sample(s), {images} image(s): {', '.join(members)}")


@ml_app.command("train")
def train_command(
    config_path: Annotated[Path, typer.Argument(help="Training YAML configuration.")],
    device: Annotated[
        str | None, typer.Option(help="Override the configured device: cpu, cuda or auto.")
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace results of a different training run."),
    ] = False,
) -> None:
    """Train the U-Net, saving the best and the last checkpoint and the validation history."""
    try:
        config = _configured(config_path, device)
        require_torch()
        from microflow.ml.train import train

        result = train(config, overwrite=overwrite, progress=_print_epoch)
    except _FAILURES as exc:
        _fail(exc)
    typer.echo(
        f"best epoch {result.best_epoch} (validation Dice {result.best_validation_dice:.4f}); "
        f"checkpoints in {result.output_directory}"
    )


@ml_app.command("evaluate")
def evaluate_command(
    config_path: Annotated[Path, typer.Argument(help="The training configuration used.")],
    checkpoint: Annotated[Path, typer.Argument(help="A checkpoint written by ml train.")],
    device: Annotated[
        str | None, typer.Option(help="Override the configured device: cpu, cuda or auto.")
    ] = None,
) -> None:
    """Score a checkpoint on the held-out test samples and write per-image metrics."""
    try:
        config = _configured(config_path, device)
        require_torch()
        from microflow.ml.evaluate import evaluate

        result = evaluate(config, checkpoint)
    except _FAILURES as exc:
        _fail(exc)
    typer.echo(
        f"test: {len(result.images)} image(s), mean Dice {result.mean_dice:.4f}, "
        f"mean IoU {result.mean_iou:.4f}; written to {result.output_directory}"
    )


@ml_app.command("fingerprint")
def fingerprint_command(
    checkpoint: Annotated[Path, typer.Argument(help="A checkpoint file.")],
) -> None:
    """Print the SHA-256 of a checkpoint. A segmenter plugin configuration must name it."""
    if not checkpoint.is_file():
        _fail(FileNotFoundError(f"{checkpoint}: no such file"))
    typer.echo(f"{sha256_file(checkpoint)}  {checkpoint}")


@ml_app.command("inspect")
def inspect_command(
    checkpoint: Annotated[Path, typer.Argument(help="A checkpoint file.")],
) -> None:
    """Validate a checkpoint for this MicroFlow version and show what it declares."""
    try:
        require_torch()
        from microflow.ml.predict import load_model

        loaded = load_model(checkpoint)
    except _FAILURES as exc:
        _fail(exc)
    spec = loaded.spec
    calibrations = ", ".join(f"{size:g} {unit}/px" for size, unit in spec.training_calibrations)
    typer.echo(f"checkpoint     {checkpoint}")
    typer.echo(f"sha256         {loaded.sha256}")
    typer.echo(f"microflow      {spec.microflow_version} (PyTorch {spec.torch_version})")
    typer.echo(f"architecture   {spec.architecture}")
    typer.echo(f"threshold      {spec.threshold:g}")
    typer.echo(
        f"normalisation  mean {spec.normalization.mean:.6g}, std {spec.normalization.std:.6g}"
    )
    typer.echo(f"training px    {calibrations or 'none recorded (uncalibrated images)'}")
    typer.echo(f"training id    {spec.training_id}")
    typer.echo(f"epoch          {spec.epoch}")
    for warning in spec.warnings:
        typer.echo(f"warning: {warning}")
    typer.echo("compatible with this MicroFlow version")


@ml_app.command("segment")
def segment_command(
    checkpoint: Annotated[Path, typer.Argument(help="A checkpoint written by ml train.")],
    input_path: Annotated[Path, typer.Argument(help="Input TIFF or .npy image.")],
    output: Annotated[Path, typer.Option(help="Output label image (int32; 0 is background).")],
    device: Annotated[str, typer.Option(help="auto, cpu or cuda.")] = "auto",
    tile_size: Annotated[
        int,
        typer.Option(
            "--tile-size",
            min=8,
            help="Window size in pixels; a multiple of 2**depth of the model.",
        ),
    ] = 256,
    overlap: Annotated[
        int | None,
        typer.Option(help="Pixels shared by neighbouring windows. Default: the receptive radius."),
    ] = None,
    threshold: Annotated[
        float | None,
        typer.Option(help="Probability threshold. Default: the value recorded in the checkpoint."),
    ] = None,
    min_area: Annotated[
        float, typer.Option(help="Minimum object area in --area-unit; 0 keeps every object.")
    ] = 4.0,
    area_unit: Annotated[str, typer.Option(help="Unit for --min-area: nm, um, mm or m.")] = "um",
    exclude_border: Annotated[
        bool, typer.Option(help="Drop objects that touch the image edge.")
    ] = False,
    channel: Annotated[
        int | None, typer.Option(help="Channel index, for images with a C axis.")
    ] = None,
    fallback_to_cpu: Annotated[
        bool,
        typer.Option(
            "--fallback-to-cpu/--no-fallback-to-cpu",
            help="Move the rest of a run to CPU after a recoverable CUDA failure.",
        ),
    ] = True,
    pixel_size: Annotated[
        float | None,
        typer.Option(help="Pixel size; overrides the file's calibration. Needs --unit."),
    ] = None,
    unit: Annotated[
        str | None, typer.Option(help="Length unit for --pixel-size: nm, um, mm or m.")
    ] = None,
    reference: Annotated[
        Path | None,
        typer.Option(help="Ground-truth label image; reports Dice and IoU against it."),
    ] = None,
) -> None:
    """Segment objects with a trained model. Writes the same outputs as `microflow segment`,
    plus an .ml.json report with the model, tiling, device and per-plane quality."""
    try:
        image = load_image(input_path, calibration=_calibration(pixel_size, unit))
        data, axes = _single_channel(image.data, image.axes, channel)
        require_torch()
        from microflow.ml.predict import TiledSegmenter, load_model

        loaded = load_model(checkpoint)
        for warning in loaded.spec.warnings:
            typer.echo(f"warning: {warning}")
        segmenter = TiledSegmenter(
            loaded,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            threshold=threshold,
            fallback_to_cpu=fallback_to_cpu,
        )
        min_unit = _length_unit(area_unit)
        min_area_px = (
            0.0
            if min_area == 0
            else area_to_pixels(Area(value=min_area, unit=min_unit), image.calibration)
        )
        calibration = _calibration_tuple(image.calibration)
        labels, results = segment_stack(
            data,
            axes,
            lambda plane: segmenter.segment_plane(
                plane,
                calibration=calibration,
                min_area_px=min_area_px,
                exclude_border=exclude_border,
            ),
        )
        scores = None
        if reference is not None:
            scores = score_segmentation(labels, load_image(reference).data)
    except _FAILURES as exc:
        _fail(exc)

    write_image(output, labels, axes)
    sidecar = output.with_suffix(".segmentation.json")
    settings = {
        "method": "ml-unet",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": loaded.sha256,
        "device": segmenter.device,
        "tile_size": segmenter.tile_size,
        "overlap": segmenter.overlap,
        "threshold": segmenter.threshold,
        "min_object_area": {"value": min_area, "unit": area_unit},
        "exclude_border": exclude_border,
        "fallback_to_cpu": fallback_to_cpu,
    }
    _write_json(
        sidecar,
        {
            "source": str(input_path),
            "axes": axes,
            "object_counts": [result.quality.object_count for result in results],
            "thresholds": [segmenter.threshold] * len(results),
            "settings": settings,
            "scores": scores.model_dump() if scores is not None else None,
        },
    )
    report = output.with_suffix(".ml.json")
    _write_json(
        report,
        {
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": loaded.sha256,
                "training_id": loaded.spec.training_id,
                "microflow_version": loaded.spec.microflow_version,
                "torch_version": loaded.spec.torch_version,
                "epoch": loaded.spec.epoch,
                "warnings": list(loaded.spec.warnings),
            },
            "architecture": loaded.spec.architecture,
            "normalization": loaded.spec.normalization.to_dict(),
            "tiling": {
                "tile_size": segmenter.tile_size,
                "overlap": segmenter.overlap,
                "receptive_radius": segmenter.radius,
                "overlap_sufficient": segmenter.overlap >= segmenter.radius,
                "tiles_per_plane": [len(result.plan.tiles) for result in results],
            },
            "device": {
                "requested": device,
                "used": segmenter.device,
                "fallback_used": any(
                    "device_fallback" in result.quality.flags for result in results
                ),
            },
            "planes": [result.quality.to_dict() for result in results],
        },
    )
    typer.echo(f"wrote {output} ({sum(r.quality.object_count for r in results)} objects)")
    if scores is not None:
        typer.echo(f"Dice {scores.dice:.4f}  IoU {scores.iou:.4f}")
    for flag in sorted({flag for result in results for flag in result.quality.flags}):
        typer.echo(f"quality flag: {flag}")
    typer.echo(f"wrote {sidecar}")
    typer.echo(f"wrote {report}")


@ml_app.command("compare")
def compare_command(
    ml_labels: Annotated[Path, typer.Argument(help="A label image, for example from ml segment.")],
    reference_labels: Annotated[
        Path, typer.Argument(help="Another label image, for example from microflow segment.")
    ],
    output: Annotated[
        Path | None, typer.Option(help="Write the comparison as JSON to this path.")
    ] = None,
    match_iou: Annotated[
        float,
        typer.Option(
            "--match-iou",
            min=0.01,
            max=1.0,
            help="IoU needed for two objects to count as the same object.",
        ),
    ] = 0.5,
) -> None:
    """Compare two label images: overall Dice and IoU, and one-to-one object matches."""
    try:
        first = load_image(ml_labels)
        second = load_image(reference_labels)
        if first.axes != second.axes or first.shape != second.shape:
            raise ValueError(
                f"the label images differ: {first.axes} {first.shape} and "
                f"{second.axes} {second.shape}"
            )
        comparison = compare_label_stacks(first.data, second.data, first.axes, match_iou=match_iou)
    except _FAILURES as exc:
        _fail(exc)
    typer.echo(f"Dice {comparison.dice:.4f}  IoU {comparison.iou:.4f}")
    typer.echo(
        f"objects: first {comparison.objects_a}, second {comparison.objects_b}, matched "
        f"{comparison.matched_objects} (IoU >= {match_iou:g}); planes {comparison.planes}"
    )
    if output is not None:
        _write_json(
            output,
            {
                "ml_labels": str(ml_labels),
                "reference_labels": str(reference_labels),
                **comparison.to_dict(),
            },
        )
        typer.echo(f"wrote {output}")


def _configured(config_path: Path, device: str | None) -> TrainingConfig:
    config = load_training_config(config_path)
    if device is None:
        return config
    if device not in DEVICES:
        raise ValueError(f"--device must be one of {', '.join(DEVICES)}; got {device!r}")
    return config.model_copy(update={"device": device})


def _calibration(pixel_size: float | None, unit: str | None) -> Calibration | None:
    if pixel_size is None:
        return None
    if unit is None:
        raise ValueError("--pixel-size needs --unit")
    return Calibration.model_validate({"pixel_size": pixel_size, "unit": unit})


def _calibration_tuple(calibration: Calibration | None) -> tuple[float, str] | None:
    if calibration is None:
        return None
    return (float(calibration.pixel_size), calibration.unit)


def _single_channel(
    data: NDArray[np.generic], axes: str, channel: int | None
) -> tuple[NDArray[np.generic], str]:
    if "C" not in axes:
        if channel not in (None, 0):
            raise ValueError(f"the image has no channel axis, so --channel {channel} is invalid")
        return data, axes
    index = axes.index("C")
    count = data.shape[index]
    if channel is None:
        if count != 1:
            raise ValueError(f"the image has {count} channels; choose one with --channel")
        channel = 0
    if not 0 <= channel < count:
        raise ValueError(f"--channel {channel} is outside the {count} channel(s)")
    return np.take(data, channel, axis=index), axes.replace("C", "")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _print_epoch(record: "EpochRecord") -> None:
    typer.echo(
        f"epoch {record.epoch:>3}: train loss {record.train_loss:.4f}, "
        f"validation loss {record.validation_loss:.4f}, "
        f"validation Dice {record.validation_dice:.4f}, IoU {record.validation_iou:.4f}"
    )


def _fail(exc: BaseException) -> NoReturn:
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code=1) from exc
