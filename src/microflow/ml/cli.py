"""The ``microflow ml`` command group.

``make-dataset`` and ``splits`` run without PyTorch. ``train`` and ``evaluate`` require the
optional ``ml`` extra and fail with an install hint without it.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer

from microflow.config import ConfigError
from microflow.loading import ImageLoadError
from microflow.ml.config import TrainingConfig, load_training_config
from microflow.ml.manifest import read_manifest
from microflow.ml.preparation import digests_for, plan_for
from microflow.ml.runtime import MLError, require_torch
from microflow.ml.splitting import SPLIT_NAMES
from microflow.ml.synthetic import make_synthetic_dataset
from microflow.provenance.guard import OutputConflictError

if TYPE_CHECKING:
    from microflow.ml.train import EpochRecord

ml_app = typer.Typer(
    help="Supervised segmentation with a small U-Net. train and evaluate need the 'ml' extra.",
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


def _configured(config_path: Path, device: str | None) -> TrainingConfig:
    config = load_training_config(config_path)
    if device is None:
        return config
    if device not in DEVICES:
        raise ValueError(f"--device must be one of {', '.join(DEVICES)}; got {device!r}")
    return config.model_copy(update={"device": device})


def _print_epoch(record: "EpochRecord") -> None:
    typer.echo(
        f"epoch {record.epoch:>3}: train loss {record.train_loss:.4f}, "
        f"validation loss {record.validation_loss:.4f}, "
        f"validation Dice {record.validation_dice:.4f}, IoU {record.validation_iou:.4f}"
    )


def _fail(exc: BaseException) -> NoReturn:
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code=1) from exc
