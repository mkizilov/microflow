"""Held-out evaluation: score a checkpoint on the test samples. Requires PyTorch.

Evaluation refuses to run unless the checkpoint was made from this data and this configuration: the
split (which samples are held out), the architecture, the intensity normalisation and the training
run must all match. Otherwise the scores would not describe samples that training left out.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microflow.ml.checkpoint import load_checkpoint
from microflow.ml.config import TrainingConfig
from microflow.ml.inference import ImageResult, mean, resolve_device, score_pairs
from microflow.ml.preparation import (
    IDENTITY_KEY,
    RUN_FILE,
    PreparedRun,
    prepare,
    same_normalization,
)
from microflow.ml.runtime import CheckpointError
from microflow.ml.train import TEST_METRICS_FILE, TEST_SUMMARY_FILE
from microflow.ml.unet import build_unet
from microflow.provenance.guard import read_identity

_METRIC_COLUMNS = (
    "sample",
    "image",
    "loss",
    "dice",
    "iou",
    "predicted_pixels",
    "truth_pixels",
)


@dataclass(frozen=True)
class EvaluationResult:
    output_directory: Path
    images: tuple[ImageResult, ...]
    mean_dice: float
    mean_iou: float


def evaluate(config: TrainingConfig, checkpoint_path: Path) -> EvaluationResult:
    prepared = prepare(config)
    checkpoint = load_checkpoint(checkpoint_path)
    _check_matches(checkpoint, config, prepared, checkpoint_path)
    if checkpoint.get(IDENTITY_KEY) != prepared.identity:
        raise CheckpointError(
            f"{checkpoint_path} does not belong to the training run for the current data and "
            "configuration: an image, a mask or a setting has changed since training"
        )

    out = config.output_directory
    if read_identity(out / RUN_FILE, IDENTITY_KEY) != checkpoint.get(IDENTITY_KEY):
        raise CheckpointError(
            f"{checkpoint_path} does not belong to the training run recorded in {out}; use the "
            "configuration and output directory that produced it"
        )

    device = resolve_device(config.device)
    model = build_unet(config.architecture())
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    results = score_pairs(
        model,
        prepared.pairs["test"],
        prepared.normalization,
        device=device,
        multiple=2**config.model.depth,
        threshold=config.training.threshold,
        bce_weight=config.training.bce_weight,
    )
    mean_dice = mean([result.dice for result in results])
    mean_iou = mean([result.iou for result in results])
    _write_results(out, results, checkpoint_path, mean_dice, mean_iou)
    return EvaluationResult(
        output_directory=out,
        images=results,
        mean_dice=mean_dice,
        mean_iou=mean_iou,
    )


def _check_matches(
    checkpoint: dict[str, Any],
    config: TrainingConfig,
    prepared: PreparedRun,
    path: Path,
) -> None:
    if checkpoint.get("split") != prepared.plan.to_dict():
        raise CheckpointError(
            f"{path}: the split recorded in the checkpoint differs from the split the manifest "
            "and seed give now, so the test samples would not be the ones training held out"
        )
    if checkpoint.get("architecture") != config.architecture():
        raise CheckpointError(
            f"{path}: the architecture {checkpoint.get('architecture')} differs from the "
            f"configuration ({config.architecture()})"
        )
    recorded = checkpoint.get("normalization")
    if not isinstance(recorded, dict) or not same_normalization(recorded, prepared.normalization):
        raise CheckpointError(
            f"{path}: the intensity normalisation differs from the one recorded at training, so "
            "the data has changed since the checkpoint was made"
        )


def _write_results(
    out: Path,
    results: tuple[ImageResult, ...],
    checkpoint_path: Path,
    mean_dice: float,
    mean_iou: float,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / TEST_METRICS_FILE).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_METRIC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "sample": result.sample,
                    "image": str(result.image_path),
                    "loss": result.loss,
                    "dice": result.dice,
                    "iou": result.iou,
                    "predicted_pixels": result.predicted_pixels,
                    "truth_pixels": result.truth_pixels,
                }
            )
    summary = {
        "split": "test",
        "images": len(results),
        "samples": len({result.sample for result in results}),
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "checkpoint": str(checkpoint_path),
    }
    (out / TEST_SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
