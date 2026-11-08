"""Training loop, validation, checkpoints and the run record. Requires PyTorch.

Training reads only the training split for gradients and normalisation. The validation split is
scored after each epoch and selects the best checkpoint. The test split is not read here at all;
it is used once, by ``microflow ml evaluate``.
"""

import csv
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from microflow import __version__
from microflow.ml.checkpoint import save_checkpoint
from microflow.ml.config import TrainingConfig
from microflow.ml.inference import ImageResult, mean, resolve_device, score_pairs
from microflow.ml.losses import segmentation_loss
from microflow.ml.preparation import IDENTITY_KEY, RUN_FILE, PreparedRun, prepare
from microflow.ml.splitting import SPLIT_NAMES
from microflow.ml.torch_data import PatchDataset
from microflow.ml.unet import build_unet
from microflow.provenance.guard import prepare_directory
from microflow.provenance.record import utc_now

CONFIG_FILE = "config.json"
SPLIT_FILE = "split.json"
HISTORY_FILE = "history.csv"
SUMMARY_FILE = "summary.json"
BEST_FILE = "best.pt"
LAST_FILE = "last.pt"
TEST_METRICS_FILE = "test_metrics.csv"
TEST_SUMMARY_FILE = "test_summary.json"
MANAGED_FILES = (
    RUN_FILE,
    CONFIG_FILE,
    SPLIT_FILE,
    HISTORY_FILE,
    SUMMARY_FILE,
    BEST_FILE,
    LAST_FILE,
    TEST_METRICS_FILE,
    TEST_SUMMARY_FILE,
)
_HISTORY_COLUMNS = ("epoch", "train_loss", "validation_loss", "validation_dice", "validation_iou")
# A plain str: torch.__version__ is a str subclass, which weights_only loading refuses to unpickle.
_TORCH_VERSION = str(torch.__version__)


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_dice: float
    validation_iou: float


@dataclass(frozen=True)
class TrainingResult:
    output_directory: Path
    best_epoch: int
    best_validation_dice: float
    history: tuple[EpochRecord, ...]
    training_id: str


def train(
    config: TrainingConfig,
    *,
    overwrite: bool = False,
    progress: Callable[[EpochRecord], None] | None = None,
) -> TrainingResult:
    prepared = prepare(config)
    out = config.output_directory
    prepare_directory(
        out,
        identity=prepared.identity,
        identity_file=RUN_FILE,
        identity_key=IDENTITY_KEY,
        managed=MANAGED_FILES,
        overwrite=overwrite,
        label="training results",
    )
    _write_json(
        out / RUN_FILE,
        {
            IDENTITY_KEY: prepared.identity,
            "microflow_version": __version__,
            "torch_version": _TORCH_VERSION,
            "started_utc": utc_now(),
        },
    )
    (out / CONFIG_FILE).write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _write_json(out / SPLIT_FILE, _split_audit(prepared))

    device = resolve_device(config.device)
    _seed_everything(config.seed)
    model = build_unet(config.architecture()).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    patches = PatchDataset(
        prepared.pairs["train"],
        normalization=prepared.normalization,
        size=config.patches.size,
        per_image=config.patches.per_image,
        augment=config.augment,
        seed=config.seed,
    )
    multiple = 2**config.model.depth
    history: list[EpochRecord] = []
    best: EpochRecord | None = None
    for epoch in range(config.training.epochs):
        patches.set_epoch(epoch)
        train_loss = _train_one_epoch(model, optimizer, patches, device, config, epoch)
        scores = score_pairs(
            model,
            prepared.pairs["validation"],
            prepared.normalization,
            device=device,
            multiple=multiple,
            threshold=config.training.threshold,
            bce_weight=config.training.bce_weight,
        )
        model.train()
        record = _epoch_record(epoch, train_loss, scores)
        history.append(record)
        if progress is not None:
            progress(record)
        if best is None or record.validation_dice > best.validation_dice:
            best = record
            save_checkpoint(out / BEST_FILE, model, _checkpoint_metadata(config, prepared, record))

    last = history[-1]
    save_checkpoint(out / LAST_FILE, model, _checkpoint_metadata(config, prepared, last))
    _write_history(out / HISTORY_FILE, history)
    assert best is not None  # config requires at least one epoch
    _write_json(
        out / SUMMARY_FILE,
        {
            IDENTITY_KEY: prepared.identity,
            "epochs": config.training.epochs,
            "best_epoch": best.epoch,
            "best_validation_dice": best.validation_dice,
            "device": str(device),
            "torch_version": _TORCH_VERSION,
            "microflow_version": __version__,
        },
    )
    return TrainingResult(
        output_directory=out,
        best_epoch=best.epoch,
        best_validation_dice=best.validation_dice,
        history=tuple(history),
        training_id=prepared.identity,
    )


def _train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    patches: PatchDataset,
    device: torch.device,
    config: TrainingConfig,
    epoch: int,
) -> float:
    generator = torch.Generator()
    generator.manual_seed(config.seed + epoch)
    loader = DataLoader(
        patches,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model.train()
    total = 0.0
    seen = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        logits: torch.Tensor = model(images)
        loss = segmentation_loss(logits, masks, config.training.bce_weight)
        loss.backward()
        optimizer.step()
        count = int(images.shape[0])
        total += float(loss.item()) * count
        seen += count
    return total / seen


def _epoch_record(epoch: int, train_loss: float, scores: tuple[ImageResult, ...]) -> EpochRecord:
    return EpochRecord(
        epoch=epoch,
        train_loss=train_loss,
        validation_loss=mean([score.loss for score in scores]),
        validation_dice=mean([score.dice for score in scores]),
        validation_iou=mean([score.iou for score in scores]),
    )


def _seed_everything(seed: int) -> None:
    # NumPy draws come from seeded default_rng generators, so only Python and torch need seeding.
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _checkpoint_metadata(
    config: TrainingConfig, prepared: PreparedRun, record: EpochRecord
) -> dict[str, Any]:
    return {
        IDENTITY_KEY: prepared.identity,
        "architecture": config.architecture(),
        "normalization": prepared.normalization.to_dict(),
        "threshold": config.training.threshold,
        "seed": config.seed,
        "split": prepared.plan.to_dict(),
        "epoch": record.epoch,
        "validation_dice": record.validation_dice,
        "validation_iou": record.validation_iou,
        "config": config.model_dump(mode="json"),
        "microflow_version": __version__,
        "torch_version": _TORCH_VERSION,
    }


def _split_audit(prepared: PreparedRun) -> dict[str, Any]:
    """Which samples and which image files each split holds, with their content hashes."""
    return {
        "seed": prepared.plan.seed,
        "samples": prepared.plan.to_dict()["splits"],
        "images": {
            name: [
                {"sample": pair.sample, "image": str(pair.image_path), "sha256": pair.image_sha256}
                for pair in prepared.pairs[name]
            ]
            for name in SPLIT_NAMES
        },
    }


def _write_history(path: Path, history: list[EpochRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HISTORY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in history:
            writer.writerow(
                {
                    "epoch": record.epoch,
                    "train_loss": record.train_loss,
                    "validation_loss": record.validation_loss,
                    "validation_dice": record.validation_dice,
                    "validation_iou": record.validation_iou,
                }
            )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
