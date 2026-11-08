"""PyTorch tests: network, losses, patches, checkpoints, training and evaluation. Run on CPU.

They are skipped when PyTorch is not installed. The GPU test is skipped unless CUDA is available.
"""

import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from microflow.image_io import write_image  # noqa: E402
from microflow.ml import train as train_module  # noqa: E402
from microflow.ml.checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    load_checkpoint,
    save_checkpoint,
)
from microflow.ml.config import TrainingConfig  # noqa: E402
from microflow.ml.evaluate import evaluate  # noqa: E402
from microflow.ml.inference import resolve_device  # noqa: E402
from microflow.ml.losses import segmentation_loss, soft_dice_loss  # noqa: E402
from microflow.ml.preparation import ImagePair, prepare  # noqa: E402
from microflow.ml.runtime import CheckpointError, MLError  # noqa: E402
from microflow.ml.torch_data import PatchDataset  # noqa: E402
from microflow.ml.train import train  # noqa: E402
from microflow.ml.unet import UNet, build_unet  # noqa: E402
from microflow.provenance.guard import OutputConflictError  # noqa: E402

pytestmark = pytest.mark.ml

RECORD_FIELDS = ("train_loss", "validation_loss", "validation_dice", "validation_iou")


def test_unet_keeps_the_input_size_and_has_one_upsampling_stage_per_level() -> None:
    model = UNet(base_channels=4, depth=2)
    assert len(model.upsamples) == 2
    assert len(model.decoders) == 2
    logits = model(torch.zeros(2, 1, 32, 48))
    assert tuple(logits.shape) == (2, 1, 32, 48)


def test_unet_refuses_sizes_it_cannot_halve_evenly() -> None:
    with pytest.raises(ValueError, match="multiples of 4"):
        UNet(base_channels=4, depth=2)(torch.zeros(1, 1, 30, 32))


def test_dice_loss_is_zero_for_a_perfect_prediction_and_near_one_when_inverted() -> None:
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:5, 3:6] = 1.0
    assert soft_dice_loss(target * 60 - 30, target).item() == pytest.approx(0.0, abs=1e-6)
    assert soft_dice_loss((1 - target) * 60 - 30, target).item() > 0.9


def test_segmentation_loss_interpolates_between_bce_and_dice() -> None:
    logits = torch.linspace(-2, 2, 16).reshape(1, 1, 4, 4)
    target = (torch.arange(16).reshape(1, 1, 4, 4) % 3 == 0).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target).item()
    dice = soft_dice_loss(logits, target).item()
    assert segmentation_loss(logits, target, 1.0).item() == pytest.approx(bce)
    assert segmentation_loss(logits, target, 0.0).item() == pytest.approx(dice)
    assert segmentation_loss(logits, target, 0.5).item() == pytest.approx(0.5 * (bce + dice))


def test_checkpoint_round_trips_weights_and_metadata(tmp_path: Path) -> None:
    model = UNet(base_channels=4, depth=2)
    path = tmp_path / "model.pt"
    architecture = {"in_channels": 1, "base_channels": 4, "depth": 2}
    save_checkpoint(path, model, {"architecture": architecture, "threshold": 0.5})

    loaded = load_checkpoint(path)
    assert loaded["architecture"] == architecture
    assert loaded["schema_version"] == CHECKPOINT_SCHEMA
    restored = build_unet(loaded["architecture"])
    restored.load_state_dict(loaded["model_state"])
    for (name, original), (_, copy) in zip(
        model.state_dict().items(), restored.state_dict().items(), strict=True
    ):
        assert torch.equal(original, copy), name
    assert not list(tmp_path.glob("*.partial"))


def test_damaged_or_foreign_checkpoints_are_refused(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="no such checkpoint"):
        load_checkpoint(tmp_path / "missing.pt")

    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="cannot be read"):
        load_checkpoint(junk)

    foreign = tmp_path / "foreign.pt"
    torch.save({"weights": 1}, foreign)
    with pytest.raises(CheckpointError, match="not a MicroFlow U-Net checkpoint"):
        load_checkpoint(foreign)

    stale = tmp_path / "stale.pt"
    torch.save(
        {"format": "microflow.unet", "schema_version": CHECKPOINT_SCHEMA + 1, "model_state": {}},
        stale,
    )
    with pytest.raises(CheckpointError, match="schema"):
        load_checkpoint(stale)


def test_patches_are_reproducible_per_index_and_change_with_the_epoch(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    config = ml_config(ml_manifest)
    prepared = prepare(config)
    dataset = PatchDataset(
        prepared.pairs["train"],
        normalization=prepared.normalization,
        size=32,
        per_image=2,
        augment=config.augment,
        seed=0,
    )
    assert len(dataset) == 2 * len(prepared.pairs["train"])
    image, mask = dataset[3]
    again, _ = dataset[3]
    assert tuple(image.shape) == (1, 32, 32)
    assert torch.equal(image, again)
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
    dataset.set_epoch(1)
    assert not torch.equal(dataset[3][0], image)


def test_patches_cannot_be_larger_than_the_images(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    config = ml_config(ml_manifest)
    prepared = prepare(config)
    pairs: Sequence[ImagePair] = prepared.pairs["train"]
    with pytest.raises(ValueError, match="smaller than the patch size"):
        PatchDataset(
            pairs,
            normalization=prepared.normalization,
            size=64,
            per_image=1,
            augment=config.augment,
            seed=0,
        )


def test_training_writes_checkpoints_history_and_a_split_audit(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    config = ml_config(ml_manifest, training={"epochs": 2, "batch_size": 4})
    result = train(config)
    out = config.output_directory
    for name in ("best.pt", "last.pt", "history.csv", "split.json", "run.json", "summary.json"):
        assert (out / name).is_file(), name
    assert [record.epoch for record in result.history] == [0, 1]
    for record in result.history:
        assert all(np.isfinite(getattr(record, field)) for field in RECORD_FIELDS)

    best = load_checkpoint(out / "best.pt")
    assert best["epoch"] == result.best_epoch
    assert best["validation_dice"] == pytest.approx(result.best_validation_dice)
    assert best["split"] == prepare(config).plan.to_dict()


def test_training_gives_gradients_only_training_samples_and_scores_only_validation_samples(
    ml_manifest: Path,
    ml_config: Callable[..., TrainingConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_for_training: set[str] = set()
    seen_for_validation: set[str] = set()
    real_dataset = train_module.PatchDataset
    real_score = train_module.score_pairs

    def dataset_spy(pairs: Sequence[ImagePair], **kwargs: Any) -> PatchDataset:
        seen_for_training.update(pair.sample for pair in pairs)
        return real_dataset(pairs, **kwargs)

    def score_spy(
        model: UNet, pairs: Sequence[ImagePair], normalization: Any, **kwargs: Any
    ) -> Any:
        seen_for_validation.update(pair.sample for pair in pairs)
        return real_score(model, pairs, normalization, **kwargs)

    monkeypatch.setattr(train_module, "PatchDataset", dataset_spy)
    monkeypatch.setattr(train_module, "score_pairs", score_spy)
    config = ml_config(ml_manifest)
    plan = prepare(config).plan

    train(config)

    assert seen_for_training == set(plan.members("train"))
    assert seen_for_validation == set(plan.members("validation"))
    assert not (seen_for_training | seen_for_validation) & set(plan.members("test"))


def test_the_split_audit_puts_each_sample_and_each_image_in_one_split(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    config = ml_config(ml_manifest)
    train(config)
    audit = json.loads((config.output_directory / "split.json").read_text(encoding="utf-8"))
    split_of_sample: dict[str, str] = {}
    for split, samples in audit["samples"].items():
        for sample in samples:
            assert sample not in split_of_sample
            split_of_sample[sample] = split
    hashes: set[str] = set()
    for split, images in audit["images"].items():
        for item in images:
            assert split_of_sample[item["sample"]] == split
            assert item["sha256"] not in hashes
            hashes.add(item["sha256"])


def test_the_same_seed_reproduces_the_training_history(
    tmp_path: Path, ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    first = train(ml_config(ml_manifest, output_directory=tmp_path / "first"))
    second = train(ml_config(ml_manifest, output_directory=tmp_path / "second"))
    left = [[getattr(record, field) for field in RECORD_FIELDS] for record in first.history]
    right = [[getattr(record, field) for field in RECORD_FIELDS] for record in second.history]
    np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)
    assert first.best_epoch == second.best_epoch
    assert first.training_id == second.training_id


def test_a_different_run_is_not_overwritten_without_permission(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    train(ml_config(ml_manifest))
    changed = ml_config(ml_manifest, training={"epochs": 3, "batch_size": 4})
    with pytest.raises(OutputConflictError, match="different analysis or data set"):
        train(changed)
    assert train(changed, overwrite=True).best_epoch >= 0


def test_evaluation_scores_only_the_held_out_test_samples(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    config = ml_config(ml_manifest)
    train(config)
    evaluation = evaluate(config, config.output_directory / "best.pt")

    prepared = prepare(config)
    test_samples = set(prepared.plan.members("test"))
    with (config.output_directory / "test_metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == sum(1 for record in prepared.records if record.sample in test_samples)
    assert {row["sample"] for row in rows} == test_samples
    summary = json.loads((config.output_directory / "test_summary.json").read_text("utf-8"))
    assert summary["split"] == "test"
    assert summary["images"] == len(rows)
    assert evaluation.mean_dice == pytest.approx(summary["mean_dice"])
    assert 0.0 <= evaluation.mean_dice <= 1.0


def test_evaluation_refuses_a_checkpoint_made_from_another_split(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    config = ml_config(ml_manifest)
    train(config)
    other = ml_config(ml_manifest, split={"validation": 0.4, "test": 0.2})
    with pytest.raises(CheckpointError, match="split recorded in the checkpoint"):
        evaluate(other, config.output_directory / "best.pt")


def test_evaluation_refuses_once_held_out_images_have_changed(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    config = ml_config(ml_manifest)
    train(config)
    prepared = prepare(config)
    test_samples = set(prepared.plan.members("test"))
    record = next(item for item in prepared.records if item.sample in test_samples)
    write_image(record.image, np.full((48, 48), 7, dtype=np.uint16))
    with pytest.raises(CheckpointError, match="does not belong to the training run"):
        evaluate(config, config.output_directory / "best.pt")


def test_requesting_cuda_without_a_cuda_device_is_a_clear_error() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    with pytest.raises(MLError, match="CUDA is not available"):
        resolve_device("cuda")
    assert resolve_device("auto") == torch.device("cpu")
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_one_epoch_trains_on_cuda(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    result = train(ml_config(ml_manifest, device="cuda", training={"epochs": 1, "batch_size": 4}))
    assert result.best_epoch == 0
