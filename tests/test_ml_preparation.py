from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from microflow.image_io import write_image
from microflow.ml.config import TrainingConfig
from microflow.ml.preparation import fit_normalization, prepare
from microflow.ml.runtime import LeakageError, MLError


def test_normalization_is_fitted_on_the_training_images_only(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    prepared = prepare(ml_config(ml_manifest))
    train_values = np.concatenate([pair.image.ravel() for pair in prepared.pairs["train"]])
    assert prepared.normalization.mean == pytest.approx(train_values.mean())
    assert prepared.normalization.std == pytest.approx(train_values.std())


def test_validation_and_test_pixels_cannot_move_the_normalization(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    config = ml_config(ml_manifest)
    before = prepare(config)
    held_out = next(
        record
        for record in before.records
        if before.plan.split_of(record.sample) in ("validation", "test")
    )
    shape = before.pairs[before.plan.split_of(held_out.sample)][0].image.shape
    write_image(held_out.image, np.full(shape, 60000, dtype=np.uint16))

    after = prepare(config)
    assert after.plan == before.plan
    assert after.normalization == before.normalization
    assert after.identity != before.identity


def test_training_pixels_do_move_the_normalization(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    config = ml_config(ml_manifest)
    before = prepare(config)
    training = next(
        record for record in before.records if before.plan.split_of(record.sample) == "train"
    )
    shape = before.pairs["train"][0].image.shape
    write_image(training.image, np.full(shape, 60000, dtype=np.uint16))
    assert prepare(config).normalization != before.normalization


def test_identity_ignores_location_and_device_but_follows_settings(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig], tmp_path: Path
) -> None:
    base = prepare(ml_config(ml_manifest)).identity
    moved = ml_config(
        ml_manifest,
        output_directory=tmp_path / "elsewhere",
        device="auto",
        num_workers=0,
    )
    assert prepare(moved).identity == base
    changed = ml_config(ml_manifest, training={"epochs": 5, "batch_size": 4})
    assert prepare(changed).identity != base


def test_leakage_across_samples_is_refused_before_any_split(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig], tmp_path: Path
) -> None:
    from microflow.ml.manifest import read_manifest

    records = read_manifest(ml_manifest)
    duplicate_of = records[0]
    duplicate = tmp_path / "duplicate.tif"
    duplicate.write_bytes(duplicate_of.image.read_bytes())
    manifest = ml_manifest.parent / "leaky.yaml"
    manifest.write_text(
        "samples:\n"
        + "".join(
            f"  - {{sample: {record.sample}, image: {record.image}, mask: {record.mask}}}\n"
            for record in records
        )
        + f"  - {{sample: extra-donor, image: {duplicate}, mask: {duplicate_of.mask}}}\n",
        encoding="utf-8",
    )
    with pytest.raises(LeakageError, match="same content"):
        prepare(ml_config(manifest))


def test_masks_must_be_binary_and_match_their_images(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    from microflow.ml.manifest import read_manifest

    record = read_manifest(ml_manifest)[0]
    write_image(record.mask, np.full((48, 48), 2, dtype=np.uint8))
    with pytest.raises(MLError, match="masks must hold 0"):
        prepare(ml_config(ml_manifest))

    write_image(record.mask, np.zeros((40, 40), dtype=np.uint8))
    with pytest.raises(MLError, match="differs from image shape"):
        prepare(ml_config(ml_manifest))


def test_non_finite_images_are_refused(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    from microflow.ml.manifest import read_manifest

    record = read_manifest(ml_manifest)[0]
    broken = np.full((48, 48), 100.0, dtype=np.float32)
    broken[0, 0] = np.nan
    write_image(record.image, broken)
    with pytest.raises(MLError, match="non-finite"):
        prepare(ml_config(ml_manifest))


def test_constant_training_images_have_no_defined_normalization() -> None:
    with pytest.raises(MLError, match="no intensity variation"):
        fit_normalization([np.ones((4, 4))])


def test_normalization_uses_the_population_mean_and_standard_deviation() -> None:
    normalization = fit_normalization([np.array([[0.0, 2.0], [4.0, 6.0]])])
    assert normalization.mean == pytest.approx(3.0)
    assert normalization.std == pytest.approx(np.sqrt(5.0))
    scaled = normalization.apply(np.array([[2.0, 4.0]]))
    np.testing.assert_allclose(scaled, [[-1 / np.sqrt(5), 1 / np.sqrt(5)]])
