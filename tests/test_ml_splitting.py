from pathlib import Path

import numpy as np
import pytest

from microflow.config import ConfigError
from microflow.loading import load_image
from microflow.ml.manifest import SampleImage, read_manifest
from microflow.ml.preparation import digests_for
from microflow.ml.runtime import LeakageError
from microflow.ml.splitting import SPLIT_NAMES, check_no_leakage, plan_split
from microflow.ml.synthetic import make_synthetic_dataset
from microflow.provenance.hashing import sha256_file

SAMPLES = [f"s{index:02d}" for index in range(10)]


def test_plan_is_reproducible_for_a_seed_and_changes_with_it() -> None:
    first = plan_split(SAMPLES, validation=0.2, test=0.2, seed=1)
    assert plan_split(SAMPLES, validation=0.2, test=0.2, seed=1) == first
    assert plan_split(SAMPLES, validation=0.2, test=0.2, seed=2) != first


def test_every_sample_lands_in_exactly_one_split_with_the_expected_sizes() -> None:
    plan = plan_split(SAMPLES, validation=0.2, test=0.3, seed=0)
    members = {name: plan.members(name) for name in SPLIT_NAMES}
    assert sorted(sum(members.values(), ())) == sorted(SAMPLES)
    assert len(members["test"]) == 3
    assert len(members["validation"]) == 2
    assert len(members["train"]) == 5
    for sample in SAMPLES:
        assert sample in plan.members(plan.split_of(sample))


def test_held_out_splits_are_never_empty_even_for_tiny_fractions() -> None:
    plan = plan_split(["a", "b", "c", "d"], validation=0.01, test=0.01, seed=0)
    assert plan.members("train")
    assert plan.members("validation")
    assert plan.members("test")


def test_too_few_samples_and_no_training_samples_are_refused() -> None:
    with pytest.raises(ValueError, match="at least 3 samples"):
        plan_split(["a", "b"], validation=0.2, test=0.2, seed=0)
    with pytest.raises(ValueError, match="no samples for training"):
        plan_split(["a", "b", "c"], validation=0.9, test=0.9, seed=0)


def test_identical_content_under_two_samples_is_refused(tmp_path: Path) -> None:
    mask = tmp_path / "mask.tif"
    first = SampleImage(sample="donor-a", image=tmp_path / "a.tif", mask=mask)
    second = SampleImage(sample="donor-b", image=tmp_path / "b.tif", mask=mask)
    digests = {first.image: "same", second.image: "same", mask: "mask"}
    with pytest.raises(LeakageError, match="give their images one sample name"):
        check_no_leakage([first, second], digests)


def test_identical_content_within_one_sample_is_allowed(tmp_path: Path) -> None:
    mask = tmp_path / "mask.tif"
    records = [
        SampleImage(sample="donor-a", image=tmp_path / "a.tif", mask=mask),
        SampleImage(sample="donor-a", image=tmp_path / "b.tif", mask=mask),
    ]
    check_no_leakage(records, {records[0].image: "same", records[1].image: "same", mask: "m"})


def test_manifest_reports_missing_duplicate_and_invalid_entries(tmp_path: Path) -> None:
    (tmp_path / "img.tif").write_bytes(b"x")
    (tmp_path / "msk.tif").write_bytes(b"x")
    manifest = tmp_path / "manifest.yaml"

    manifest.write_text(
        "samples:\n  - {sample: a1, image: nope.tif, mask: msk.tif}\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="image file not found"):
        read_manifest(manifest)

    manifest.write_text(
        "samples:\n"
        "  - {sample: a1, image: img.tif, mask: msk.tif}\n"
        "  - {sample: a2, image: img.tif, mask: msk.tif}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="listed more than once"):
        read_manifest(manifest)

    manifest.write_text(
        "samples:\n  - {sample: 'bad name!', image: img.tif, mask: msk.tif}\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="sample names use letters"):
        read_manifest(manifest)

    manifest.write_text(
        "samples:\n  - {sample: a1, image: img.tif, mask: img.tif}\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="cannot be its own mask"):
        read_manifest(manifest)

    manifest.write_text("samples: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="samples"):
        read_manifest(manifest)


def test_synthetic_dataset_is_deterministic_and_fields_share_their_donor(tmp_path: Path) -> None:
    first = read_manifest(
        make_synthetic_dataset(tmp_path / "one", samples=4, images_per_sample=2, size=32, seed=3)
    )
    second = read_manifest(
        make_synthetic_dataset(tmp_path / "two", samples=4, images_per_sample=2, size=32, seed=3)
    )
    assert [record.sample for record in first] == [record.sample for record in second]
    for left, right in zip(first, second, strict=True):
        assert sha256_file(left.image) == sha256_file(right.image)
        assert sha256_file(left.mask) == sha256_file(right.mask)
    assert len(first) == 8
    assert len({record.sample for record in first}) == 4


def test_synthetic_fields_have_distinct_content_and_binary_truth(tmp_path: Path) -> None:
    records = read_manifest(
        make_synthetic_dataset(tmp_path / "data", samples=3, images_per_sample=2, size=32)
    )
    digests = digests_for(records)
    check_no_leakage(records, digests)
    assert len({digests[record.image] for record in records}) == len(records)
    mask = np.asarray(load_image(records[0].mask, axes="YX").data)
    assert set(np.unique(mask).tolist()) <= {0, 1}
    assert mask.sum() > 0
