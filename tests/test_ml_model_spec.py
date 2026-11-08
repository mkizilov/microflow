from typing import Any

import pytest

from microflow.ml.model_spec import minor_version, validate_metadata
from microflow.ml.runtime import ModelCompatibilityError

CURRENT = "0.1.0"


def _metadata(**changes: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "architecture": {"in_channels": 1, "base_channels": 8, "depth": 2},
        "normalization": {"mean": 100.0, "std": 50.0},
        "threshold": 0.5,
        "training_id": "abc123",
        "microflow_version": CURRENT,
        "training_calibrations": [[0.5, "um"]],
        "epoch": 3,
        "torch_version": "2.4.1",
    }
    metadata.update(changes)
    return metadata


def _accept(metadata: dict[str, Any], **kwargs: Any) -> Any:
    return validate_metadata(metadata, source="model.pt", microflow_version=CURRENT, **kwargs)


def test_a_matching_checkpoint_is_accepted_and_its_settings_are_read() -> None:
    spec = _accept(_metadata())
    assert spec.architecture == {"in_channels": 1, "base_channels": 8, "depth": 2}
    assert spec.normalization.mean == 100.0
    assert spec.normalization.std == 50.0
    assert spec.threshold == 0.5
    assert spec.training_calibrations == ((0.5, "um"),)
    assert spec.warnings == ()


def test_missing_declarations_are_named_in_the_refusal() -> None:
    metadata = _metadata()
    del metadata["training_calibrations"]
    with pytest.raises(ModelCompatibilityError, match="does not declare training_calibrations"):
        _accept(metadata)


def test_the_minor_version_must_match_while_patch_releases_are_accepted() -> None:
    assert _accept(_metadata(microflow_version="0.1.7")).microflow_version == "0.1.7"
    with pytest.raises(ModelCompatibilityError, match="minor versions must match"):
        _accept(_metadata(microflow_version="0.2.0"))
    with pytest.raises(ModelCompatibilityError, match="minor versions must match"):
        _accept(_metadata(microflow_version="1.1.0"))


@pytest.mark.parametrize(
    "architecture",
    [
        {"in_channels": 1, "base_channels": 0, "depth": 2},
        {"in_channels": 1, "base_channels": 8, "depth": 9},
        {"in_channels": 3, "base_channels": 8, "depth": 2},
        {"in_channels": 1, "base_channels": 8, "depth": 2, "dropout": 0.1},
        {"in_channels": 1, "base_channels": True, "depth": 2},
    ],
)
def test_unsupported_architectures_are_refused(architecture: dict[str, Any]) -> None:
    with pytest.raises(ModelCompatibilityError, match="architecture"):
        _accept(_metadata(architecture=architecture))


@pytest.mark.parametrize(
    "normalization",
    [
        {"mean": 100.0, "std": 0.0},
        {"mean": 100.0, "std": float("nan")},
        {"mean": 100.0},
    ],
)
def test_normalization_must_be_usable(normalization: dict[str, Any]) -> None:
    with pytest.raises(ModelCompatibilityError, match=r"normalization|finite|positive"):
        _accept(_metadata(normalization=normalization))


@pytest.mark.parametrize("threshold", [0.0, 1.0, 1.5, True])
def test_threshold_must_be_a_probability(threshold: Any) -> None:
    with pytest.raises(ModelCompatibilityError, match="threshold"):
        _accept(_metadata(threshold=threshold))


@pytest.mark.parametrize(
    "calibrations",
    [[[0.0, "um"]], [["x", "um"]], [[0.5]], "um", [[0.5, 3]]],
)
def test_training_calibrations_are_validated(calibrations: Any) -> None:
    with pytest.raises(ModelCompatibilityError, match="calibration"):
        _accept(_metadata(training_calibrations=calibrations))


def test_epoch_and_training_identity_are_checked() -> None:
    with pytest.raises(ModelCompatibilityError, match="epoch"):
        _accept(_metadata(epoch=-1))
    with pytest.raises(ModelCompatibilityError, match="training_id"):
        _accept(_metadata(training_id=""))


def test_a_different_pytorch_minor_version_only_warns() -> None:
    spec = validate_metadata(
        _metadata(torch_version="2.4.1"),
        source="model.pt",
        runtime_torch_version="2.5.0",
        microflow_version=CURRENT,
    )
    assert spec.warnings and "PyTorch 2.4.1" in spec.warnings[0]
    same = validate_metadata(
        _metadata(torch_version="2.4.1"),
        source="model.pt",
        runtime_torch_version="2.4.9",
        microflow_version=CURRENT,
    )
    assert same.warnings == ()


def test_minor_version_parsing() -> None:
    assert minor_version("0.1.0") == (0, 1)
    assert minor_version("2.4.1+cpu") == (2, 4)
    assert minor_version("unknown") is None
