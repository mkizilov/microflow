"""Inference with trained models: receptive field, tiled and whole-plane prediction, determinism,
model compatibility and the device policy. They run on CPU and are skipped without PyTorch."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from microflow import __version__  # noqa: E402
from microflow.ml.checkpoint import save_checkpoint  # noqa: E402
from microflow.ml.config import TrainingConfig  # noqa: E402
from microflow.ml.model_spec import CHECKPOINT_FORMAT, CHECKPOINT_SCHEMA, ModelSpec  # noqa: E402
from microflow.ml.predict import LoadedModel, TiledSegmenter, load_model  # noqa: E402
from microflow.ml.preparation import Normalization  # noqa: E402
from microflow.ml.runtime import CheckpointError, MLError, ModelCompatibilityError  # noqa: E402
from microflow.ml.tiling import pad_to_multiple, receptive_radius  # noqa: E402
from microflow.ml.train import train  # noqa: E402
from microflow.ml.unet import UNet  # noqa: E402

pytestmark = pytest.mark.ml

NORMALIZATION = Normalization(mean=100.0, std=40.0)


def _random_model(base_channels: int = 4, depth: int = 2, seed: int = 0) -> UNet:
    torch.manual_seed(seed)
    model = UNet(base_channels=base_channels, depth=depth)
    model.eval()
    return model


def _loaded(model: UNet, **spec_changes: Any) -> LoadedModel:
    spec = ModelSpec(
        architecture={"in_channels": 1, "base_channels": 4, "depth": 2},
        normalization=NORMALIZATION,
        threshold=0.5,
        training_id="test-run",
        microflow_version=__version__,
        torch_version=str(torch.__version__),
        training_calibrations=(),
        epoch=0,
    )
    return LoadedModel(
        spec=replace(spec, **spec_changes), model=model, path=Path("random.pt"), sha256="0" * 64
    )


def _whole_plane(model: UNet, plane: np.ndarray, multiple: int = 4) -> np.ndarray:
    """Reference: one forward pass over the whole padded plane, cropped back."""
    rows, cols = plane.shape
    padded = pad_to_multiple(NORMALIZATION.apply(plane), multiple)
    with torch.no_grad():
        batch = torch.as_tensor(padded, dtype=torch.float32)[None, None]
        probabilities = torch.sigmoid(model(batch))[0, 0].numpy()
    return np.asarray(probabilities[:rows, :cols])


def _metadata(**changes: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "architecture": {"in_channels": 1, "base_channels": 4, "depth": 2},
        "normalization": {"mean": 100.0, "std": 40.0},
        "threshold": 0.5,
        "training_id": "test-run",
        "microflow_version": __version__,
        "training_calibrations": [[0.5, "um"]],
        "epoch": 1,
        "seed": 0,
        "torch_version": str(torch.__version__),
    }
    metadata.update(changes)
    return metadata


@pytest.mark.parametrize("depth", [1, 2])
def test_measured_input_support_stays_within_the_receptive_radius(depth: int) -> None:
    model = _random_model(base_channels=2, depth=depth)
    size = 48 if depth == 1 else 96
    centre = size // 2
    x = torch.randn(1, 1, size, size, requires_grad=True)
    model(x)[0, 0, centre, centre].backward()
    rows, cols = torch.nonzero(x.grad[0, 0].abs() > 0, as_tuple=True)
    reach = max(int((rows - centre).abs().max()), int((cols - centre).abs().max()))
    assert 0 < reach <= receptive_radius(depth)


def test_tiled_inference_matches_whole_plane_inference_when_overlap_covers_the_field() -> None:
    model = _random_model()
    plane = np.random.default_rng(0).normal(100.0, 30.0, size=(100, 90))
    segmenter = TiledSegmenter(_loaded(model), device="cpu", tile_size=96, fallback_to_cpu=False)
    tiled = segmenter.segment_plane(plane).probabilities
    assert segmenter.overlap == receptive_radius(2)
    np.testing.assert_allclose(tiled, _whole_plane(model, plane), rtol=0, atol=1e-5)


def test_insufficient_overlap_changes_the_result_near_tile_seams() -> None:
    model = _random_model()
    plane = np.random.default_rng(0).normal(100.0, 30.0, size=(100, 90))
    segmenter = TiledSegmenter(
        _loaded(model), device="cpu", tile_size=96, overlap=0, fallback_to_cpu=False
    )
    tiled = segmenter.segment_plane(plane).probabilities
    assert np.abs(tiled - _whole_plane(model, plane)).max() > 1e-4


def test_outputs_have_the_input_shape_and_repeat_exactly() -> None:
    segmenter = TiledSegmenter(_loaded(_random_model()), device="cpu", tile_size=96)
    plane = np.random.default_rng(1).normal(100.0, 30.0, size=(37, 23))
    first = segmenter.segment_plane(plane)
    second = segmenter.segment_plane(plane)
    assert first.probabilities.shape == (37, 23)
    assert first.labels.shape == (37, 23)
    assert first.labels.dtype == np.int32
    np.testing.assert_array_equal(first.probabilities, second.probabilities)
    np.testing.assert_array_equal(first.labels, second.labels)


def test_calibration_mismatch_with_training_is_reported_on_the_plane() -> None:
    segmenter = TiledSegmenter(
        _loaded(_random_model(), training_calibrations=((0.5, "um"),)), device="cpu"
    )
    plane = np.random.default_rng(2).normal(100.0, 30.0, size=(24, 24))
    result = segmenter.segment_plane(plane, calibration=(0.25, "um"))
    assert "calibration_mismatch" in result.quality.flags


def test_device_policy_auto_resolves_and_explicit_cuda_is_refused_without_cuda() -> None:
    loaded = _loaded(_random_model())
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert TiledSegmenter(loaded, device="auto").device == expected
    if not torch.cuda.is_available():
        with pytest.raises(MLError, match="CUDA is not available"):
            TiledSegmenter(loaded, device="cuda")


def test_a_recoverable_device_failure_moves_the_run_to_cpu_and_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _random_model()
    plane = np.random.default_rng(4).normal(100.0, 30.0, size=(40, 40))
    reference = TiledSegmenter(_loaded(model), device="cpu", tile_size=96).segment_plane(plane)

    segmenter = TiledSegmenter(_loaded(model), device="cpu", tile_size=96, fallback_to_cpu=True)
    # Stand in for an accelerator: fallback is only enabled when the primary device is not the CPU.
    segmenter._fallback_enabled = True

    def failing(window: np.ndarray) -> np.ndarray:
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(segmenter, "_primary_window", failing)
    result = segmenter.segment_plane(plane)
    np.testing.assert_array_equal(result.probabilities, reference.probabilities)
    assert "device_fallback" in result.quality.flags
    assert "device_fallback" not in reference.quality.flags


def test_a_valid_checkpoint_loads_with_its_declared_settings(tmp_path: Path) -> None:
    model = _random_model()
    path = tmp_path / "good.pt"
    save_checkpoint(path, model, _metadata())
    loaded = load_model(path)
    assert loaded.spec.training_id == "test-run"
    assert loaded.spec.training_calibrations == ((0.5, "um"),)
    assert len(loaded.sha256) == 64
    for (name, original), (_, copy) in zip(
        model.state_dict().items(), loaded.model.state_dict().items(), strict=True
    ):
        assert torch.equal(original, copy), name


def test_a_checkpoint_from_another_microflow_minor_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "old.pt"
    save_checkpoint(path, _random_model(), _metadata(microflow_version="0.0.9"))
    with pytest.raises(ModelCompatibilityError, match="minor versions must match"):
        load_model(path)


def test_weights_that_do_not_fit_the_declared_architecture_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.pt"
    save_checkpoint(path, _random_model(base_channels=8), _metadata())
    with pytest.raises(CheckpointError, match="do not match the declared architecture"):
        load_model(path)


def test_checkpoints_from_the_previous_schema_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "previous.pt"
    payload = {
        **_metadata(),
        "format": CHECKPOINT_FORMAT,
        "schema_version": CHECKPOINT_SCHEMA - 1,
        "model_state": _random_model().state_dict(),
    }
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="schema"):
        load_model(path)


def test_a_checkpoint_missing_a_required_declaration_is_refused(tmp_path: Path) -> None:
    metadata = _metadata()
    del metadata["training_calibrations"]
    path = tmp_path / "incomplete.pt"
    save_checkpoint(path, _random_model(), metadata)
    with pytest.raises(ModelCompatibilityError, match="does not declare training_calibrations"):
        load_model(path)


def test_a_trained_checkpoint_segments_planes_end_to_end(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    config = ml_config(ml_manifest, training={"epochs": 1, "batch_size": 4})
    result = train(config)
    loaded = load_model(config.output_directory / "best.pt")
    assert loaded.spec.training_id == result.training_id
    plane = np.random.default_rng(3).normal(100.0, 30.0, size=(48, 48))
    segmenter = TiledSegmenter(loaded, device="cpu", tile_size=32, overlap=8)
    segmented = segmenter.segment_plane(plane)
    assert segmented.probabilities.shape == (48, 48)
    assert "overlap_below_receptive_field" in segmented.quality.flags
