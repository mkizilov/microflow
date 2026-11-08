"""Model checkpoints. Requires PyTorch.

A checkpoint is a dictionary holding the weights and everything needed to rebuild the model and to
check that it is used on the data it was trained with: architecture, normalisation statistics,
split, run identity, configuration and versions. It is written atomically, and it is loaded with
``weights_only=True``, so opening a checkpoint cannot run code.
"""

import os
from pathlib import Path
from typing import Any

import torch

from microflow.ml.runtime import CheckpointError

CHECKPOINT_FORMAT = "microflow.unet"
CHECKPOINT_SCHEMA = 1


def save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    payload = {
        **metadata,
        "format": CHECKPOINT_FORMAT,
        "schema_version": CHECKPOINT_SCHEMA,
        "model_state": model.state_dict(),
    }
    partial = path.with_name(path.name + ".partial")
    torch.save(payload, partial)
    os.replace(partial, path)


def load_checkpoint(path: Path, device: str = "cpu") -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointError(f"{path}: no such checkpoint")
    try:
        payload: Any = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:  # torch raises several unrelated types for damaged or foreign files
        raise CheckpointError(f"{path}: cannot be read as a checkpoint: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"{path}: not a MicroFlow U-Net checkpoint")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise CheckpointError(
            f"{path}: checkpoint schema {payload.get('schema_version')!r} is not supported; "
            f"expected {CHECKPOINT_SCHEMA}"
        )
    return payload
