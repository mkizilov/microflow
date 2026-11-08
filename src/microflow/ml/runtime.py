"""Errors of the ML module, and the single check for the optional PyTorch dependency.

Nothing in this module imports PyTorch at load time, so configuration, splitting and dataset
preparation work on a machine without it.
"""

from importlib import import_module
from types import ModuleType


class MLError(RuntimeError):
    """A training or evaluation run cannot proceed as configured."""


class MLDependencyError(MLError):
    """PyTorch is not installed."""


class LeakageError(MLError):
    """Two splits would share an original sample, or the same image content under two samples."""


class CheckpointError(MLError):
    """A checkpoint is missing, malformed, or does not belong to the configuration given."""


def require_torch() -> ModuleType:
    try:
        return import_module("torch")
    except ImportError as exc:
        raise MLDependencyError(
            "PyTorch is required for training and evaluation. Install the optional extra with "
            f"'uv sync --extra ml' ({exc})"
        ) from exc
