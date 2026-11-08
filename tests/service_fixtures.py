"""Shared helpers for the service tests: a project tree, batch configuration files, and a runner
that stands in for the batch engine.

The fake runner can be held open until a test releases it, reports how many runs overlap, honours
cancellation the way the batch engine does, and can write result files. That makes concurrency and
cancellation tests deterministic instead of timing-dependent.
"""

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from microflow.service.jobs import RunControl, RunOutcome
from microflow.service.settings import ServiceSettings

TOKEN = "t" * 40
COUNTS = {"ok": 1, "failed": 0, "cancelled": 0, "resumed": 0}


@dataclass(frozen=True)
class Tree:
    root: Path
    inputs: Path
    outputs: Path
    state: Path


def make_tree(base: Path) -> Tree:
    inputs = base / "input"
    outputs = base / "output"
    inputs.mkdir(parents=True)
    outputs.mkdir()
    return Tree(root=base, inputs=inputs, outputs=outputs, state=base / "state")


def settings_for(tree: Tree, **overrides: Any) -> ServiceSettings:
    values: dict[str, Any] = {
        "state_directory": tree.state,
        "input_roots": (tree.inputs,),
        "output_roots": (tree.outputs,),
    }
    values.update(overrides)
    return ServiceSettings.model_validate(values)


def write_images(tree: Tree, names: tuple[str, ...] = ("a.tif", "b.tif")) -> list[Path]:
    """Empty placeholder files. Validation checks that images exist, not what they contain."""
    folder = tree.inputs / "data"
    folder.mkdir(exist_ok=True)
    paths = []
    for name in names:
        path = folder / name
        path.write_bytes(b"placeholder")
        paths.append(path)
    return paths


def batch_mapping(
    tree: Tree, *, output: str = "run-1", images: tuple[str, ...] = ("a.tif", "b.tif"), **extra: Any
) -> dict[str, Any]:
    paths = write_images(tree, images)
    mapping: dict[str, Any] = {
        "name": "service-test",
        "dataset": {
            "images": [
                {"name": path.stem, "path": str(path)} for path in paths
            ]
        },
        "output_directory": str(tree.outputs / output),
        "workers": 1,
    }
    mapping.update(extra)
    return mapping


def write_batch_config(
    directory: Path, mapping: Mapping[str, Any], name: str = "batch.yaml"
) -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(dict(mapping), sort_keys=False), encoding="utf-8")
    return path


class FakeRunner:
    """Stands in for ``run_batch``. ``hold`` keeps each run open until ``release`` is called."""

    def __init__(
        self,
        *,
        hold: bool = False,
        status: str = "complete",
        error: BaseException | None = None,
        write: Mapping[str, str] | None = None,
    ) -> None:
        self.gate = threading.Event()
        if not hold:
            self.gate.set()
        self.status = status
        self.error = error
        self.write = dict(write or {})
        self.calls = 0
        self.active = 0
        self.peak = 0
        self.started = threading.Event()
        self._lock = threading.Lock()

    def release(self) -> None:
        self.gate.set()

    def __call__(self, config: Any, control: RunControl) -> RunOutcome:
        with self._lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        self.started.set()
        try:
            control.progress(1, 2)
            while not self.gate.wait(0.01):
                if control.cancelled():
                    counts = dict(COUNTS, cancelled=1)
                    return RunOutcome(status="cancelled", counts=counts, items=1, workers=1)
            if self.error is not None:
                raise self.error
            for relative, text in self.write.items():
                target = config.output_directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            return RunOutcome(status=self.status, counts=dict(COUNTS), items=1, workers=1)
        finally:
            with self._lock:
                self.active -= 1


def wait_for(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    """Poll a condition until it holds. Raises if it does not hold within the timeout."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met within the timeout")
        time.sleep(0.01)
