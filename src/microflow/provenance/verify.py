"""Verify a finished batch run against the file hashes it recorded. Standard library only.

A run records two kinds of file hash. ``batch_manifest.json`` lists each input image's SHA-256
under ``provenance.inputs``. Each item's ``result.json`` lists the SHA-256 of every output that item
wrote.
Verification hashes each recorded file again and compares. It shows that the files on disk are the
ones the run recorded. It does not recompute the analysis, so a passing check is evidence of file
integrity, not of scientific correctness.

This module imports nothing from the scientific stack, so it runs wherever Python does.
"""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_FILE = "batch_manifest.json"
ITEMS_DIR = "items"
ITEM_RESULT = "result.json"


class VerificationError(RuntimeError):
    """The run cannot be verified: its manifest is missing or unreadable."""


@dataclass(frozen=True)
class FileCheck:
    kind: str
    name: str
    path: Path
    status: str


@dataclass(frozen=True)
class RunVerification:
    run_directory: Path
    run_status: str
    checks: tuple[FileCheck, ...]

    @property
    def failed(self) -> tuple[FileCheck, ...]:
        return tuple(check for check in self.checks if check.status != "ok")

    @property
    def ok(self) -> bool:
        return not self.failed


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run(run_directory: Path) -> RunVerification:
    manifest_path = run_directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise VerificationError(
            f"{run_directory}: no {MANIFEST_FILE}; only finished runs can be verified"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inputs = manifest["provenance"]["inputs"]
        status = str(manifest["status"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise VerificationError(f"{manifest_path}: cannot read the run manifest: {exc}") from exc
    checks = [
        _check("input", entry["name"], Path(entry["path"]), entry["sha256"]) for entry in inputs
    ]
    checks.extend(_output_checks(run_directory))
    return RunVerification(run_directory=run_directory, run_status=status, checks=tuple(checks))


def _output_checks(run_directory: Path) -> Iterator[FileCheck]:
    for result_path in sorted((run_directory / ITEMS_DIR).glob(f"*/{ITEM_RESULT}")):
        item: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
        for output_name, output in sorted(item.get("outputs", {}).items()):
            yield _check(
                "output",
                f"{item['name']}/{output_name}",
                result_path.parent / output["path"],
                output["sha256"],
            )


def _check(kind: str, name: str, path: Path, expected: str) -> FileCheck:
    if not path.is_file():
        return FileCheck(kind=kind, name=name, path=path, status="missing")
    status = "ok" if sha256_of(path) == expected else "modified"
    return FileCheck(kind=kind, name=name, path=path, status=status)
