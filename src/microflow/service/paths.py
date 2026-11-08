"""Path policy for the service: what it may read and write, decided on resolved paths. Torch-free.

Every path a client supplies must be absolute and is resolved before any comparison, so symbolic
links and ``..`` are followed first: a link that leads outside a configured directory is refused
like any other outside path. Input paths must exist inside an input directory. Output directories
must lie strictly inside an output directory and must not be an existing file. Result files are
served only from a finished run, by relative name, and never when the name is hidden, the path
crosses a symbolic link, or the file kind is not a result kind.
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from microflow.service.errors import ServiceError

MAX_PATH_LENGTH = 4096
MAX_RELATIVE_LENGTH = 512
RESULT_SUFFIXES = frozenset({".csv", ".json", ".tif", ".tiff", ".png", ".svg", ".html", ".parquet"})
LISTING_LIMIT = 5000


@dataclass(frozen=True)
class ResultFile:
    path: str
    size_bytes: int


def resolve_input(text: object, roots: Sequence[Path]) -> Path:
    """An existing file or directory inside one of the input directories."""
    resolved = _resolve(_absolute(text))
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ServiceError(
            "path_not_allowed", "the path is outside the directories this service may read"
        )
    if not resolved.exists():
        raise ServiceError("not_found", "the requested path does not exist")
    return resolved


def resolve_output(text: object, roots: Sequence[Path]) -> Path:
    """A directory strictly inside one of the output directories. It may not exist yet."""
    resolved = _resolve(_absolute(text))
    if not any(resolved != root and resolved.is_relative_to(root) for root in roots):
        raise ServiceError(
            "path_not_allowed",
            "the output directory must be inside one of the directories this service may write to",
        )
    if resolved.exists() and not resolved.is_dir():
        raise ServiceError("bad_request", "the output path exists and is not a directory")
    return resolved


def resolve_result_file(run_dir: Path, relative: object) -> Path:
    """The file a client named inside a finished run directory. ``run_dir`` must be resolved."""
    if (
        not isinstance(relative, str)
        or not relative
        or len(relative) > MAX_RELATIVE_LENGTH
        or "\x00" in relative
        or "\\" in relative
        or relative.startswith("/")
    ):
        raise ServiceError("bad_request", "the file name is not valid")
    parts = relative.split("/")
    if any(part == "" for part in parts):
        raise ServiceError("bad_request", "the file name is not valid")
    if any(part in (".", "..") or part.startswith(".") for part in parts):
        raise ServiceError("path_not_allowed", "hidden and parent-relative names are not served")
    if Path(parts[-1]).suffix.lower() not in RESULT_SUFFIXES:
        raise ServiceError("path_not_allowed", "this kind of file is not served")
    current = run_dir
    for part in parts:
        current = current / part
        if os.path.islink(current):
            raise ServiceError("path_not_allowed", "symbolic links are not served")
    if not current.is_file():
        raise ServiceError("not_found", "no such result file")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(run_dir):
        raise ServiceError("path_not_allowed", "the file is outside its run directory")
    return resolved


def list_result_files(
    run_dir: Path, limit: int = LISTING_LIMIT
) -> tuple[tuple[ResultFile, ...], bool]:
    """Result files under a run directory, sorted by path. Returns the first ``limit`` and whether
    there were more. The walk visits names in sorted order, so the same directory always gives the
    same listing. It stops once it has ``limit + 1`` files, which bounds the work."""
    found: list[ResultFile] = []
    for current, dirs, files in os.walk(run_dir):
        base = Path(current)
        dirs[:] = sorted(
            name for name in dirs if not name.startswith(".") and not (base / name).is_symlink()
        )
        for name in sorted(files):
            full = base / name
            if name.startswith(".") or full.suffix.lower() not in RESULT_SUFFIXES:
                continue
            if full.is_symlink() or not full.is_file():
                continue
            relative = full.relative_to(run_dir).as_posix()
            found.append(ResultFile(path=relative, size_bytes=full.stat().st_size))
            if len(found) > limit:
                break
        if len(found) > limit:
            break
    ordered = sorted(found, key=lambda item: item.path)
    return tuple(ordered[:limit]), len(ordered) > limit


def redact(text: str, replacements: Mapping[str, str]) -> str:
    """Replace absolute server paths in a message with the labels clients are allowed to see."""
    for original, label in sorted(replacements.items(), key=lambda item: -len(item[0])):
        if original:
            text = text.replace(original, label)
    return text


def _absolute(text: object) -> Path:
    if not isinstance(text, str) or not text:
        raise ServiceError("bad_request", "a path is required")
    if len(text) > MAX_PATH_LENGTH or "\x00" in text:
        raise ServiceError("bad_request", "the path is too long or contains a null byte")
    path = Path(text)
    if not path.is_absolute():
        raise ServiceError("bad_request", "paths must be absolute")
    return path


def _resolve(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ServiceError("bad_request", "the path cannot be resolved") from exc
