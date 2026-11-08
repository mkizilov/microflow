"""Protection against overwriting results that belong to another analysis or data set.

A directory that holds results records their identity in a small JSON file. Before writing, the
caller names the files it manages. If any of them exist and the recorded identity differs (or is
missing), the write is refused unless the caller asks to overwrite. Overwriting removes only the
managed files, so unrelated files in the same directory are never touched.
"""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path


class OutputConflictError(RuntimeError):
    """A directory already holds results that this write would replace."""


def read_identity(path: Path, key: str) -> str | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get(key)
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def prepare_directory(
    directory: Path,
    *,
    identity: str,
    identity_file: str,
    identity_key: str,
    managed: Sequence[str],
    overwrite: bool,
    label: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    present = [name for name in managed if (directory / name).exists()]
    if not present:
        return
    recorded = read_identity(directory / identity_file, identity_key)
    if recorded == identity:
        return
    if not overwrite:
        if recorded is None:
            raise OutputConflictError(
                f"{directory} already holds {label} with no recorded identity "
                f"({', '.join(present)}); pass overwrite to replace them, "
                "or choose another directory"
            )
        raise OutputConflictError(
            f"{directory} holds {label} from a different analysis or data set "
            f"(recorded {recorded[:12]}, current {identity[:12]}); pass overwrite to replace "
            "them, or choose another directory"
        )
    for name in present:
        remove_path(directory / name)


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
