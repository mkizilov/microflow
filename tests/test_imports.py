"""Every module in the package must import.

A broken module-level annotation or name is an error here, not a skip. The module list comes from
the source tree, not from importing packages, so collection itself imports nothing. A module that
needs an optional third-party package that is not installed (PyTorch, FastAPI, uvicorn, httpx) is
skipped with the package's name. Any other failure, including a NameError from a bad annotation,
fails the test.
"""

import importlib
from pathlib import Path

import pytest

import microflow

OPTIONAL_PACKAGES = {"torch", "fastapi", "starlette", "uvicorn", "httpx"}
PACKAGE_ROOT = Path(microflow.__file__).resolve().parent


def _module_names() -> list[str]:
    names = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        parts = list(path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


@pytest.mark.parametrize("name", _module_names())
def test_every_module_imports(name: str) -> None:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        if missing in OPTIONAL_PACKAGES:
            pytest.skip(f"needs the optional package {missing}")
        raise
