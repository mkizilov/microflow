"""Versions of the software, the algorithms and the libraries that produced a result."""

import platform
from importlib import metadata

# Bump an entry when that algorithm's output for the same input and settings changes.
ALGORITHM_VERSIONS: dict[str, str] = {
    "loading": "1",
    "preprocessing": "1",
    "segmentation": "1",
    "quantification": "1",
    "quality_control": "1",
    "batch": "1",
}

# Libraries whose versions change the numbers themselves. They are part of an analysis identity.
COMPUTATIONAL_LIBRARIES = ("numpy", "scipy", "scikit-image", "tifffile")

# Every library recorded in a provenance record.
REPORTED_LIBRARIES = (
    *COMPUTATIONAL_LIBRARIES,
    "matplotlib",
    "pyarrow",
    "pydantic",
    "pyyaml",
    "typer",
)


def library_versions(names: tuple[str, ...] = REPORTED_LIBRARIES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def runtime_info() -> dict[str, str]:
    """Interpreter and platform. Host name and user are deliberately not recorded."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
