import importlib
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile


@pytest.fixture
def write_tiff(tmp_path: Path) -> Callable[..., Path]:
    """Write a synthetic TIFF under tmp_path. Keyword arguments go to ``tifffile.imwrite``."""

    def _write(name: str, data: np.ndarray, **kwargs: Any) -> Path:
        path = tmp_path / name
        tifffile.imwrite(path, data, **kwargs)
        return path

    return _write


@pytest.fixture
def plugin_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., None]]:
    """Install fake distributions with entry points on a private sys.path entry.

    The layout matches what pip or uv write, so entry-point discovery runs as it would for a real
    installation. Modules imported from the site are removed again after the test.
    """
    site = tmp_path / "plugin-site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))

    def install(
        dist: str,
        groups: dict[str, dict[str, str]],
        modules: dict[str, str] | None = None,
    ) -> None:
        for filename, source in (modules or {}).items():
            (site / filename).write_text(textwrap.dedent(source), encoding="utf-8")
        info = site / f"{dist}-1.0.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {dist}\nVersion: 1.0.0\n", encoding="utf-8"
        )
        lines: list[str] = []
        for group, entries in groups.items():
            lines.append(f"[{group}]")
            lines.extend(f"{name} = {target}" for name, target in entries.items())
            lines.append("")
        (info / "entry_points.txt").write_text("\n".join(lines), encoding="utf-8")
        importlib.invalidate_caches()

    yield install
    for name, module in list(sys.modules.items()):
        if str(getattr(module, "__file__", "") or "").startswith(str(site)):
            del sys.modules[name]


@pytest.fixture
def stack_cyx() -> np.ndarray:
    """Deterministic (2, 8, 8) uint16 stack whose values differ per channel and pixel."""
    return (np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8) * 37).astype(np.uint16)
