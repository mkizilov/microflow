"""Comparing configurations: which settings differ, and which of them change the analysis."""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

ABSENT = "<absent>"


class Change(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    old: Any
    new: Any
    affects_analysis: bool


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Map every leaf of a nested mapping or list to a dotted path such as ``a.b[0].c``."""
    if isinstance(value, dict):
        if not value:
            return {prefix: {}}
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    if isinstance(value, list):
        if not value:
            return {prefix: []}
        flat = {}
        for index, item in enumerate(value):
            flat.update(flatten(item, f"{prefix}[{index}]"))
        return flat
    return {prefix: value}


def diff_configs(
    old: dict[str, Any], new: dict[str, Any], *, analysis_sections: tuple[str, ...]
) -> tuple[Change, ...]:
    before = flatten(old)
    after = flatten(new)
    changes: list[Change] = []
    for path in sorted(set(before) | set(after)):
        previous = before.get(path, ABSENT)
        current = after.get(path, ABSENT)
        if previous == current:
            continue
        top = re.split(r"[.\[]", path, maxsplit=1)[0]
        changes.append(
            Change(
                path=path,
                old=previous,
                new=current,
                affects_analysis=top in analysis_sections,
            )
        )
    return tuple(changes)
