"""Output contracts for plugins: what MicroFlow requires of what a plugin returns.

Every check raises :class:`PluginError` with the plugin's name and the rule that was broken, so a
faulty plugin fails the item it ran on with an explanation. Nothing is silently converted,
truncated, or dropped.
"""

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from microflow.plugins.api import FEATURE_NAME_PATTERN, FeatureColumn, PluginError

# Columns that identify each row; a feature with one of these names would overwrite them.
RESERVED_FEATURE_COLUMNS = ("plane", "label")


def call[T](
    kind: str,
    name: str,
    stage: str,
    function: Callable[..., T],
    *args: Any,
) -> T:
    """Run one plugin method. Any exception it raises is re-raised, naming the plugin."""
    try:
        return function(*args)
    except PluginError:
        raise
    except (Exception, SystemExit) as exc:
        raise PluginError(
            f"{kind} plugin {name!r} failed during {stage}: {type(exc).__name__}: {exc}"
        ) from exc


def check_image(
    output: Any, expected_shape: tuple[int, ...], label: str
) -> npt.NDArray[np.float64]:
    array = np.asarray(output)
    if array.shape != expected_shape:
        raise PluginError(
            f"{label} returned shape {array.shape}; the input has shape {expected_shape}"
        )
    if array.dtype.kind not in "iuf":
        raise PluginError(f"{label} returned dtype {array.dtype}; a numeric array is required")
    as_float = array.astype(np.float64)
    if np.isinf(as_float).any():
        raise PluginError(f"{label} returned infinite values; mark invalid pixels as NaN instead")
    return as_float


def check_labels(output: Any, expected_shape: tuple[int, ...], label: str) -> npt.NDArray[np.int32]:
    array = np.asarray(output)
    if array.shape != expected_shape:
        raise PluginError(
            f"{label} returned shape {array.shape}; the plane has shape {expected_shape}"
        )
    if array.dtype.kind not in "iu":
        raise PluginError(f"{label} returned dtype {array.dtype}; integer labels are required")
    if array.size and array.min() < 0:
        raise PluginError(
            f"{label} returned negative labels; background is 0, objects are positive"
        )
    if array.size and array.max() > array.size:
        raise PluginError(
            f"{label} returned label {int(array.max())}, which exceeds the number of pixels "
            f"({array.size}); use labels 1, 2, 3, ..."
        )
    return array.astype(np.int32)


def check_feature_declaration(columns: Any, label: str) -> tuple[FeatureColumn, ...]:
    declared = tuple(columns)
    if not declared:
        raise PluginError(f"{label} declares no features")
    seen: set[str] = set()
    for column in declared:
        if not isinstance(column, FeatureColumn):
            raise PluginError(f"{label} declared an entry that is not a FeatureColumn")
        if not re.fullmatch(FEATURE_NAME_PATTERN, column.name):
            raise PluginError(
                f"{label} declared feature {column.name!r}; feature names are lower-case letters, "
                "digits and '_', starting with a letter"
            )
        if column.name in RESERVED_FEATURE_COLUMNS:
            raise PluginError(
                f"{label} declared feature {column.name!r}; "
                f"{' and '.join(repr(name) for name in RESERVED_FEATURE_COLUMNS)} "
                "are reserved for object identity"
            )
        if column.name in seen:
            raise PluginError(f"{label} declared feature {column.name!r} more than once")
        if not column.unit.strip() or not column.description.strip():
            raise PluginError(f"{label} must give a unit and a description for {column.name!r}")
        seen.add(column.name)
    return declared


def check_features(
    output: Any,
    object_labels: set[int],
    columns: Sequence[FeatureColumn],
    label: str,
) -> dict[int, dict[str, float]]:
    if not isinstance(output, Mapping):
        raise PluginError(f"{label} must return a mapping from object label to feature values")
    by_label: dict[int, Any] = {}
    for key, values in output.items():
        if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
            raise PluginError(f"{label} returned the key {key!r}; object labels must be integers")
        by_label[int(key)] = values
    returned = set(by_label)
    missing = sorted(object_labels - returned)
    if missing:
        raise PluginError(f"{label} returned no values for object(s) {missing[:10]}")
    unexpected = sorted(returned - object_labels)
    if unexpected:
        raise PluginError(
            f"{label} returned values for label(s) {unexpected[:10]}, which are not objects"
        )
    names = [column.name for column in columns]
    table: dict[int, dict[str, float]] = {}
    for key in sorted(returned):
        values = by_label[key]
        if not isinstance(values, Mapping):
            raise PluginError(f"{label} returned a non-mapping for object {key}")
        if set(values) != set(names):
            raise PluginError(
                f"{label} returned features {sorted(values)} for object {key}; "
                f"the declared features are {names}"
            )
        row: dict[str, float] = {}
        for name in names:
            value = values[name]
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise PluginError(
                    f"{label} returned {type(value).__name__} for {name!r} of object {key}; "
                    "a number is required"
                )
            number = float(value)
            if math.isinf(number):
                raise PluginError(
                    f"{label} returned an infinite value for {name!r} of object {key}; "
                    "use NaN for undefined values"
                )
            row[name] = number
        table[key] = row
    return table


def check_findings(output: Any, label: str) -> tuple[str, ...]:
    if isinstance(output, (str, bytes)) or not isinstance(output, Sequence):
        raise PluginError(f"{label} must return a sequence of text messages")
    messages: list[str] = []
    for item in output:
        if not isinstance(item, str) or not item.strip():
            raise PluginError(f"{label} returned an empty or non-text message")
        messages.append(item.strip())
    return tuple(messages)
