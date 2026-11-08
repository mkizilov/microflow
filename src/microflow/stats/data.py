"""Object-level measurements, and their assignment to biological samples.

Objects are the rows of a measurement table (one per segmented object, as written by
``microflow export``). They are summarised within each biological sample before any inference, so
the sample is the unit of analysis. This module reads the table, checks it against the design, and
produces one value per sample.
"""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from microflow.stats.design import StatsDesign
from microflow.stats.errors import StatsError

MISSING_TEXT = frozenset({"", "NA", "na", "nan", "NaN"})


@dataclass(frozen=True)
class ObjectTable:
    items: NDArray[np.str_]
    values: dict[str, NDArray[np.float64]]
    units: dict[str, str]

    def __post_init__(self) -> None:
        for measure, column in self.values.items():
            if column.shape != self.items.shape:
                raise StatsError(f"{measure}: {column.size} values for {self.items.size} objects")

    @property
    def objects(self) -> int:
        return int(self.items.size)


@dataclass(frozen=True)
class SampleValue:
    """One biological sample's summary of one measure. ``value`` is NaN when the sample has no
    valid object for the measure, in which case the sample is excluded from inference."""

    sample: str
    group: str
    items: int
    objects: int
    value: float


def read_measurements(
    path: Path, *, measures: Sequence[str], dictionary: Path | None = None
) -> ObjectTable:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if "item" not in fieldnames:
                raise StatsError(f"{path}: the table needs an 'item' column naming each image")
            missing = [name for name in measures if name not in fieldnames]
            if missing:
                raise StatsError(f"{path}: no column for measure(s) {', '.join(missing)}")
            items: list[str] = []
            columns: dict[str, list[float]] = {name: [] for name in measures}
            for number, row in enumerate(reader, start=2):
                if None in row or None in row.values():
                    raise StatsError(
                        f"{path}: row {number} has a different number of fields from the header"
                    )
                if not row["item"].strip():
                    raise StatsError(f"{path}: row {number} has no item name")
                items.append(row["item"])
                for name in measures:
                    columns[name].append(_number(row[name], path, number, name))
    except OSError as exc:
        raise StatsError(f"{path}: cannot read measurements: {exc.strerror}") from exc
    units = _units(dictionary) if dictionary is not None else {}
    return ObjectTable(
        items=np.asarray(items, dtype=str),
        values={name: np.asarray(values, dtype=np.float64) for name, values in columns.items()},
        units={name: units.get(name, "") for name in measures},
    )


def check_assignments(table: ObjectTable, design: StatsDesign) -> None:
    """Every image in the table must belong to a sample, and every design image must have
    measurements. Either mismatch would silently change the samples or the sample sizes, so both are
    refused."""
    assigned = {item for group in design.groups.values() for s in group.samples for item in s.items}
    present = set(table.items.tolist())
    unassigned = sorted(present - assigned)
    if unassigned:
        raise StatsError(
            f"{len(unassigned)} image(s) in the measurements belong to no sample; assign them in "
            f"the design or remove them from the table: {', '.join(unassigned[:10])}"
        )
    unmeasured = sorted(assigned - present)
    if unmeasured:
        raise StatsError(
            f"{len(unmeasured)} image(s) in the design have no measurements: "
            f"{', '.join(unmeasured[:10])}"
        )


def sample_values(table: ObjectTable, design: StatsDesign, measure: str) -> tuple[SampleValue, ...]:
    """Summarise each sample's valid objects for one measure, in design order."""
    column = table.values[measure]
    finite = np.isfinite(column)
    summarise = np.median if design.per_sample == "median" else np.mean
    values: list[SampleValue] = []
    for group_name, group in design.groups.items():
        for definition in group.samples:
            used = np.isin(table.items, definition.items) & finite
            objects = int(used.sum())
            value = float(summarise(column[used])) if objects else float("nan")
            values.append(
                SampleValue(
                    sample=definition.sample,
                    group=group_name,
                    items=len(definition.items),
                    objects=objects,
                    value=value,
                )
            )
    return tuple(values)


def group_objects(
    table: ObjectTable, design: StatsDesign, measure: str, group: str
) -> NDArray[np.float64]:
    """Every valid object value in one group. Used only for descriptive statistics and plots."""
    items = [item for s in design.groups[group].samples for item in s.items]
    column = table.values[measure]
    return column[np.isin(table.items, items) & np.isfinite(column)]


def _number(text: str, path: Path, row: int, name: str) -> float:
    if text.strip() in MISSING_TEXT:
        return float("nan")
    try:
        value = float(text)
    except ValueError as exc:
        raise StatsError(f"{path}: row {row}, {name}: {text!r} is not a number") from exc
    if not np.isfinite(value) and value == value:
        raise StatsError(f"{path}: row {row}, {name}: infinite values are not supported")
    return value


def _units(path: Path) -> dict[str, str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return {row["name"]: row["unit"] for row in csv.DictReader(handle)}
    except OSError as exc:
        raise StatsError(f"{path}: cannot read the data dictionary: {exc.strerror}") from exc
    except KeyError as exc:
        raise StatsError(f"{path}: the data dictionary needs name and unit columns") from exc
