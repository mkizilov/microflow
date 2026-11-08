"""Column-typed tables and their CSV, Parquet and data-dictionary writers.

Parquet needs pyarrow, which is imported only when a Parquet file is written. CSV output uses
LF line endings and the shortest round-trip form of each float, so identical data gives identical
bytes.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DTYPES = ("string", "int64", "float64")


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    unit: str
    description: str


@dataclass(frozen=True)
class Table:
    columns: tuple[ColumnSpec, ...]
    data: dict[str, list[Any]]

    @property
    def length(self) -> int:
        return len(self.data[self.columns[0].name]) if self.columns else 0


def write_csv(path: Path, table: Table) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([spec.name for spec in table.columns])
        for index in range(table.length):
            writer.writerow([table.data[spec.name][index] for spec in table.columns])


def write_dictionary(path: Path, table: Table) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["name", "dtype", "unit", "description"])
        for spec in table.columns:
            writer.writerow([spec.name, spec.dtype, spec.unit, spec.description])


def write_parquet(path: Path, table: Table, metadata: dict[str, str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow_types = {"string": pa.string(), "int64": pa.int64(), "float64": pa.float64()}
    fields = [
        pa.field(
            spec.name,
            arrow_types[spec.dtype],
            metadata={"unit": spec.unit, "description": spec.description},
        )
        for spec in table.columns
    ]
    arrays = [
        pa.array(table.data[spec.name], type=arrow_types[spec.dtype]) for spec in table.columns
    ]
    schema = pa.schema(fields, metadata=metadata)
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path)
