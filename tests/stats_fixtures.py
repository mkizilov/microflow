"""Small hand-checkable studies for the statistics tests.

Two groups, ``control`` and ``treated``, with two samples each (``c1``, ``c2``, ``t1``, ``t2``).
Each sample has two images, and each image has known object areas, chosen so that every sample mean
and group statistic can be stated exactly.
"""

import csv
from pathlib import Path
from typing import Any

import yaml

# Object areas per image. Sample means: c1 = 10, c2 = 12, t1 = 16, t2 = 18.
AREAS: dict[str, list[float]] = {
    "c1_a": [8.0, 10.0],
    "c1_b": [12.0],
    "c2_a": [11.0, 13.0],
    "c2_b": [12.0],
    "t1_a": [15.0, 17.0],
    "t1_b": [16.0],
    "t2_a": [18.0],
    "t2_b": [17.0, 19.0],
}
SAMPLES: dict[str, dict[str, list[str]]] = {
    "control": {"c1": ["c1_a", "c1_b"], "c2": ["c2_a", "c2_b"]},
    "treated": {"t1": ["t1_a", "t1_b"], "t2": ["t2_a", "t2_b"]},
}
DICTIONARY = (("name", "unit"), ("area_um2", "um2"), ("eccentricity", "1"))


def write_measurements(directory: Path, areas: dict[str, list[float]] = AREAS) -> Path:
    path = directory / "measurements.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["item", "label", "area_um2", "eccentricity"])
        for item, values in areas.items():
            for label, area in enumerate(values, start=1):
                writer.writerow([item, label, area, 0.5])
    return path


def write_dictionary(directory: Path) -> Path:
    path = directory / "data_dictionary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(DICTIONARY)
    return path


def design_mapping(
    directory: Path,
    *,
    groups: dict[str, dict[str, list[str]]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    chosen = SAMPLES if groups is None else groups
    mapping: dict[str, Any] = {
        "name": "fixture",
        "measurements": "measurements.csv",
        "data_dictionary": "data_dictionary.csv",
        "output_directory": "results",
        "measures": ["area_um2"],
        "per_sample": "mean",
        "groups": {
            name: {
                "samples": [{"sample": sample, "items": items} for sample, items in samples.items()]
            }
            for name, samples in chosen.items()
        },
        "comparisons": [["control", "treated"]],
        "bootstrap": {"resamples": 1000, "confidence": 0.95, "seed": 0},
        "permutation": {"resamples": 1000, "max_exact": 1000, "seed": 0},
        "correction": "bh",
        "min_samples_per_group": 2,
        "plots": False,
    }
    mapping.update(overrides)
    return mapping


def write_design(directory: Path, mapping: dict[str, Any], name: str = "design.yaml") -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return path
