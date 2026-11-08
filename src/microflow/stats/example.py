"""A synthetic two-condition example in the export format, for trying the statistics end to end.

The values are generated, not measured. The only designed effect is on area: treated objects have a
1.4-fold median area. Eccentricity and raw intensity do not depend on the condition, so they serve
as
null measures. Biological variation is between donors, and imaging variation is between fields
within a
donor, so the analysis should use four donors per condition rather than the 16 fields or several
hundred objects. Generate the data with ``microflow stats example DIRECTORY``. The design file
``examples/stats/design.yaml`` names the same items.
"""

import csv
from pathlib import Path

import numpy as np

CONDITIONS = ("control", "treated")
TREATMENT_AREA_LOG_EFFECT = float(np.log(1.4))
DONORS_PER_CONDITION = 4
FIELDS_PER_DONOR = 2
DONOR_SD = 0.06
FIELD_SD = 0.03
OBJECT_SD_AREA = 0.3
OBJECT_SD_INTENSITY = 0.12
HEADER = (
    "item",
    "plane",
    "label",
    "area_um2",
    "eccentricity",
    "raw_c0_mean",
    "raw_c0_valid_pixels",
)
DICTIONARY = (
    ("item", "string", "", "image the object belongs to"),
    ("plane", "string", "", "plane that the row belongs to"),
    ("label", "int64", "1", "object label within its plane (0 is background)"),
    ("area_um2", "float64", "um2", "object area (pixels x pixel area)"),
    ("eccentricity", "float64", "1", "0 for a disc, approaching 1 for a line"),
    ("raw_c0_mean", "float64", "intensity", "channel c0: mean of finite intensities"),
    ("raw_c0_valid_pixels", "int64", "count", "channel c0: finite pixels in the object"),
)


def write_example(directory: Path, *, seed: int = 0) -> tuple[Path, Path]:
    """Write ``measurements.csv`` and ``data_dictionary.csv``. Returns their paths."""
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    measurements = directory / "measurements.csv"
    dictionary = directory / "data_dictionary.csv"
    with measurements.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        for condition in CONDITIONS:
            area_effect = TREATMENT_AREA_LOG_EFFECT if condition == "treated" else 0.0
            for donor in range(1, DONORS_PER_CONDITION + 1):
                donor_area = rng.normal(0.0, DONOR_SD)
                donor_intensity = rng.normal(0.0, DONOR_SD)
                for field in range(FIELDS_PER_DONOR):
                    item = f"{condition}-{donor:02d}_field{field}"
                    field_area = rng.normal(0.0, FIELD_SD)
                    field_intensity = rng.normal(0.0, FIELD_SD)
                    objects = int(rng.integers(35, 46))
                    for label in range(1, objects + 1):
                        area = 40.0 * np.exp(
                            area_effect + donor_area + field_area + rng.normal(0.0, OBJECT_SD_AREA)
                        )
                        eccentricity = float(rng.beta(4, 2))
                        intensity = 800.0 * np.exp(
                            donor_intensity + field_intensity + rng.normal(0.0, OBJECT_SD_INTENSITY)
                        )
                        writer.writerow(
                            [
                                item,
                                "plane0",
                                label,
                                f"{area:.6g}",
                                f"{eccentricity:.6g}",
                                f"{intensity:.6g}",
                                max(1, round(area)),
                            ]
                        )
    with dictionary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["name", "dtype", "unit", "description"])
        writer.writerows(DICTIONARY)
    return measurements, dictionary
