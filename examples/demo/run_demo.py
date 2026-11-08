"""End-to-end demonstration: raw synthetic microscopy images to segmentation, measurements,
statistics, an HTML report and provenance checks.

Run from the repository root with the dependencies installed (``uv sync``):

    uv run python examples/demo/run_demo.py [--clean]

The data are synthetic and generated from a fixed seed. Two conditions (control and treated) have
four donors each, with two fields of view per donor. Treated objects are about 1.4 times larger in
area, and each image carries a known ground truth, so segmentation can be scored (Dice and IoU) and
not only inspected. Each step runs the same command a user would type (``python -m microflow ...``),
and the demo stops at the first failure. Outputs go to data/demo/, which is git-ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "data" / "demo"
RAW = DEMO / "raw"
TRUTH = DEMO / "truth"
RESULTS = DEMO / "results"
BATCH_CONFIG = Path(__file__).with_name("batch.yaml")
DESIGN = Path(__file__).with_name("design.yaml")
SEED = 20260101
SIZE = 128
DISCS_PER_IMAGE = 14
DONORS = 4
FIELDS = 2
# Base radius in pixels. Treated radii are scaled by sqrt(1.4), so areas differ by about 1.4 times.
CONDITIONS = {"control": 6.0, "treated": 6.0 * math.sqrt(1.4)}


def make_data(seed: int) -> list[dict[str, Any]]:
    """Write raw images and ground-truth masks, and return a record of how each was generated."""
    import numpy as np

    from microflow import __version__
    from microflow.image_io import write_image
    from microflow.sample_data import Disc, make_phantom

    RAW.mkdir(parents=True, exist_ok=True)
    TRUTH.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for condition, base_radius in CONDITIONS.items():
        for donor in range(1, DONORS + 1):
            donor_factor = math.exp(float(rng.normal(0.0, 0.04)))
            for field in range(FIELDS):
                name = f"{condition}-{donor:02d}_field{field}"
                discs = []
                for _ in range(DISCS_PER_IMAGE):
                    radius = base_radius * donor_factor * float(rng.uniform(0.85, 1.15))
                    low, high = radius + 2, SIZE - radius - 2
                    centre = (float(rng.uniform(low, high)), float(rng.uniform(low, high)))
                    discs.append(Disc(centre=centre, radius=radius))
                image_seed = int(rng.integers(0, 2**31 - 1))
                image, truth = make_phantom(discs, (SIZE, SIZE), seed=image_seed)
                write_image(RAW / f"{name}.tif", np.round(image).astype(np.uint16))
                write_image(TRUTH / f"{name}.tif", (truth > 0).astype(np.uint8))
                records.append(
                    {
                        "name": name,
                        "condition": condition,
                        "donor": f"{condition}-{donor:02d}",
                        "discs": DISCS_PER_IMAGE,
                        "base_radius_px": base_radius,
                        "donor_factor": donor_factor,
                        "image_seed": image_seed,
                    }
                )
    (DEMO / "generation.json").write_text(
        json.dumps(
            {"seed": seed, "microflow_version": __version__, "size_px": SIZE, "images": records},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return records


def run_microflow(*args: str) -> None:
    print(f"$ microflow {' '.join(args)}")
    subprocess.run([sys.executable, "-m", "microflow", *args], cwd=ROOT, check=True)


def score_against_truth(records: list[dict[str, Any]]) -> dict[str, float]:
    """Dice and IoU of each image's classical labels against its ground truth."""
    from microflow.batch.store import item_directory_name
    from microflow.loading import load_image
    from microflow.segmentation import score_segmentation

    rows = []
    for record in records:
        name = str(record["name"])
        labels = load_image(
            RESULTS / "batch" / "items" / item_directory_name(name) / "labels.tif", axes="YX"
        ).data
        truth = load_image(TRUTH / f"{name}.tif", axes="YX").data
        scores = score_segmentation(labels, truth)
        rows.append(
            {
                "name": name,
                "condition": record["condition"],
                "dice": f"{scores.dice:.4f}",
                "iou": f"{scores.iou:.4f}",
                "predicted_pixels": scores.predicted_pixels,
                "truth_pixels": scores.truth_pixels,
            }
        )
    path = RESULTS / "segmentation_quality.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    means: dict[str, float] = {}
    for condition in CONDITIONS:
        dice = [float(row["dice"]) for row in rows if row["condition"] == condition]
        means[condition] = sum(dice) / len(dice)
    return means


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", action="store_true", help="Delete data/demo before running.")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for the synthetic images.")
    args = parser.parse_args(argv)
    if args.clean and DEMO.exists():
        shutil.rmtree(DEMO)

    print("1. synthetic images and ground truth")
    records = make_data(args.seed)
    print(f"   {len(records)} images in {RAW.relative_to(ROOT)}")

    print("2. batch segmentation and measurement (calibrated, 0.5 um per pixel)")
    run_microflow("batch", str(BATCH_CONFIG))

    print("3. segmentation quality against the known truth")
    means = score_against_truth(records)
    for condition, dice in means.items():
        print(f"   mean Dice {condition}: {dice:.4f}")

    print("4. export measurements with a data dictionary and provenance")
    run_microflow(
        "export", str(RESULTS / "batch"), "--output", str(RESULTS / "export"), "--format", "csv"
    )

    print("5. statistics by biological sample, with figures")
    run_microflow("stats", "run", str(DESIGN))

    print("6. one HTML report, linked to each image's quality control")
    run_microflow(
        "stats",
        "report",
        str(RESULTS / "stats"),
        "--output",
        str(RESULTS / "report.html"),
        "--batch-dir",
        str(RESULTS / "batch"),
    )

    print("7. re-hash every recorded input and output of the batch run")
    run_microflow("verify-run", str(RESULTS / "batch"))

    print(f"done: open {(RESULTS / 'report.html').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
