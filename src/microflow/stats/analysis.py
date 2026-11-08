"""Analysis of a study design: descriptive statistics, comparisons and corrections.

Inference is at the level of biological samples. Each measure is first summarised within every
sample
(see ``data.sample_values``), and every descriptive statistic, interval and test here takes those
sample values as input. Object-level values are reported as counts and descriptive summaries only.

The result is a set of plain rows. Writing them to files is ``output``'s job, and drawing them is
``plots``'.
"""

import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from microflow.stats import corrections, estimators
from microflow.stats.data import (
    ObjectTable,
    SampleValue,
    check_assignments,
    group_objects,
    sample_values,
)
from microflow.stats.design import StatsDesign

LOW_REPLICATION = 3
"""Groups with fewer samples than this are flagged: tests have little power, and bootstrap intervals
are unreliable."""
STABLE_BOOTSTRAP = 0.9
"""Minimum share of bootstrap resamples that must give a finite Hedges' g."""


@dataclass(frozen=True)
class SampleRow:
    measure: str
    unit: str
    group: str
    sample: str
    items: int
    objects: int
    value: float
    included: bool


@dataclass(frozen=True)
class DescriptiveRow:
    measure: str
    unit: str
    group: str
    samples: int
    items: int
    objects: int
    sample_mean: float
    sample_sd: float
    sem: float
    mean_ci_low: float
    mean_ci_high: float
    sample_median: float
    sample_q25: float
    sample_q75: float
    object_mean: float
    object_sd: float
    object_median: float
    flags: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonRow:
    measure: str
    unit: str
    group_a: str
    group_b: str
    samples_a: int
    samples_b: int
    objects_a: int
    objects_b: int
    mean_a: float
    mean_b: float
    difference: float
    difference_ci_low: float
    difference_ci_high: float
    hedges_g: float
    hedges_g_ci_low: float
    hedges_g_ci_high: float
    rank_biserial: float
    p_welch: float
    p_welch_adjusted: float
    p_mannwhitney: float
    p_mannwhitney_adjusted: float
    p_permutation: float
    p_permutation_adjusted: float
    permutation_method: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisResult:
    name: str
    descriptive: tuple[DescriptiveRow, ...]
    comparisons: tuple[ComparisonRow, ...]
    samples: tuple[SampleRow, ...]
    families: dict[str, int]
    warnings: tuple[str, ...]
    correction: str
    confidence: float


@dataclass(frozen=True)
class _Group:
    values: NDArray[np.float64]
    sample_names: tuple[str, ...]
    items: int
    objects: NDArray[np.float64]
    excluded: int


def run_analysis(design: StatsDesign, table: ObjectTable) -> AnalysisResult:
    check_assignments(table, design)
    descriptive: list[DescriptiveRow] = []
    comparisons: list[ComparisonRow] = []
    samples: list[SampleRow] = []
    warnings: list[str] = []
    for measure in design.measures:
        unit = table.units.get(measure, "")
        entries = sample_values(table, design, measure)
        samples.extend(_sample_row(measure, unit, entry) for entry in entries)
        groups = {name: _group(design, table, measure, entries, name) for name in design.groups}
        for name, group in groups.items():
            descriptive.append(_descriptive(design, measure, unit, name, group))
            if group.values.size == 0:
                warnings.append(f"{measure}: no sample in group {name} has valid objects")
            elif group.excluded:
                warnings.append(
                    f"{measure}: {group.excluded} sample(s) in group {name} have no valid "
                    "objects and are excluded"
                )
        for first, second in design.comparisons:
            row = _comparison(design, measure, unit, first, second, groups[first], groups[second])
            comparisons.append(row)
            if "insufficient_replication" in row.flags:
                warnings.append(
                    f"{measure}: {first} vs {second} not tested; fewer than "
                    f"{design.min_samples_per_group} samples in a group"
                )
    comparisons, families = _corrected(design, comparisons)
    return AnalysisResult(
        name=design.name,
        descriptive=tuple(descriptive),
        comparisons=tuple(comparisons),
        samples=tuple(samples),
        families=families,
        warnings=tuple(warnings),
        correction=design.correction,
        confidence=design.bootstrap.confidence,
    )


def _sample_row(measure: str, unit: str, entry: SampleValue) -> SampleRow:
    return SampleRow(
        measure=measure,
        unit=unit,
        group=entry.group,
        sample=entry.sample,
        items=entry.items,
        objects=entry.objects,
        value=entry.value,
        included=bool(np.isfinite(entry.value)),
    )


def _group(
    design: StatsDesign,
    table: ObjectTable,
    measure: str,
    entries: Sequence[SampleValue],
    name: str,
) -> _Group:
    own = [entry for entry in entries if entry.group == name]
    included = [entry for entry in own if np.isfinite(entry.value)]
    return _Group(
        values=np.asarray([entry.value for entry in included], dtype=np.float64),
        sample_names=tuple(entry.sample for entry in included),
        items=sum(entry.items for entry in included),
        objects=group_objects(table, design, measure, name),
        excluded=len(own) - len(included),
    )


def _descriptive(
    design: StatsDesign, measure: str, unit: str, name: str, group: _Group
) -> DescriptiveRow:
    values = group.values
    count = values.size
    flags: list[str] = []
    if count < design.min_samples_per_group:
        flags.append("insufficient_replication")
    elif count < LOW_REPLICATION:
        flags.append("low_replication")
    if group.excluded:
        flags.append(f"excluded_samples:{group.excluded}")
    mean, sd = estimators.mean_and_sd(values)
    low, high = estimators.bootstrap_mean_interval(
        values,
        rng=_rng(design.bootstrap.seed, "mean", measure, name),
        resamples=design.bootstrap.resamples,
        confidence=design.bootstrap.confidence,
    )
    quartiles = _quantiles(values, (0.25, 0.5, 0.75))
    object_mean, object_sd = estimators.mean_and_sd(group.objects)
    return DescriptiveRow(
        measure=measure,
        unit=unit,
        group=name,
        samples=count,
        items=group.items,
        objects=int(group.objects.size),
        sample_mean=mean,
        sample_sd=sd,
        sem=sd / math.sqrt(count) if count >= 2 else float("nan"),
        mean_ci_low=low,
        mean_ci_high=high,
        sample_median=quartiles[1],
        sample_q25=quartiles[0],
        sample_q75=quartiles[2],
        object_mean=object_mean,
        object_sd=object_sd,
        object_median=_quantiles(group.objects, (0.5,))[0],
        flags=tuple(flags),
    )


def _comparison(
    design: StatsDesign,
    measure: str,
    unit: str,
    first: str,
    second: str,
    a: _Group,
    b: _Group,
) -> ComparisonRow:
    minimum = design.min_samples_per_group
    na, nb = a.values.size, b.values.size
    testable = min(na, nb) >= max(minimum, 1)
    flags: list[str] = []
    if not testable:
        flags.append("insufficient_replication")
    elif min(na, nb) < LOW_REPLICATION:
        flags.append("low_replication")
    if a.excluded or b.excluded:
        flags.append(f"excluded_samples:{a.excluded + b.excluded}")
    mean_a, _ = estimators.mean_and_sd(a.values)
    mean_b, _ = estimators.mean_and_sd(b.values)
    key = f"{measure}|{first}|{second}"
    bootstrap = design.bootstrap
    permutation = design.permutation
    nan = float("nan")
    difference_low = difference_high = g = g_low = g_high = rank_biserial = nan
    p_welch = p_mwu = p_perm = nan
    method = "not_applicable"
    if testable:
        difference_low, difference_high = estimators.bootstrap_difference_interval(
            a.values,
            b.values,
            rng=_rng(bootstrap.seed, "difference", key),
            resamples=bootstrap.resamples,
            confidence=bootstrap.confidence,
        )
        g = estimators.hedges_g(a.values, b.values)
        g_low, g_high, usable = estimators.bootstrap_hedges_interval(
            a.values,
            b.values,
            rng=_rng(bootstrap.seed, "hedges", key),
            resamples=bootstrap.resamples,
            confidence=bootstrap.confidence,
        )
        if usable < STABLE_BOOTSTRAP:
            flags.append("unstable_bootstrap")
        if math.isnan(g) and min(na, nb) >= 2:
            flags.append("zero_variance")
        p_welch = estimators.welch_p(a.values, b.values)
        p_mwu, rank_biserial = estimators.mann_whitney(a.values, b.values)
        p_perm, method = estimators.permutation_p(
            a.values,
            b.values,
            rng=_rng(permutation.seed, "permutation", key),
            resamples=permutation.resamples,
            max_exact=permutation.max_exact,
        )
    return ComparisonRow(
        measure=measure,
        unit=unit,
        group_a=first,
        group_b=second,
        samples_a=na,
        samples_b=nb,
        objects_a=int(a.objects.size),
        objects_b=int(b.objects.size),
        mean_a=mean_a,
        mean_b=mean_b,
        difference=mean_a - mean_b,
        difference_ci_low=difference_low,
        difference_ci_high=difference_high,
        hedges_g=g,
        hedges_g_ci_low=g_low,
        hedges_g_ci_high=g_high,
        rank_biserial=rank_biserial,
        p_welch=p_welch,
        p_welch_adjusted=float("nan"),
        p_mannwhitney=p_mwu,
        p_mannwhitney_adjusted=float("nan"),
        p_permutation=p_perm,
        p_permutation_adjusted=float("nan"),
        permutation_method=method,
        flags=tuple(flags),
    )


def _corrected(
    design: StatsDesign, rows: list[ComparisonRow]
) -> tuple[list[ComparisonRow], dict[str, int]]:
    """Correct each test family across all measures and comparisons, separately per test."""
    welch = corrections.adjust([row.p_welch for row in rows], design.correction)
    mwu = corrections.adjust([row.p_mannwhitney for row in rows], design.correction)
    permutation = corrections.adjust([row.p_permutation for row in rows], design.correction)
    corrected = [
        replace(
            row,
            p_welch_adjusted=welch[index],
            p_mannwhitney_adjusted=mwu[index],
            p_permutation_adjusted=permutation[index],
        )
        for index, row in enumerate(rows)
    ]
    families = {
        "welch": sum(math.isfinite(value) for value in welch),
        "mannwhitney": sum(math.isfinite(value) for value in mwu),
        "permutation": sum(math.isfinite(value) for value in permutation),
    }
    return corrected, families


def _quantiles(values: NDArray[np.float64], points: tuple[float, ...]) -> tuple[float, ...]:
    if values.size == 0:
        return tuple(float("nan") for _ in points)
    return tuple(float(value) for value in np.quantile(values, points))


def _rng(seed: int, *keys: str) -> np.random.Generator:
    """A generator that depends only on the seed and the names, so results do not depend on the
    order in which measures or comparisons are processed."""
    return np.random.default_rng([seed, zlib.crc32("|".join(keys).encode("utf-8"))])
