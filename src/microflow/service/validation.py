"""Validation of batch and statistics configurations under the service's path and plugin policy.

The same checks run when a job is submitted and again when it starts, so a file that changed or
moved in the meantime is caught before any work is done. Torch-free and FastAPI-free.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microflow.batch import BatchConfig, load_batch_config
from microflow.config import ConfigError
from microflow.service.errors import ServiceError
from microflow.service.paths import redact, resolve_input, resolve_output
from microflow.service.settings import SERVICE_WORKERS, Layout
from microflow.stats.data import check_assignments, read_measurements
from microflow.stats.design import load_design
from microflow.stats.errors import StatsError


@dataclass(frozen=True)
class PreparedBatch:
    config: BatchConfig
    config_path: Path
    output_directory: Path
    output_name: str
    warnings: tuple[str, ...]


def prepare_batch(text: object, layout: Layout) -> PreparedBatch:
    config_path = resolve_input(text, layout.input_roots)
    try:
        config = load_batch_config(config_path)
    except ConfigError as exc:
        raise _invalid(exc, {str(config_path): config_path.name}) from exc
    uses = plugin_uses(config)
    if uses:
        raise ServiceError(
            "plugins_disabled",
            f"the service does not run plugins; this configuration uses {', '.join(uses)}",
        )
    for image in config.dataset.images:
        try:
            resolve_input(str(image.path), layout.input_roots)
        except ServiceError as exc:
            raise ServiceError(exc.code, f"image '{image.name}': {exc.message}") from exc
    output = resolve_output(str(config.output_directory), layout.output_roots)
    warnings: list[str] = []
    if config.workers > SERVICE_WORKERS:
        warnings.append(
            f"workers reduced from {config.workers} to {SERVICE_WORKERS} by the service"
        )
    return PreparedBatch(
        config=config,
        config_path=config_path,
        output_directory=output,
        output_name=_label(output, layout.output_roots),
        warnings=tuple(warnings),
    )


def batch_summary(prepared: PreparedBatch) -> dict[str, Any]:
    config = prepared.config
    return {
        "name": config.name,
        "images": len(config.dataset.images),
        "output": prepared.output_name,
        "workers": min(config.workers, SERVICE_WORKERS),
    }


def prepare_stats(text: object, layout: Layout) -> dict[str, Any]:
    design_path = resolve_input(text, layout.input_roots)
    try:
        design = load_design(design_path)
    except ConfigError as exc:
        raise _invalid(exc, {str(design_path): design_path.name}) from exc
    resolve_input(str(design.measurements), layout.input_roots)
    if design.data_dictionary is not None:
        resolve_input(str(design.data_dictionary), layout.input_roots)
    output = resolve_output(str(design.output_directory), layout.output_roots)
    try:
        table = read_measurements(
            design.measurements, measures=design.measures, dictionary=design.data_dictionary
        )
        check_assignments(table, design)
    except StatsError as exc:
        raise _invalid(exc, {str(design.measurements): "measurements"}) from exc
    return {
        "name": design.name,
        "measures": list(design.measures),
        "groups": {name: len(group.samples) for name, group in design.groups.items()},
        "comparisons": [list(pair) for pair in design.comparisons],
        "object_rows": table.objects,
        "output": _label(output, layout.output_roots),
    }


def plugin_uses(config: BatchConfig) -> tuple[str, ...]:
    uses: list[str] = []
    if config.segmentation.method == "plugin":
        uses.append("segmentation method 'plugin'")
    if config.features:
        uses.append("features")
    if config.quality_checks:
        uses.append("quality_checks")
    if any(step.kind == "plugin" for step in config.preprocessing):
        uses.append("a plugin preprocessing step")
    return tuple(uses)


def _invalid(exc: Exception, replacements: Mapping[str, str]) -> ServiceError:
    lines = redact(str(exc), replacements).splitlines() or ["the configuration is invalid"]
    return ServiceError("invalid_config", lines[0], details=lines[1:])


def _label(output: Path, roots: tuple[Path, ...]) -> str:
    root = next(root for root in roots if output.is_relative_to(root))
    return f"{root.name}/{output.relative_to(root).as_posix()}"
