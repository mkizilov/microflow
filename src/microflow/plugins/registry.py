"""Discovery and loading of plugins through entry points.

Discovery reads distribution metadata only, so listing plugins runs no plugin code. :func:`load`
imports a plugin's module, which runs it, and is called only for plugins that a trusted
configuration names. :func:`configure` loads a plugin and validates its options.
"""

import re
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from pydantic import BaseModel, ValidationError

from microflow.plugins.api import (
    ENTRY_POINT_GROUPS,
    KINDS,
    PLUGIN_API_VERSION,
    PLUGIN_NAME_PATTERN,
    REQUIRED_ATTRIBUTES,
    REQUIRED_METHODS,
    PluginError,
    PluginSpec,
)
from microflow.plugins.contracts import call, check_feature_declaration


@dataclass(frozen=True)
class Discovered:
    kind: str
    name: str
    distribution: str
    target: str


@dataclass(frozen=True)
class Configured:
    kind: str
    name: str
    plugin: Any
    options: BaseModel

    @property
    def version(self) -> str:
        return str(self.plugin.version)


def discover(kind: str | None = None) -> tuple[Discovered, ...]:
    """Installed plugins, by kind, name and distribution. Nothing is imported or run."""
    kinds = KINDS if kind is None else (_checked_kind(kind),)
    found = [
        Discovered(
            kind=item_kind,
            name=entry.name,
            distribution=_distribution_name(entry),
            target=entry.value,
        )
        for item_kind in kinds
        for entry in metadata.entry_points(group=ENTRY_POINT_GROUPS[item_kind])
    ]
    return tuple(sorted(found, key=lambda item: (item.kind, item.name, item.distribution)))


def load(kind: str, name: str) -> Any:
    """Import and create the named plugin, and check that it implements its interface."""
    kind = _checked_kind(kind)
    entries = [
        entry
        for entry in metadata.entry_points(group=ENTRY_POINT_GROUPS[kind])
        if entry.name == name
    ]
    if not entries:
        installed = sorted({entry.name for entry in discover(kind)})
        available = ", ".join(installed) if installed else "none"
        raise PluginError(
            f"no {kind} plugin named {name!r} is installed (installed: {available}); "
            "install the package that provides it"
        )
    if len(entries) > 1:
        owners = ", ".join(sorted(_distribution_name(entry) for entry in entries))
        raise PluginError(
            f"{kind} plugin {name!r} is provided by more than one distribution ({owners}); "
            "uninstall all but one"
        )
    entry = entries[0]
    label = f"{kind} plugin {name!r} ({entry.value} from {_distribution_name(entry)})"
    try:
        target = entry.load()
    except Exception as exc:
        raise PluginError(f"{label} could not be imported: {type(exc).__name__}: {exc}") from exc
    try:
        plugin = target() if isinstance(target, type) else target
    except Exception as exc:
        raise PluginError(f"{label} could not be created: {type(exc).__name__}: {exc}") from exc
    _validate_declaration(plugin, kind, name, label)
    return plugin


def configure(kind: str, spec: PluginSpec) -> Configured:
    """Load the plugin named by ``spec`` and validate its options against its options model."""
    plugin = load(kind, spec.name)
    try:
        options = plugin.options_model.model_validate(spec.options)
    except ValidationError as exc:
        raise PluginError(
            f"{kind} plugin {spec.name!r}: invalid options: {_describe(exc)}"
        ) from exc
    return Configured(kind=kind, name=spec.name, plugin=plugin, options=options)


def _validate_declaration(plugin: Any, kind: str, name: str, label: str) -> None:
    missing = [attribute for attribute in REQUIRED_ATTRIBUTES if not hasattr(plugin, attribute)]
    missing += [
        method for method in REQUIRED_METHODS[kind] if not callable(getattr(plugin, method, None))
    ]
    if missing:
        raise PluginError(
            f"{label} does not implement the {kind} interface; missing: {', '.join(missing)}"
        )
    if plugin.api_version != PLUGIN_API_VERSION:
        raise PluginError(
            f"{label} was written for plugin API {plugin.api_version}; "
            f"this MicroFlow provides API {PLUGIN_API_VERSION}"
        )
    if plugin.kind != kind:
        raise PluginError(f"{label} declares kind {plugin.kind!r}; it is registered as {kind!r}")
    if plugin.name != name or not re.fullmatch(PLUGIN_NAME_PATTERN, name):
        raise PluginError(
            f"{label} declares name {plugin.name!r}; the name must equal its entry-point name"
        )
    if not isinstance(plugin.version, str) or not plugin.version.strip():
        raise PluginError(f"{label} must declare a non-empty text version")
    options_model = plugin.options_model
    if not (isinstance(options_model, type) and issubclass(options_model, BaseModel)):
        raise PluginError(f"{label}: options_model must be a pydantic BaseModel subclass")
    if options_model.model_config.get("extra") != "forbid":
        raise PluginError(
            f"{label}: options_model must set extra='forbid', so that unknown options are "
            "rejected rather than ignored"
        )
    if kind == "feature":
        try:
            columns = call("feature", name, "features", plugin.features)
        except PluginError as exc:
            raise PluginError(f"{label}: {exc}") from exc
        check_feature_declaration(columns, label)


def _checked_kind(kind: str) -> str:
    if kind not in KINDS:
        raise PluginError(f"unknown plugin kind {kind!r}; the kinds are {', '.join(KINDS)}")
    return kind


def _distribution_name(entry: metadata.EntryPoint) -> str:
    dist = getattr(entry, "dist", None)
    name = getattr(dist, "name", None) if dist is not None else None
    return str(name) if name else "unknown distribution"


def _describe(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'options'}: {error['msg']}"
        for error in exc.errors()
    )
