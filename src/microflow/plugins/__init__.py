"""Plugin architecture: documented interfaces, entry-point discovery and output contracts.

``api`` holds the interfaces and ``PluginSpec``; ``registry`` discovers and loads plugins;
``contracts`` checks what they return. See docs/plugins.md for authoring and trust rules.
"""

from microflow.plugins.api import (
    ENTRY_POINT_GROUPS,
    KINDS,
    PLUGIN_API_VERSION,
    FeatureColumn,
    PluginError,
    PluginSpec,
)

__all__ = [
    "ENTRY_POINT_GROUPS",
    "KINDS",
    "PLUGIN_API_VERSION",
    "FeatureColumn",
    "PluginError",
    "PluginSpec",
]
