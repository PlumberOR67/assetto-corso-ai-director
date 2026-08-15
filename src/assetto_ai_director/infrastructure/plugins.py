"""Plugin discovery through standard entry points.

Five extension groups, matching the five things a third party might reasonably
want to add: where sessions come from, where they go, and how they are shot.

**Entry points rather than a bespoke registry.** A registry needs the plugin to
be imported before it can register itself, which means the host has to know about
it — which is not extensibility, it is a list with extra steps. Entry points are
declared in a package's own metadata, so installing the package is the whole
installation procedure.

**A broken plugin never takes the host down.** Third-party code fails in ways this
project cannot anticipate: a missing transitive dependency, an import-time side
effect, a version skew. Discovery therefore isolates each entry point and reports
the failure as data, so the tool starts, works, and tells the user which plugin
is broken — rather than refusing to run at all because something optional is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from importlib.metadata import EntryPoint, entry_points
from typing import Any

__all__ = ["PluginGroup", "PluginRegistry", "discover"]


class PluginGroup(StrEnum):
    """The extension points this project offers.

    Attributes:
        REPLAY_PROVIDERS: Read a session from a replay format.
        TELEMETRY_PROVIDERS: Read a session from a telemetry source.
        CAMERA_PACKS: Add or replace camera rigs.
        STYLE_PACKS: Add or replace directorial styles.
        EXPORTERS: Write a timeline to another format.
    """

    REPLAY_PROVIDERS = "assetto_ai_director.replay_providers"
    TELEMETRY_PROVIDERS = "assetto_ai_director.telemetry_providers"
    CAMERA_PACKS = "assetto_ai_director.camera_packs"
    STYLE_PACKS = "assetto_ai_director.style_packs"
    EXPORTERS = "assetto_ai_director.exporters"


@dataclass(frozen=True, slots=True)
class PluginRegistry:
    """What was found, and what failed to load.

    Attributes:
        loaded: Successfully loaded objects, keyed by entry-point name.
        failures: Why each failed entry point failed, keyed by name. Surfaced
            rather than swallowed: a plugin that quietly does not load is far
            harder to diagnose than one that says so.
    """

    group: PluginGroup
    loaded: dict[str, Any] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        """Every successfully loaded plugin's name.

        Returns:
            The names, sorted.
        """
        return tuple(sorted(self.loaded))

    def get(self, name: str) -> Any | None:
        """Look up one plugin.

        Args:
            name: The entry-point name.

        Returns:
            The loaded object, or ``None`` if it is absent or failed to load.
        """
        return self.loaded.get(name)

    @property
    def healthy(self) -> bool:
        """Whether every declared plugin in this group loaded.

        Returns:
            ``True`` if nothing failed.
        """
        return not self.failures


def discover(group: PluginGroup) -> PluginRegistry:
    """Find and load every plugin registered against a group.

    Args:
        group: Which extension point to search.

    Returns:
        The registry, including any failures.
    """
    loaded: dict[str, Any] = {}
    failures: dict[str, str] = {}

    for entry in _entry_points(group):
        try:
            loaded[entry.name] = entry.load()
        except Exception as error:
            # Deliberately broad. A plugin can raise anything at import time, and
            # the host's job is to survive it and say which one broke.
            failures[entry.name] = f"{type(error).__name__}: {error}"

    return PluginRegistry(group=group, loaded=loaded, failures=failures)


def _entry_points(group: PluginGroup) -> tuple[EntryPoint, ...]:
    """Read the entry points declared for a group.

    Args:
        group: The group to query.

    Returns:
        The declared entry points, or an empty tuple if the metadata cannot be
        read at all — which happens in some frozen and vendored environments and
        should degrade to "no plugins" rather than to a crash.
    """
    try:
        return tuple(entry_points(group=group.value))
    except Exception:
        return ()
