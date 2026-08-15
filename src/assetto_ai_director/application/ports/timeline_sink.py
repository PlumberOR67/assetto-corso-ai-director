"""The timeline sink port: how a finished edit leaves the system.

Every export target loses something, and they lose different things. Assetto
Corsa's ``cameras.ini`` describes trackside cameras selected by position along
the track, so it structurally *cannot* cut on an event — it has no concept of
replay time. An edit-decision list can express cuts and speed ramps but not
camera placement.

So a sink does not merely succeed or fail. It reports, before writing, exactly
which parts of the timeline survive the trip. That report is shown to the user,
because "your Initial D cut has been exported" is a lie if the target dropped
every event-triggered cut in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from assetto_ai_director.domain.timeline import Timeline

__all__ = ["SupportReport", "TimelineSink"]


@dataclass(frozen=True, slots=True)
class SupportReport:
    """What a sink can and cannot preserve about a given timeline.

    Attributes:
        supported: Whether the export can proceed at all.
        lossy: Whether anything will be dropped or approximated.
        limitations: Plain-language descriptions of what will be lost, written
            for a user rather than a developer.
        dropped_shots: How many shots the target cannot represent.
    """

    supported: bool
    lossy: bool = False
    limitations: tuple[str, ...] = field(default_factory=tuple)
    dropped_shots: int = 0

    @classmethod
    def full(cls) -> SupportReport:
        """Build a report for a target that preserves everything.

        Returns:
            A lossless support report.
        """
        return cls(supported=True)

    @classmethod
    def refused(cls, reason: str) -> SupportReport:
        """Build a report for a target that cannot handle this timeline.

        Args:
            reason: Why the export cannot proceed.

        Returns:
            An unsupported report carrying the reason.
        """
        return cls(supported=False, limitations=(reason,))

    @property
    def summary(self) -> str:
        """A one-line description suitable for a CLI or status bar.

        Returns:
            Human-readable summary of fidelity.
        """
        if not self.supported:
            return f"Not supported: {'; '.join(self.limitations)}"
        if not self.lossy:
            return "Exports without loss."
        detail = "; ".join(self.limitations)
        return f"Exports with loss: {detail}"


@runtime_checkable
class TimelineSink(Protocol):
    """Writes a timeline to some external representation."""

    @property
    def name(self) -> str:
        """A short identifier for this sink.

        Returns:
            The sink name, e.g. ``"json"`` or ``"cameras-ini"``.
        """
        ...

    @property
    def extension(self) -> str:
        """The conventional file extension this sink writes, including the dot.

        Returns:
            The extension, e.g. ``".json"``.
        """
        ...

    def supports(self, timeline: Timeline) -> SupportReport:
        """Report what would survive exporting this timeline.

        Must not write anything. Called before :meth:`write` so the user can be
        warned, and called by the GUI to grey out targets that cannot represent
        the current edit.

        Args:
            timeline: The timeline to assess.

        Returns:
            The fidelity report.
        """
        ...

    def write(self, timeline: Timeline, destination: Path) -> None:
        """Write the timeline.

        Args:
            timeline: The timeline to export.
            destination: Where to write it.

        Raises:
            OSError: If the destination cannot be written.
            ValueError: If the timeline is unsupported by this sink.
        """
        ...
