"""The telemetry source port: where a session's data comes from.

The signature that matters here is :meth:`TelemetrySource.availability`. A source
**declares what it has** before anyone reads from it, rather than promising a
fixed set of channels.

This is not defensive design for its own sake. The two real sources differ
fundamentally: Assetto Corsa's shared memory exposes one car and carries no
per-wheel slip angle, while the replay file on disk carries every car with
per-wheel slip angle, slip ratio, normalised slip and load. A port that promised
a uniform struct would force the shared-memory implementation to return zeros for
channels it cannot observe — and a zero slip angle is indistinguishable from a
car tracking perfectly straight.

So the contract is: ask first, then read. Analyses that need a missing channel
either degrade to a documented proxy or refuse to run, and say which.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from assetto_ai_director.application.ports.progress import ProgressReporter
    from assetto_ai_director.domain.session import SessionMeta
    from assetto_ai_director.domain.telemetry import ChannelAvailability, SessionTelemetry

__all__ = ["TelemetrySource"]


@runtime_checkable
class TelemetrySource(Protocol):
    """Provides one session's telemetry.

    Implementations are expected to make :meth:`describe` and
    :meth:`availability` cheap — the GUI calls them to populate a session list
    without committing to a full parse — and to do the expensive work only in
    :meth:`load`.
    """

    @property
    def name(self) -> str:
        """A short identifier for this source, used in errors and logs.

        Returns:
            The source name, e.g. ``"acreplay"`` or ``"synthetic"``.
        """
        ...

    def describe(self) -> SessionMeta:
        """Read the session's metadata without loading telemetry.

        Must be cheap. For a replay file this means parsing a few hundred bytes
        of header rather than several hundred megabytes of frames.

        Returns:
            The session metadata.

        Raises:
            ReplayError: If the underlying data cannot be interpreted.
        """
        ...

    def availability(self) -> ChannelAvailability:
        """Declare which channels this source actually provides.

        Callers must consult this before reading a channel. Sources must not
        report a channel they will fill with fabricated values.

        Returns:
            The capability declaration.
        """
        ...

    def load(self, progress: ProgressReporter | None = None) -> SessionTelemetry:
        """Load the full session.

        Args:
            progress: Optional reporter for long operations.

        Returns:
            The complete session telemetry.

        Raises:
            ReplayError: If the data is malformed or its layout cannot be
                verified.
        """
        ...
