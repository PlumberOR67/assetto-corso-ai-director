"""Server-side session state for the MCP tools.

MCP tools are stateless calls, but directing a replay is not: decoding a 300 MB
file takes seconds and every later step depends on it. Re-reading per call would
make an interactive workflow unusable.

So a client opens a session, works against its handle, and the expensive stages
are computed once and cached. Each stage is computed **lazily on first use**, so
a client that only wants a shot list never pays for camera paths, and one that
only wants to inspect metadata never decodes a frame.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.application.pipeline import SessionAnalysis, analyse
from assetto_ai_director.infrastructure.loader import load_session

if TYPE_CHECKING:
    from pathlib import Path

    from assetto_ai_director.application.events.detect import CarSignals, DetectionResult
    from assetto_ai_director.domain.scoring import ScoreVector
    from assetto_ai_director.domain.session import SessionMeta
    from assetto_ai_director.domain.timeline import Timeline

__all__ = ["Session", "SessionStore"]


@dataclass
class Session:
    """One opened replay and everything derived from it.

    Attributes:
        handle: The identifier a client passes back.
        path: The replay file.
        meta: Session metadata.
        signals: Decoded per-car signals, excluding parked entries.
        time: Time per frame, seconds.
        sample_rate: Frames per second.
        timelines: Directed timelines, keyed by style name.
        source_notes: Caveats from loading — parked cars dropped, channels a CSV
            could not supply. Distinct from the analysis's own notes, and carried
            separately so neither can silently absorb the other.
    """

    handle: str
    path: Path
    meta: SessionMeta
    signals: dict[int, CarSignals]
    time: np.ndarray
    sample_rate: float
    timelines: dict[str, Timeline] = field(default_factory=dict)
    source_notes: tuple[str, ...] = ()
    _analysis: SessionAnalysis | None = None

    @property
    def analysis(self) -> SessionAnalysis:
        """Events and scores, computed once on first use.

        Keyed by nothing, because nothing in it depends on the style — which is
        why ``timelines`` above *is* keyed by style name and this is not.

        Returns:
            The analysis.
        """
        if self._analysis is None:
            self._analysis = analyse(self.signals, self.time, self.sample_rate)
        return self._analysis

    @property
    def notes(self) -> tuple[str, ...]:
        """Everything a client should be told about this session.

        Returns:
            Loading caveats followed by analysis caveats.
        """
        return (*self.source_notes, *self.analysis.notes)

    @property
    def detection(self) -> DetectionResult:
        """Detected events, computed once on first use.

        Returns:
            The detection result.
        """
        return self.analysis.detection

    @property
    def scores(self) -> dict[int, list[ScoreVector]]:
        """Per-frame scores per car, computed once on first use.

        Returns:
            The score vectors.
        """
        return self.analysis.scores

    @property
    def duration(self) -> float:
        """How long the analysed span runs.

        Returns:
            Duration in seconds.
        """
        return float(self.time[-1]) if self.time.size else 0.0


class SessionStore:
    """Holds open sessions for the lifetime of the server process."""

    def __init__(self) -> None:
        """Create an empty store."""
        self._sessions: dict[str, Session] = {}

    def open(self, path: Path, seconds: float = 0.0) -> Session:
        """Decode a replay and register it.

        Args:
            path: The replay file.
            seconds: Analyse only the first N seconds. Zero means all.

        Returns:
            The opened session.

        Raises:
            ReplayError: If the replay cannot be read.
            ValueError: If no car in the replay moved.
        """
        loaded = load_session(path, seconds=seconds)
        session = Session(
            handle=uuid.uuid4().hex[:12],
            path=loaded.path,
            meta=loaded.meta,
            signals=loaded.signals,
            time=loaded.time,
            sample_rate=loaded.sample_rate,
            source_notes=loaded.notes,
        )
        self._sessions[session.handle] = session
        return session

    def get(self, handle: str) -> Session:
        """Look up an open session.

        Args:
            handle: The identifier returned when it was opened.

        Returns:
            The session.

        Raises:
            KeyError: If no session has that handle. The message lists the open
                handles, because a client that lost track of one otherwise has no
                way to recover short of reopening everything.
        """
        try:
            return self._sessions[handle]
        except KeyError:
            known = ", ".join(sorted(self._sessions)) or "none"
            message = f"No open session {handle!r}. Open sessions: {known}."
            raise KeyError(message) from None

    def close(self, handle: str) -> None:
        """Forget a session and release its telemetry.

        Args:
            handle: The session to close.
        """
        self._sessions.pop(handle, None)

    @property
    def handles(self) -> tuple[str, ...]:
        """Every open session's handle.

        Returns:
            The handles, sorted.
        """
        return tuple(sorted(self._sessions))
