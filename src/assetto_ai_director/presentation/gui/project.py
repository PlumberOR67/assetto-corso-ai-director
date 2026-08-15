"""What the window is currently working on.

The old window had no such concept. Opening a replay from the file dialog
immediately started the entire analysis, so "open" and "run" were the same act
and a user who wanted to look at a file before committing twenty-four minutes to
it had no way to. Everything since — the Run button, the stage list, the estimate
in the button's own label — depends on those two being separable.

**The replay is the document; the timeline is an artifact.** A user asked "why can
I save some timeline and what do I do with it", and that question is the design
review. The old window offered ``Ctrl+S`` → "Export timeline…", which teaches that
the timeline is the thing you are making. It is not: it is an intermediate that
feeds the camera set and, later, video. What you are working on is a session.

**STALE is the state that makes the window honest.** Changing the style does not
throw away the result on screen; it marks it as no longer matching the controls.
Without that, a user changes a dropdown, looks at an edit built with the previous
style, and has no way to know.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from assetto_ai_director.application.pipeline import SessionAnalysis
    from assetto_ai_director.domain.timeline import Timeline
    from assetto_ai_director.infrastructure.loader import LoadedSession

__all__ = ["Phase", "Project", "RunKey"]

RunKey = tuple[str, float]
"""What identifies a directed result: ``(style name, scope seconds)``.

Scope belongs in the key as much as style does. It changes the horizon rather
than merely the weighting, so a three-minute edit and a whole-session edit in the
same style are different results, not the same one at different lengths.
"""


class Phase(StrEnum):
    """What the window can currently do.

    Attributes:
        EMPTY: Nothing open.
        INSPECTED: A session's header has been read — track, cars, quality are
            known — but nothing has been decoded or analysed. Nearly free, and
            the whole reason opening and running are separate acts.
        RUNNING: A worker owns the project.
        READY: A result exists and matches the current controls.
        STALE: A result exists but the controls have moved past it.
        FAILED: The last run raised.
    """

    EMPTY = "empty"
    INSPECTED = "inspected"
    RUNNING = "running"
    READY = "ready"
    STALE = "stale"
    FAILED = "failed"

    @property
    def has_result(self) -> bool:
        """Whether a timeline is on screen.

        Returns:
            ``True`` for :attr:`READY` and :attr:`STALE`.
        """
        return self in (Phase.READY, Phase.STALE)

    @property
    def can_run(self) -> bool:
        """Whether the Run button should be enabled.

        Returns:
            ``True`` whenever a session is open and nothing is running.
        """
        return self is not Phase.EMPTY and self is not Phase.RUNNING


@dataclass
class Project:
    """The session the window is working on, and everything derived from it.

    Attributes:
        path: The session file, or ``None`` when nothing is open.
        phase: What the window can currently do.
        session: The decoded session, once a run has loaded it.
        analysis: Events and scores. Style-independent, so it survives a style
            change — which is the whole reason changing style is cheap.
        timelines: Results, keyed by ``(style, scope)``.
        current: The key whose result is on screen.
        message: The last failure or cancellation, for display.
    """

    path: Path | None = None
    phase: Phase = Phase.EMPTY
    session: LoadedSession | None = None
    analysis: SessionAnalysis | None = None
    timelines: dict[RunKey, Timeline] = field(default_factory=dict)
    current: RunKey | None = None
    message: str = ""

    @property
    def timeline(self) -> Timeline | None:
        """The result currently on screen.

        Returns:
            The timeline, or ``None`` when there is none.
        """
        return self.timelines.get(self.current) if self.current else None

    def opened(self, path: Path) -> Project:
        """Start a new project on a different session.

        Everything derived is dropped: a different file shares nothing with the
        previous one, and keeping a stale analysis around to be accidentally
        reused is precisely the kind of bug that produces a confident wrong
        answer.

        Args:
            path: The session file.

        Returns:
            A fresh project.
        """
        return Project(path=path, phase=Phase.INSPECTED)

    def cached(self, key: RunKey) -> Timeline | None:
        """Look up an already-computed result.

        Args:
            key: The style and scope to look for.

        Returns:
            The timeline, or ``None`` if it has not been computed.
        """
        return self.timelines.get(key)

    def invalidated_by_scope(self) -> Project:
        """Drop everything that a change of scope makes wrong.

        Scope changes the span that was decoded and analysed, so the analysis
        goes too — unlike a style change, which keeps it. Getting this backwards
        would silently direct a three-minute analysis over a twenty-four minute
        horizon.

        Returns:
            The project with derived state cleared.
        """
        return replace(
            self,
            session=None,
            analysis=None,
            timelines={},
            current=None,
            phase=Phase.INSPECTED if self.path else Phase.EMPTY,
        )

    def marked_stale(self) -> Project:
        """Note that the controls no longer match the result on screen.

        Returns:
            The project in :attr:`Phase.STALE`, or unchanged when there is no
            result to be stale.
        """
        if not self.phase.has_result:
            return self
        return replace(self, phase=Phase.STALE)

    def matching(self, key: RunKey) -> Project:
        """Show an already-computed result without re-running.

        Args:
            key: The style and scope to show.

        Returns:
            The project showing that result, or unchanged if it is not cached.
        """
        if key not in self.timelines:
            return self
        return replace(self, current=key, phase=Phase.READY, message="")

    def completed(self, key: RunKey, timeline: Timeline) -> Project:
        """Record a finished run.

        Args:
            key: What was run.
            timeline: The result.

        Returns:
            The project showing it.
        """
        return replace(
            self,
            timelines={**self.timelines, key: timeline},
            current=key,
            phase=Phase.READY,
            message="",
        )

    def failed(self, message: str) -> Project:
        """Record a failure.

        Args:
            message: What went wrong.

        Returns:
            The project in :attr:`Phase.FAILED`.
        """
        return replace(self, phase=Phase.FAILED, message=message)

    def stopped(self, message: str) -> Project:
        """Record a cancellation.

        A cancelled run must not destroy a result the user already had, so this
        returns to :attr:`Phase.STALE` when one exists rather than to
        :attr:`Phase.FAILED`. Pressing Cancel is not a failure and must not look
        like one.

        Args:
            message: Which stage was interrupted.

        Returns:
            The project with the previous result intact, if there was one.
        """
        phase = Phase.STALE if self.timeline is not None else Phase.INSPECTED
        return replace(self, phase=phase, message=message)
