"""Progress reporting, so long analyses can be observed without coupling to a UI.

Parsing a 340 MB replay and running the full analysis takes long enough that a
silent wait is unacceptable. But the pipeline must not know whether it is being
watched by a Qt progress bar, a Rich spinner, an MCP progress notification or
nothing at all.

Hence a Protocol with a no-op default. Callers that care supply one; callers that
do not pass ``None`` and the pipeline never branches on it.

**Stages are enumerated here rather than passed as free strings**, and that is
load-bearing for the UI rather than tidiness. A progress display that grows one
row at a time as stages announce themselves is barely better than a spinner: the
user still cannot tell whether the thing they are waiting for is nearly done or
has five more phases to go. :data:`PIPELINE_STAGES` lets a UI draw the whole
checklist *before* the run starts, so the shape of the wait is visible from the
first second.

A named enum also makes a whole class of bug impossible. The stage names used to
be string literals at each call site, and three of the four never reached any
reporter — a UI showing 100% under the wrong stage name for three quarters of the
run. Nothing about a bare string can be asserted; a closed set can be, and
``tests/application/test_pipeline_progress.py`` does.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

__all__ = ["PIPELINE_STAGES", "NullProgress", "ProgressReporter", "Stage"]


class Stage(StrEnum):
    """A named phase of the analysis.

    The value is what a user sees, so it is written for them rather than for a
    developer reading a traceback.
    """

    DECODE = "Decoding cars"
    """Reading car frames out of the replay. Reported by the loader, not the
    pipeline, because it happens before any signals exist to hand over."""

    DETECT = "Detecting events"
    SCORE = "Scoring moments"
    CANDIDATES = "Generating candidates"
    SCHEDULE = "Selecting shots"
    CAMERA = "Building camera paths"
    """Longest stage on a full session, and the one that used to report nothing
    at all — the progress bar sat at 100% throughout it."""


PIPELINE_STAGES: tuple[Stage, ...] = (
    Stage.DECODE,
    Stage.DETECT,
    Stage.SCORE,
    Stage.CANDIDATES,
    Stage.SCHEDULE,
    Stage.CAMERA,
)
"""Every stage of a full run, in the order they happen.

Includes :attr:`Stage.DECODE`, which the pipeline itself never reports: a UI
drawing the checklist needs the whole journey, not just the part one function
happens to own.
"""


@runtime_checkable
class ProgressReporter(Protocol):
    """Receives progress updates from a long-running operation."""

    def start(self, description: str, total: int | None = None) -> None:
        """Announce that a phase has begun.

        Args:
            description: Human-readable phase name, e.g. ``"Reading car frames"``.
            total: Total units of work, or ``None`` when it cannot be known in
                advance.
        """
        ...

    def advance(self, amount: int = 1) -> None:
        """Report that some work has completed.

        Args:
            amount: How many units finished since the last call.
        """
        ...

    def finish(self) -> None:
        """Announce that the current phase has ended."""
        ...


class NullProgress:
    """A reporter that discards everything.

    Lets the pipeline call progress methods unconditionally instead of guarding
    every call site with a null check.
    """

    def start(self, description: str, total: int | None = None) -> None:
        """Ignore the start of a phase.

        Args:
            description: Ignored.
            total: Ignored.
        """

    def advance(self, amount: int = 1) -> None:
        """Ignore a unit of progress.

        Args:
            amount: Ignored.
        """

    def finish(self) -> None:
        """Ignore the end of a phase."""
