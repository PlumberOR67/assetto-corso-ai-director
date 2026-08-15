"""A Rich-backed implementation of the progress port.

Exists because measuring a twenty-minute corpus of dense optical flow takes
minutes, and a silent terminal is indistinguishable from a hung process. The
application layer stays unaware of Rich: it reports progress through the
:class:`~assetto_ai_director.application.ports.progress.ProgressReporter`
protocol and this adapter renders it.
"""

from __future__ import annotations

from types import TracebackType
from typing import Self

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)

__all__ = ["RichProgress"]


class RichProgress:
    """Renders progress to the terminal as a live bar.

    Implements the ``ProgressReporter`` protocol structurally rather than by
    inheritance, so the application layer never imports Rich.

    Use as a context manager so the live display is always torn down, including
    when a parse raises part-way through.
    """

    def __init__(self, console: Console | None = None) -> None:
        """Create a reporter.

        Args:
            console: Console to render to. A fresh one is created if omitted.
        """
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console or Console(),
            transient=True,
        )
        self._task_id: TaskID | None = None
        self._started = False

    def __enter__(self) -> Self:
        """Start the live display.

        Returns:
            This reporter.
        """
        self._progress.start()
        self._started = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the live display.

        Args:
            exc_type: Exception type, if one is propagating.
            exc: The exception, if any.
            traceback: The traceback, if any.
        """
        self.finish()
        self._progress.stop()
        self._started = False

    def start(self, description: str, total: int | None = None) -> None:
        """Begin a phase.

        Args:
            description: What is happening.
            total: Units of work, or ``None`` when unknown.
        """
        if not self._started:
            self._progress.start()
            self._started = True
        self.finish()
        self._task_id = self._progress.add_task(description, total=total)

    def advance(self, amount: int = 1) -> None:
        """Report completed work.

        Args:
            amount: Units finished since the last call.
        """
        if self._task_id is not None:
            self._progress.advance(self._task_id, amount)

    def finish(self) -> None:
        """End the current phase."""
        if self._task_id is not None:
            self._progress.remove_task(self._task_id)
            self._task_id = None
