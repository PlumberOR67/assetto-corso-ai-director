"""Running the analysis off the GUI thread.

Directing a replay takes seconds to minutes. Doing it on the GUI thread freezes
the window, and a frozen window is one the operating system offers to kill.

Uses Qt's documented pattern — a plain ``QObject`` moved to a ``QThread`` — rather
than subclassing ``QThread``. Subclassing puts the worker's slots on the *calling*
thread while ``run()`` executes on the new one, which is the single most common
way threaded Qt code ends up with data races that only appear under load.

Progress crosses the thread boundary as a signal, which Qt queues and delivers on
the GUI thread. Touching a widget from the worker thread is undefined behaviour
in Qt and, on Windows, usually silent corruption rather than a crash.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, Signal

from assetto_ai_director.application.pipeline import SessionAnalysis
from assetto_ai_director.application.pipeline import analyse as run_analysis
from assetto_ai_director.application.pipeline import direct as run_director
from assetto_ai_director.application.ports.cancellation import CancelFlag
from assetto_ai_director.domain.errors import OperationCancelledError
from assetto_ai_director.infrastructure.loader import LoadedSession, load_session

if TYPE_CHECKING:
    from pathlib import Path

    from assetto_ai_director.domain.style import StyleProfile

# LoadedSession is re-exported: it used to be defined here, and it belongs to the
# infrastructure layer now that four callers share it.
__all__ = ["DirectorWorker", "LoadedSession", "SignalProgress"]


MIN_EMIT_INTERVAL_S = 0.03
"""Shortest gap between two progress emissions, roughly 33 per second.

**This is a correctness constraint, not a nicety.** The scheduler's inner loop
runs over hundreds of thousands of candidates. A queued cross-thread signal per
tick would arrive faster than the GUI thread can drain its event queue, and the
window would spend its time repainting a bar instead of staying responsive —
making a watched run *slower* than an unwatched one. Above about 30 Hz a progress
bar conveys nothing a person can read anyway.

Start and finish always emit regardless, so no stage is ever missed and no bar is
left short of its total.
"""


class SignalProgress(QObject):
    """Adapts the progress port onto a Qt signal.

    Satisfies ``ProgressReporter`` structurally while emitting across the thread
    boundary, so the application layer stays unaware that Qt exists.
    """

    changed = Signal(str, int, int)

    def __init__(self) -> None:
        """Create a reporter with no active phase."""
        super().__init__()
        self._description = ""
        self._total = 0
        self._done = 0
        self._last_emit = 0.0

    def start(self, description: str, total: int | None = None) -> None:
        """Begin a phase.

        Args:
            description: What is happening.
            total: Units of work, or ``None`` when unknown.
        """
        self._description = description
        self._total = total or 0
        self._done = 0
        self._emit(0)

    def advance(self, amount: int = 1) -> None:
        """Report completed work.

        Args:
            amount: Units finished since the last call.
        """
        self._done += amount
        now = time.monotonic()
        if now - self._last_emit >= MIN_EMIT_INTERVAL_S or self._done >= self._total > 0:
            self._emit(self._done, now)

    def finish(self) -> None:
        """End the current phase."""
        self._emit(self._total)

    def _emit(self, done: int, now: float | None = None) -> None:
        """Send an update and record when it went.

        Args:
            done: Units completed.
            now: Current monotonic time, when the caller already has it.
        """
        self._last_emit = time.monotonic() if now is None else now
        self.changed.emit(self._description, done, self._total)


class DirectorWorker(QObject):
    """Decodes a session and directs a cut, off the GUI thread.

    Accepts a replay or a CSV — the loader branches on the suffix, so the GUI
    gained CSV support the moment it stopped decoding replays itself.
    """

    progress = Signal(str, int, int)
    loaded = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)

    def __init__(
        self,
        path: Path,
        style: StyleProfile,
        seconds: float = 0.0,
        analysis: SessionAnalysis | None = None,
    ) -> None:
        """Configure the work.

        Args:
            path: The session to direct.
            style: The style to direct in.
            seconds: Analyse only the first N seconds. Zero means all.
            analysis: A previous :func:`analyse` result for this same session and
                span, so a style change skips decode, detection and scoring.
        """
        super().__init__()
        self.path = path
        self.style = style
        self.seconds = seconds
        self.analysis = analysis
        self._cancel = CancelFlag()

    def cancel(self) -> None:
        """Ask the run to stop at its next checkpoint.

        Safe to call from the GUI thread while the worker runs: the token is a
        ``threading.Event`` and nothing else crosses the boundary.
        """
        self._cancel.cancel()

    def run(self) -> None:
        """Decode and direct, emitting progress and the result.

        Every exception is caught and reported as a signal. An exception escaping
        a worker's slot terminates the thread silently, leaving the window
        waiting for a result that will never arrive — which is a far worse
        failure than an error message.
        """
        reporter = SignalProgress()
        reporter.changed.connect(self.progress.emit)
        try:
            session = self._session(reporter)
            self.loaded.emit(session)

            analysis = self.analysis
            if analysis is None:
                analysis = run_analysis(
                    session.signals,
                    session.time,
                    session.sample_rate,
                    progress=reporter,
                    cancel=self._cancel,
                )
            timeline = run_director(
                session.signals,
                session.meta,
                self.style,
                session.time,
                session.sample_rate,
                progress=reporter,
                cancel=self._cancel,
                analysis=analysis,
            )
            self.analysis = analysis
            self.finished.emit(timeline)
        except OperationCancelledError as stopped:
            # Before the general case. A user who pressed Cancel and is shown
            # "Could not direct this replay" has been told their machine broke.
            self.cancelled.emit(str(stopped))
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")

    def _session(self, reporter: SignalProgress) -> LoadedSession:
        """Load the session, or reuse one already decoded.

        Args:
            reporter: Progress reporter for the decode stage.

        Returns:
            The decoded session.
        """
        return load_session(self.path, seconds=self.seconds, progress=reporter, cancel=self._cancel)


def run_in_thread(worker: DirectorWorker) -> QThread:
    """Move a worker onto its own thread and start it.

    Args:
        worker: The worker to run.

    Returns:
        The started thread. The caller must keep a reference: a ``QThread`` that
        goes out of scope is destroyed while running, which crashes the process.
    """
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.start()
    return thread
