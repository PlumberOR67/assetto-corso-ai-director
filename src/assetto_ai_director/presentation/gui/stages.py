"""The stage checklist: what the machine is doing, and how much is left.

Replaces a single progress bar that reported one of six stages under the wrong
name. A user watched it read "Detecting events — 75%" for ten minutes; detection
had finished in three seconds.

**Every row exists before the run starts.** A display that grows a row at a time
as stages announce themselves is barely better than a spinner — the user still
cannot tell whether the thing they are waiting on has one more phase or five.
Drawing all six up front makes the shape of the wait visible from the first
second, which is most of what was missing.

**Elapsed time is frozen when a stage finishes.** Afterwards the user can see
*which* stage ate the time, which is the difference between "that took a while"
and "the scheduler took eighty seconds of it".

**Never a percentage without the stage name beside it.** A bare percentage is
precisely the artifact that produced the original complaint: it converts "I do
not know how long this will take" into "this is finished and hung".
"""

from __future__ import annotations

import time
from typing import Final

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from assetto_ai_director.application.ports.progress import PIPELINE_STAGES, Stage
from assetto_ai_director.presentation.gui.theme import COLOURS

__all__ = ["StageList", "StageRow", "format_duration"]

PENDING: Final = "○"
RUNNING: Final = "◐"
DONE: Final = "●"
FAILED: Final = "✕"
"""Glyphs for the four states a stage can be in.

Geometric shapes rather than emoji: they render in every font Qt is likely to
fall back to, and they carry their meaning by fill rather than by colour, so the
list still reads for someone who cannot distinguish the colours.
"""

_SECONDS_PER_MINUTE: Final = 60

_MIN_SAMPLES_FOR_ESTIMATE: Final = 8
"""Units a stage must complete before its remaining time is worth showing.

Below this the extrapolation swings wildly between one update and the next, and a
figure that jumps from four seconds to nine minutes and back teaches the user to
ignore it.
"""


def format_duration(seconds: float) -> str:
    """Render a duration the way a person would say it.

    Args:
        seconds: The duration.

    Returns:
        A short string: seconds under a minute, minutes and seconds above.
    """
    if seconds < 1.0:
        return "<1s"
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds:.0f}s"
    minutes, remainder = divmod(int(seconds), _SECONDS_PER_MINUTE)
    return f"{minutes}m {remainder:02d}s"


class StageRow(QWidget):
    """One stage: its state, its counter, and how long it took."""

    def __init__(self, stage: Stage, parent: QWidget | None = None) -> None:
        """Build a pending row.

        Args:
            stage: Which stage this row reports.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.stage = stage
        self._started_at: float | None = None
        self._elapsed = 0.0
        self._done = 0
        self._total = 0

        self.setObjectName("StageRow")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        self.glyph = QLabel(PENDING)
        self.glyph.setFixedWidth(14)
        self.glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.name = QLabel(str(stage))
        self.name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.bar = QProgressBar()
        self.bar.setFixedWidth(110)
        self.bar.setFixedHeight(6)
        self.bar.setTextVisible(False)
        self.bar.setVisible(False)

        self.counter = QLabel("")
        self.counter.setFixedWidth(88)
        self.counter.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.timing = QLabel("")
        self.timing.setFixedWidth(56)
        self.timing.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        for widget in (self.glyph, self.name, self.bar, self.counter, self.timing):
            layout.addWidget(widget)

        self._paint(PENDING, COLOURS["text_dim"])

    def reset(self) -> None:
        """Return to the pending state."""
        self._started_at = None
        self._elapsed = 0.0
        self._done = 0
        self._total = 0
        self.bar.setVisible(False)
        self.counter.setText("")
        self.timing.setText("")
        self._paint(PENDING, COLOURS["text_dim"])

    def begin(self, total: int) -> None:
        """Mark the stage as running.

        Args:
            total: Units of work it declared.
        """
        self._started_at = time.monotonic()
        self._total = total
        self._done = 0
        self.bar.setVisible(total > 0)
        self.bar.setRange(0, max(total, 1))
        self.bar.setValue(0)
        self._paint(RUNNING, COLOURS["accent"])
        self._refresh()

    def update_progress(self, done: int) -> None:
        """Record how far the stage has got.

        Args:
            done: Units completed.
        """
        self._done = done
        if self._total > 0:
            self.bar.setValue(min(done, self._total))
        self._refresh()

    def end(self) -> None:
        """Mark the stage as complete and stop its clock."""
        if self._started_at is not None:
            self._elapsed = time.monotonic() - self._started_at
            self._started_at = None
        self.bar.setVisible(False)
        self.counter.setText("")
        self.timing.setText(format_duration(self._elapsed) if self._elapsed else "")
        self._paint(DONE, COLOURS["ok"])

    def fail(self) -> None:
        """Mark the stage as having stopped without finishing."""
        self._started_at = None
        self.bar.setVisible(False)
        self._paint(FAILED, COLOURS["error"])

    @property
    def elapsed(self) -> float:
        """How long the stage ran.

        Returns:
            Seconds, still counting while it runs.
        """
        if self._started_at is not None:
            return time.monotonic() - self._started_at
        return self._elapsed

    @property
    def remaining(self) -> float | None:
        """Estimated seconds left in this stage.

        Returns:
            The estimate, or ``None`` when there is not yet enough evidence for
            one worth showing.
        """
        if self._started_at is None or self._total <= 0 or self._done < _MIN_SAMPLES_FOR_ESTIMATE:
            return None
        elapsed = time.monotonic() - self._started_at
        return elapsed / self._done * (self._total - self._done)

    def _refresh(self) -> None:
        """Update the counter and the running clock."""
        self.counter.setText(f"{self._done}/{self._total}" if self._total > 0 else "")
        self.timing.setText(format_duration(self.elapsed))

    def _paint(self, glyph: str, colour: object) -> None:
        """Set the row's glyph and colour.

        Args:
            glyph: The state character.
            colour: The colour for it.
        """
        self.glyph.setText(glyph)
        self.glyph.setStyleSheet(f"color: {colour.name()};")  # type: ignore[attr-defined]
        dim = glyph == PENDING
        self.name.setStyleSheet(
            f"color: {COLOURS['text_dim'].name() if dim else COLOURS['text'].name()};"
        )


class StageList(QWidget):
    """Every stage of a run, drawn before the run begins."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the checklist.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.rows: dict[str, StageRow] = {}
        for stage in PIPELINE_STAGES:
            row = StageRow(stage)
            self.rows[str(stage)] = row
            layout.addWidget(row)

    def reset(self) -> None:
        """Return every row to pending."""
        for row in self.rows.values():
            row.reset()

    def report(self, description: str, done: int, total: int) -> None:
        """Apply one progress update.

        Takes the reporter's three-value signal rather than a richer structure,
        because that is what crosses the thread boundary. A stage starting is
        inferred from ``done == 0``, and any earlier stage still marked running
        is closed — a stage cannot be running once a later one has begun, and
        relying on the worker to say so would leave a row spinning forever if a
        run were interrupted between the two.

        Args:
            description: The stage name.
            done: Units completed.
            total: Units of work in the stage.
        """
        row = self.rows.get(description)
        if row is None:
            return
        if done == 0:
            self._close_earlier(description)
            row.begin(total)
        elif total > 0 and done >= total:
            row.update_progress(done)
            row.end()
        else:
            row.update_progress(done)

    def fail_running(self) -> None:
        """Mark whichever stage was running as stopped."""
        for row in self.rows.values():
            if row.glyph.text() == RUNNING:
                row.fail()

    @property
    def remaining(self) -> float | None:
        """Estimated seconds left in the current stage.

        Deliberately not a whole-run estimate. Extrapolating across stages needs
        per-stage rates this widget does not have, and an estimate assembled from
        guesses about work that has not started is worse than none.

        Returns:
            The estimate, or ``None`` when there is not one worth showing.
        """
        for row in self.rows.values():
            if row.glyph.text() == RUNNING:
                return row.remaining
        return None

    def _close_earlier(self, description: str) -> None:
        """End any stage still marked running before the one now starting.

        Args:
            description: The stage that has just begun.
        """
        for name, row in self.rows.items():
            if name != description and row.glyph.text() == RUNNING:
                row.end()
