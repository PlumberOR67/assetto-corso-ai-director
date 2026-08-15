"""Which session is open, and what it is.

A user asked which replay it was working on, and the answer was that the window
never said. The filename appeared briefly in the status bar during loading and
was overwritten by "Done." The window title was set once at construction and
never touched again.

Everything shown here was already being computed and thrown away. ``SessionMeta``
carries the track, layout, weather, sample rate, frame count and every car's
identity; ``CarIdentity`` has a ``display_name`` whose docstring says it exists
"for logs, **the GUI** and shot metadata"; ``RecordingQuality`` has an ``advice``
string written for a user, and ``domain/session.py`` claims the rate is "surfaced
in the GUI". None of it ever was.

Two things get a warning rather than a line of text, because both silently change
what the analysis means: a **wrapped** recording begins mid-session, so its first
frames are the most recent ones in wall-clock terms; and a **coarse** recording
rate cannot place an event precisely however good the detector is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from assetto_ai_director.domain.session import RecordingQuality
from assetto_ai_director.presentation.gui.theme import COLOURS

if TYPE_CHECKING:
    from pathlib import Path

    from assetto_ai_director.domain.session import SessionMeta

__all__ = ["SessionHeader", "format_clock"]

_BYTES_PER_MIB = 1024 * 1024


def format_clock(seconds: float) -> str:
    """Render a session length as a clock.

    Args:
        seconds: The duration.

    Returns:
        ``M:SS`` under an hour, ``H:MM:SS`` above.
    """
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class SessionHeader(QFrame):
    """A permanent statement of what is open."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build an empty header.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setObjectName("SessionHeader")
        self.setFrameShape(QFrame.Shape.NoFrame)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 10, 16, 10)
        outer.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(10)
        self.filename = QLabel()
        self.filename.setObjectName("HeaderFilename")
        self.filename.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # Not `self.size`: that is QWidget.size(), which Qt calls on every
        # layout pass. Shadowing it breaks the widget at runtime, silently.
        self.file_size = QLabel()
        self.file_size.setStyleSheet(f"color: {COLOURS['text_dim'].name()};")
        top.addWidget(self.filename)
        top.addWidget(self.file_size)
        outer.addLayout(top)

        middle = QHBoxLayout()
        middle.setSpacing(10)
        self.detail = QLabel()
        self.detail.setStyleSheet(f"color: {COLOURS['text_dim'].name()};")
        self.detail.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.quality = QLabel()
        self.quality.setObjectName("QualityBadge")
        self.quality.setAlignment(Qt.AlignmentFlag.AlignCenter)
        middle.addWidget(self.detail)
        middle.addWidget(self.quality)
        outer.addLayout(middle)

        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(f"color: {COLOURS['warning'].name()};")
        self.warning.setVisible(False)
        outer.addWidget(self.warning)

        self.clear()

    def clear(self) -> None:
        """Show the empty state."""
        self.filename.setText("No session open")
        self.filename.setStyleSheet(f"color: {COLOURS['text_dim'].name()};")
        self.file_size.setText("")
        self.detail.setText("Open a replay to begin.")
        self.quality.setText("")
        self.quality.setVisible(False)
        self.warning.setVisible(False)

    def show_session(self, path: Path, meta: SessionMeta, skipped: int = 0) -> None:
        """Describe an open session.

        Args:
            path: The session file.
            meta: Its metadata.
            skipped: How many cars were dropped for never moving. Zero before a
                run has decoded anything, which is why it is separate from the
                car count rather than folded into it.
        """
        self.filename.setText(path.name)
        self.filename.setStyleSheet(f"color: {COLOURS['text'].name()};")
        self.file_size.setText(self._file_size(path))
        self.detail.setText(self._describe(meta, skipped))
        self._show_quality(meta.quality)
        self._show_warnings(meta)

    @staticmethod
    def _file_size(path: Path) -> str:
        """Render a file's size.

        Args:
            path: The file.

        Returns:
            The size in MiB, or empty when it cannot be read — a file on a
            disconnected drive should not stop the header from drawing.
        """
        try:
            return f"{path.stat().st_size / _BYTES_PER_MIB:.0f} MB"
        except OSError:
            return ""

    @staticmethod
    def _describe(meta: SessionMeta, skipped: int) -> str:
        """Build the one-line summary.

        Args:
            meta: Session metadata.
            skipped: Cars dropped for never moving.

        Returns:
            Track, duration, car count and who the player is.
        """
        cars = f"{meta.car_count} car{'s' if meta.car_count != 1 else ''}"
        if skipped:
            cars += f" ({skipped} parked)"
        parts = [meta.track_label, format_clock(float(meta.duration)), cars]
        player = meta.player()
        if player is not None:
            parts.append(f"you: {player.display_name}")
        return "  ·  ".join(parts)

    def _show_quality(self, quality: RecordingQuality) -> None:
        """Show the recording-quality badge.

        Args:
            quality: The band this session's rate falls into.
        """
        colour = {
            RecordingQuality.UNUSABLE: COLOURS["error"],
            RecordingQuality.COARSE: COLOURS["warning"],
            RecordingQuality.GOOD: COLOURS["ok"],
            RecordingQuality.EXCELLENT: COLOURS["ok"],
        }[quality]
        self.quality.setText(quality.name)
        self.quality.setToolTip(quality.advice)
        self.quality.setStyleSheet(
            f"color: {colour.name()}; border: 1px solid {colour.name()};"
            f" border-radius: 3px; padding: 1px 6px;"
        )
        self.quality.setVisible(True)

    def _show_warnings(self, meta: SessionMeta) -> None:
        """Show anything that changes what the analysis means.

        Args:
            meta: Session metadata.
        """
        warnings: list[str] = []
        if meta.wrapped:
            warnings.append(
                "This recording wrapped: Assetto Corsa's ring buffer overwrote the start, "
                "so the file begins mid-session and its earliest frames are the most "
                "recent ones."
            )
        if not meta.quality.supports_event_timing:
            warnings.append(meta.quality.advice)
        self.warning.setText("  ".join(warnings))
        self.warning.setVisible(bool(warnings))
