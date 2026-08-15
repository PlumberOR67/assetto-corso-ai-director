"""The first-run page.

Inline rather than a modal wizard, and that is the whole design. A modal is
clicked through in four seconds and the application is exactly as opaque
afterwards; a page that sits where the results will later appear is read while
the user is deciding what to do, and it goes away the moment they do it.

The last paragraph about recording quality is not filler. It is this project's
honesty rule reaching the user *before* they spend twenty minutes analysing an
8 Hz recording that cannot support event timing at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from assetto_ai_director.presentation.gui.theme import COLOURS

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["STEPS", "WelcomePage"]

STEPS: tuple[tuple[str, str], ...] = (
    (
        "Open a replay.",
        (
            "Assetto Corsa keeps them in Documents\\Assetto Corsa\\replay. "
            "A telemetry CSV from another simulator works too."
        ),
    ),
    (
        "Pick a style.",
        (
            "initial_d holds long, locked-off shots on a slide. broadcast is Gran "
            "Turismo replay theatre. Each one says what it does when you select it."
        ),
    ),
    (
        "Press Run.",
        (
            "Six stages, each shown as it happens, with what is left. You can stop "
            "at any point without losing anything you already had."
        ),
    ),
    (
        "Take the results.",
        (
            "They go into a folder you choose and stay there. Each file says what it "
            "is and what to do with it."
        ),
    ),
)

_QUALITY_NOTE = (
    "Recording quality matters. Assetto Corsa's replay quality setting decides how "
    "precisely a braking point can be located, and no analysis recovers timing the "
    "file never stored. This tells you what your recording supports before you "
    "spend time on it."
)


class WelcomePage(QWidget):
    """What this application is, in four steps."""

    open_requested = Signal()
    output_change_requested = Signal()
    dismissed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the page.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 40, 48, 40)
        layout.setSpacing(18)

        title = QLabel("Assetto AI Director watches your replay and decides where the cameras go.")
        title.setObjectName("WelcomeTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        for index, (heading, detail) in enumerate(STEPS, start=1):
            layout.addWidget(self._step(index, heading, detail))

        layout.addSpacing(8)

        destination = QHBoxLayout()
        destination.setSpacing(8)
        self.output_label = QLabel()
        self.output_label.setWordWrap(True)
        self.output_label.setStyleSheet(f"color: {COLOURS['text_dim'].name()};")
        change = QPushButton("Change…")
        change.clicked.connect(self.output_change_requested.emit)
        destination.addWidget(self.output_label, 1)
        destination.addWidget(change)
        layout.addLayout(destination)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        open_button = QPushButton("Open a replay…")
        open_button.setObjectName("RunButton")
        open_button.setMinimumHeight(38)
        open_button.clicked.connect(self.open_requested.emit)
        skip = QPushButton("Skip this")
        skip.clicked.connect(self.dismissed.emit)
        buttons.addWidget(open_button)
        buttons.addWidget(skip)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        note = QLabel(_QUALITY_NOTE)
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {COLOURS['text_dim'].name()};")
        layout.addWidget(note)
        layout.addStretch(1)

    def show_output(self, folder: Path) -> None:
        """State where results will be written.

        Args:
            folder: The output root.
        """
        self.output_label.setText(f"Results will be saved to  {folder}")

    @staticmethod
    def _step(number: int, heading: str, detail: str) -> QWidget:
        """Build one numbered step.

        Args:
            number: Its position.
            heading: What to do.
            detail: Why, and where.

        Returns:
            The step widget.
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        index = QLabel(str(number))
        index.setFixedWidth(18)
        index.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        index.setStyleSheet(f"color: {COLOURS['accent'].name()};")

        text = QVBoxLayout()
        text.setSpacing(1)
        title = QLabel(heading)
        body = QLabel(detail)
        body.setWordWrap(True)
        body.setStyleSheet(f"color: {COLOURS['text_dim'].name()};")
        text.addWidget(title)
        text.addWidget(body)

        layout.addWidget(index)
        layout.addLayout(text, 1)
        return row
