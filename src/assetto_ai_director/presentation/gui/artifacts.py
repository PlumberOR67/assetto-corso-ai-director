"""What the run produced, where it is, and what each file is for.

The question that prompted this panel: why can I save "some timeline", and what
am I supposed to do with it? The old window offered ``Ctrl+S`` -> "Export
timeline..." and a save dialog, which answers neither half.

So each file carries a sentence saying what it is and what you do with it, and a
button that opens the folder containing it. "Where does it save to" stops being a
question the moment the answer is on screen next to the file — and it is shown
*before* the run, not announced in a status bar afterwards.

**Lossy exports are written, and their limitations shown permanently.** The CLI
refuses a lossy export without ``--force``, and it is right to: a terminal prints
once and scrolls away, so refusing is the only way to make someone read the
caveat. A panel can keep the caveat next to the file forever, so refusing would
only add a click before the same outcome. That divergence is deliberate and is
recorded here so it reads as a decision rather than a drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from assetto_ai_director.presentation.gui.theme import COLOURS

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["EXPLANATIONS", "Artifact", "ArtifactsPanel"]

_BYTES_PER_KIB = 1024
_BYTES_PER_MIB = 1024 * 1024

EXPLANATIONS: dict[str, str] = {
    "timeline": (
        "The edit itself — every shot, its camera move, and when it cuts. "
        "This is what the camera set and, later, video export are made from. "
        "Keep it to re-export without analysing again."
    ),
    "cameras-ini": (
        "A trackside camera set for Assetto Corsa. Copy it into the track's "
        "layout folder, then pick camera set 9 in game."
    ),
    "diagnostic": (
        "What every camera would actually see, checked without the game — with "
        "the number behind each verdict and a list of what could not be checked. "
        "The Cameras tab shows the same thing; this is the copy you can send."
    ),
    "csv": (
        "The decoded telemetry as a table. Open it in a spreadsheet to check "
        "the analysis against what you remember driving."
    ),
    "project": "What this run was: the session, the style, the span, how long each stage took.",
}
"""One sentence per kind, written for the person who has to decide whether to keep
the file. Held in a dict rather than on the dataclass so the same text can be
reused by the help page without a second copy drifting from this one."""


@dataclass(frozen=True, slots=True)
class Artifact:
    """One file a run produced.

    Attributes:
        path: Where it is.
        kind: Which explanation applies. An open string rather than an enum
            because video export will add to it and nothing here needs to switch
            exhaustively on the value.
        bytes: Size on disk.
        limitations: What the target format could not represent, from the sink's
            own support report.
        error: Why it could not be written, when it could not.
    """

    path: Path
    kind: str
    bytes: int = 0
    limitations: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None

    @property
    def explanation(self) -> str:
        """What this file is for.

        Returns:
            The sentence for this kind, or empty for an unknown one.
        """
        return EXPLANATIONS.get(self.kind, "")

    @property
    def size_label(self) -> str:
        """The size, rendered for a person.

        Returns:
            A short string in KB or MB.
        """
        if self.bytes >= _BYTES_PER_MIB:
            return f"{self.bytes / _BYTES_PER_MIB:.1f} MB"
        if self.bytes >= _BYTES_PER_KIB:
            return f"{self.bytes / _BYTES_PER_KIB:.0f} KB"
        return f"{self.bytes} B"


class ArtifactRow(QFrame):
    """One file, with its explanation and its caveats."""

    def __init__(self, artifact: Artifact, parent: QWidget | None = None) -> None:
        """Build a row.

        Args:
            artifact: The file to describe.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.artifact = artifact
        self.setObjectName("ArtifactRow")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(8)
        name = QLabel(artifact.path.name)
        name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        top.addWidget(name)

        if artifact.error is None:
            size = QLabel(artifact.size_label)
            size.setStyleSheet(f"color: {COLOURS['text_dim'].name()};")
            top.addWidget(size)
            self.open_button = QPushButton("Open")
            self.open_button.clicked.connect(self._open_file)
            self.folder_button = QPushButton("Folder")
            self.folder_button.clicked.connect(self._open_folder)
            top.addWidget(self.open_button)
            top.addWidget(self.folder_button)
        else:
            name.setStyleSheet(f"color: {COLOURS['error'].name()};")
        outer.addLayout(top)

        if artifact.explanation:
            explanation = QLabel(artifact.explanation)
            explanation.setWordWrap(True)
            explanation.setStyleSheet(f"color: {COLOURS['text_dim'].name()};")
            outer.addWidget(explanation)

        for limitation in artifact.limitations:
            caveat = QLabel(f"⚠ {limitation}")
            caveat.setWordWrap(True)
            caveat.setStyleSheet(f"color: {COLOURS['warning'].name()};")
            outer.addWidget(caveat)

        if artifact.error is not None:
            failure = QLabel(f"Could not write this file: {artifact.error}")
            failure.setWordWrap(True)
            failure.setStyleSheet(f"color: {COLOURS['error'].name()};")
            outer.addWidget(failure)

    def _open_file(self) -> None:
        """Open the file in whatever the system associates with it."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.artifact.path)))

    def _open_folder(self) -> None:
        """Open the folder containing the file."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.artifact.path.parent)))


class ArtifactsPanel(QWidget):
    """Everything a run produced, with its destination stated."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build an empty panel.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

        self.destination = QLabel()
        self.destination.setWordWrap(True)
        self.destination.setStyleSheet(f"color: {COLOURS['text_dim'].name()};")
        self._layout.addWidget(self.destination)

        self.empty = QLabel("Nothing written yet. Results appear here after a run.")
        self.empty.setWordWrap(True)
        self.empty.setStyleSheet(f"color: {COLOURS['text_dim'].name()};")
        self._layout.addWidget(self.empty)
        self._layout.addStretch(1)

        self.rows: list[ArtifactRow] = []

    def show_destination(self, folder: Path) -> None:
        """State where the next run will write, before it runs.

        Args:
            folder: The output folder for the current settings.
        """
        self.destination.setText(f"Results go to  {folder}")

    def show_artifacts(self, artifacts: list[Artifact]) -> None:
        """Replace the listed files.

        Not named ``show``: that is ``QWidget.show``, which Qt calls itself
        during layout, and shadowing it with a different signature would break
        the widget in a way that only appears at runtime.

        Args:
            artifacts: What the run produced, in the order to show them.
        """
        for row in self.rows:
            self._layout.removeWidget(row)
            row.deleteLater()
        self.rows = []

        self.empty.setVisible(not artifacts)
        for index, artifact in enumerate(artifacts):
            row = ArtifactRow(artifact)
            self.rows.append(row)
            # Before the trailing stretch, after the destination and empty labels.
            self._layout.insertWidget(2 + index, row)

    def open_folder(self, folder: Path) -> None:
        """Show a folder in the system file manager.

        Args:
            folder: The folder to open.
        """
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
