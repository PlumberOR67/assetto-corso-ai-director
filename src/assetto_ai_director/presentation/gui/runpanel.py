"""The four steps, and the button.

Reads top to bottom as a numbered sequence, because the complaint that started
this was that it was not clear where to do what, or what to press. Numbering is
not decoration here; it is the answer.

**Step 3 is the cheapest fix in the whole rework.** ``DirectorWorker`` has always
accepted a ``seconds`` argument and the GUI never exposed it, so the only way to
try a style was to commit to the entire session. A user's first run was
twenty-four minutes long. Three minutes is the default now: long enough to tell
whether the style is right, short enough to try another.

**The button carries the estimate.** "Run (≈3 min)" cold, "Re-run with mf_ghost
(≈50 s)" when only the style moved — because the analysis is style-independent
and cached, so that really is most of the cost gone. A user who can see what a
click will cost does not have to guess whether to make it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.presentation.gui.project import Phase
from assetto_ai_director.presentation.gui.stages import format_duration
from assetto_ai_director.presentation.gui.theme import COLOURS

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["RunPanel"]

_SECONDS_PER_MINUTE = 60.0

_STEP_HINTS = {
    3: "Start with a few minutes to try a style. You can run the whole session after.",
}


class _Step(QWidget):
    """A numbered section of the panel."""

    def __init__(self, number: int, title: str, parent: QWidget | None = None) -> None:
        """Build a step.

        Args:
            number: Its position in the sequence.
            title: What the step is for.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(4)

        heading = QLabel(f"{number}   {title}")
        heading.setObjectName("StepHeading")
        self.body.addWidget(heading)

        hint = _STEP_HINTS.get(number)
        if hint:
            label = QLabel(hint)
            label.setWordWrap(True)
            label.setStyleSheet(f"color: {COLOURS['text_dim'].name()};")
            self.body.addWidget(label)


class RunPanel(QWidget):
    """Steps one to four, plus Run and Cancel."""

    open_requested = Signal()
    run_requested = Signal()
    cancel_requested = Signal()
    output_change_requested = Signal()
    open_output_requested = Signal()
    settings_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the panel.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        layout.addWidget(self._build_session_step())
        layout.addWidget(self._build_style_step())
        layout.addWidget(self._build_scope_step())
        layout.addWidget(self._build_output_step())
        layout.addWidget(self._build_run_button())
        layout.addStretch(1)

    # --- Construction ------------------------------------------------------

    def _build_session_step(self) -> QWidget:
        """Build step one.

        Returns:
            The step widget.
        """
        step = _Step(1, "SESSION")
        self.open_button = QPushButton("Open a replay…")
        self.open_button.clicked.connect(self.open_requested.emit)
        step.body.addWidget(self.open_button)
        return step

    def _build_style_step(self) -> QWidget:
        """Build step two.

        Returns:
            The step widget.
        """
        step = _Step(2, "STYLE")
        self.style_box = QComboBox()
        self.style_box.addItems(sorted(BUILTIN_STYLES))
        self.style_box.currentTextChanged.connect(self._on_style_changed)
        step.body.addWidget(self.style_box)

        # StyleProfile.description has existed since the styles did and the GUI
        # never showed it, leaving five names with no way to tell them apart.
        self.style_description = QLabel()
        self.style_description.setWordWrap(True)
        self.style_description.setStyleSheet(f"color: {COLOURS['text_dim'].name()};")
        step.body.addWidget(self.style_description)
        self._on_style_changed(self.style_box.currentText())
        return step

    def _build_scope_step(self) -> QWidget:
        """Build step three.

        Returns:
            The step widget.
        """
        step = _Step(3, "HOW MUCH")
        self.scope_group = QButtonGroup(self)

        whole = QHBoxLayout()
        self.whole_radio = QRadioButton("Whole session")
        self.scope_group.addButton(self.whole_radio)
        whole.addWidget(self.whole_radio)
        whole.addStretch(1)
        self.whole_length = QLabel("")
        self.whole_length.setStyleSheet(f"color: {COLOURS['text_dim'].name()};")
        whole.addWidget(self.whole_length)
        step.body.addLayout(whole)

        partial = QHBoxLayout()
        self.partial_radio = QRadioButton("First")
        self.partial_radio.setChecked(True)
        self.scope_group.addButton(self.partial_radio)
        self.minutes = QSpinBox()
        self.minutes.setRange(1, 600)
        self.minutes.setValue(3)
        self.minutes.setSuffix(" min")
        partial.addWidget(self.partial_radio)
        partial.addWidget(self.minutes)
        partial.addStretch(1)
        step.body.addLayout(partial)

        self.whole_radio.toggled.connect(self._on_scope_changed)
        self.minutes.valueChanged.connect(self._on_scope_changed)
        return step

    def _build_output_step(self) -> QWidget:
        """Build step four.

        Returns:
            The step widget.
        """
        step = _Step(4, "RESULTS GO TO")
        self.output_label = QLabel()
        self.output_label.setWordWrap(True)
        step.body.addWidget(self.output_label)

        buttons = QHBoxLayout()
        change = QPushButton("Change…")
        change.clicked.connect(self.output_change_requested.emit)
        reveal = QPushButton("Open folder")
        reveal.clicked.connect(self.open_output_requested.emit)
        buttons.addWidget(change)
        buttons.addWidget(reveal)
        buttons.addStretch(1)
        step.body.addLayout(buttons)
        return step

    def _build_run_button(self) -> QWidget:
        """Build the Run and Cancel controls.

        Returns:
            The container widget.
        """
        container = QFrame()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("RunButton")
        self.run_button.setMinimumHeight(40)
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_requested.emit)
        layout.addWidget(self.run_button)

        self.cancel_button = QPushButton("Stop")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        layout.addWidget(self.cancel_button)
        return container

    # --- State -------------------------------------------------------------

    @property
    def style_name(self) -> str:
        """The chosen style name.

        Not ``style``: that is ``QWidget.style()``, which returns the widget's
        QStyle and is called by Qt itself. mypy caught this one; at runtime it
        would have surfaced as an inexplicable painting failure.

        Returns:
            The name.
        """
        return self.style_box.currentText()

    @property
    def scope_seconds(self) -> float:
        """How much of the session to direct.

        Returns:
            The span in seconds; zero means all of it.
        """
        if self.whole_radio.isChecked():
            return 0.0
        return float(self.minutes.value()) * _SECONDS_PER_MINUTE

    def restore(self, style: str, scope_seconds: float) -> None:
        """Apply remembered settings without emitting change signals.

        Args:
            style: The style to select.
            scope_seconds: The span; zero selects the whole session.
        """
        blocked = [self.style_box, self.whole_radio, self.minutes]
        for widget in blocked:
            widget.blockSignals(True)
        try:
            if style in BUILTIN_STYLES:
                self.style_box.setCurrentText(style)
            if scope_seconds <= 0.0:
                self.whole_radio.setChecked(True)
            else:
                self.partial_radio.setChecked(True)
                self.minutes.setValue(max(1, round(scope_seconds / _SECONDS_PER_MINUTE)))
        finally:
            for widget in blocked:
                widget.blockSignals(False)
        self._on_style_changed(self.style_box.currentText())

    def show_session_length(self, seconds: float) -> None:
        """State how long the open session runs, next to the whole-session choice.

        Args:
            seconds: The session's duration.
        """
        self.whole_length.setText(format_duration(seconds))

    def show_output(self, folder: Path) -> None:
        """State where results will be written.

        Args:
            folder: The destination for the current settings.
        """
        self.output_label.setText(str(folder))
        self.output_label.setToolTip(str(folder))

    def apply_phase(self, phase: Phase, estimate: float | None = None) -> None:
        """Enable the controls that make sense in this phase.

        Args:
            phase: What the window can currently do.
            estimate: Seconds the next run is expected to take, when known.
        """
        running = phase is Phase.RUNNING
        self.open_button.setEnabled(not running)
        self.style_box.setEnabled(not running)
        self.whole_radio.setEnabled(not running)
        self.partial_radio.setEnabled(not running)
        self.minutes.setEnabled(not running)
        self.run_button.setEnabled(phase.can_run)
        self.run_button.setVisible(not running)
        self.cancel_button.setVisible(running)
        self.run_button.setText(self._run_label(phase, estimate))

    def _run_label(self, phase: Phase, estimate: float | None) -> str:
        """Compose the Run button's text.

        Args:
            phase: The current phase.
            estimate: Seconds the run is expected to take.

        Returns:
            The label, carrying the cost of the click where it is known.
        """
        verb = "Re-run" if phase in (Phase.STALE, Phase.READY, Phase.FAILED) else "Run"
        if phase is Phase.STALE:
            verb = f"Re-run with {self.style_name}"
        if estimate is None:
            return verb
        return f"{verb}  (≈{format_duration(estimate)})"

    # --- Signals -----------------------------------------------------------

    def _on_style_changed(self, name: str) -> None:
        """Show the chosen style's description and report the change.

        Args:
            name: The newly selected style.
        """
        profile = BUILTIN_STYLES.get(name)
        self.style_description.setText(profile.description if profile else "")
        self.settings_changed.emit()

    def _on_scope_changed(self) -> None:
        """Report a change of span."""
        self.minutes.setEnabled(self.partial_radio.isChecked())
        self.settings_changed.emit()
