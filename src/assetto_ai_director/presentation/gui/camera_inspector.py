"""The Cameras tab: what every camera sees, and one button that hands it over.

This is the whole point of the diagnostic work, from the point of view of the
person using it. They open a replay, press Run as they always have, come to this
tab, and press **Copy report**. What lands on the clipboard is a document
somebody else — or an assistant — can act on without ever seeing their screen.

**No dialogs, no paths, no arguments.** Everything the report needs is already in
the window by the time this tab is populated, and everything about the machine is
discovered rather than asked for. A verification step with a prerequisite is a
verification step that gets skipped exactly when it matters.

**Borders are coloured by the measured verdict.** Two checks cannot run until the
track's geometry can be read, so every camera's strict verdict is at best
unknown; a grid of forty identical borders would tell a reader nothing. The
strict verdict stays visible in the panel beside the grid, where it can be read
rather than merely glanced at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStyleOptionGraphicsItem,
    QVBoxLayout,
    QWidget,
)

from assetto_ai_director.application.diagnostics.scene import StrokeKind
from assetto_ai_director.domain.diagnostics import CheckStatus
from assetto_ai_director.presentation.gui.theme import COLOURS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from assetto_ai_director.application.diagnostics.scene import Preview, Stroke
    from assetto_ai_director.domain.diagnostics import CameraDiagnostic, DiagnosticReport

__all__ = ["CameraInspector"]

CELL_WIDTH = 320.0
CELL_HEIGHT = 180.0
_GAP = 10.0
_COLUMNS = 3

_OUTSIDE = 1.6
"""How far past the frame edge geometry is still drawn, in frame half-widths.

Matches the SVG contact sheet, so the picture on screen and the picture in the
file are the same picture.
"""

_MIN_STROKE_POINTS = 2

_STATUS_COLOUR: dict[CheckStatus, str] = {
    CheckStatus.PASS: "ok",
    CheckStatus.WARN: "warning",
    CheckStatus.FAIL: "error",
    CheckStatus.UNKNOWN: "accent",
}

_STROKE_COLOUR: dict[StrokeKind, tuple[str, float]] = {
    StrokeKind.HORIZON: ("quiet", 1.0),
    StrokeKind.OTHER_PATH: ("quiet", 1.0),
    StrokeKind.SUBJECT_PATH: ("corner", 1.4),
    StrokeKind.OTHER_CAR: ("drift", 1.2),
    StrokeKind.SUBJECT_CAR: ("warning", 2.0),
    StrokeKind.FRAME_CENTRE: ("error", 1.0),
}


class _PreviewItem(QGraphicsItem):
    """One camera's view, painted as a single item.

    A single item rather than one per stroke, for the same reason the track map
    is one item: a scene graph holding thousands of tiny items spends all its
    time on bookkeeping.

    Attributes:
        preview: What to draw.
        index: Position in the grid, used to place it.
    """

    def __init__(self, preview: Preview, index: int) -> None:
        """Place one preview in the grid.

        Args:
            preview: What to draw.
            index: Position in the grid.
        """
        super().__init__()
        self.preview = preview
        self.index = index
        self.setPos(
            (index % _COLUMNS) * (CELL_WIDTH + _GAP),
            (index // _COLUMNS) * (CELL_HEIGHT + _GAP),
        )
        self.setAcceptHoverEvents(False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

    def boundingRect(self) -> QRectF:
        """Return the cell's bounds.

        Returns:
            The rectangle.
        """
        return QRectF(0.0, 0.0, CELL_WIDTH, CELL_HEIGHT)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint the preview.

        Args:
            painter: The painter.
            option: Style options, unused.
            widget: The widget being painted on, unused.
        """
        del option, widget
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.boundingRect(), COLOURS["surface"])

        # The frame the camera actually sees sits inside the cell, so geometry
        # it misses has somewhere to be drawn. A preview cropped to the frame
        # cannot show a car the camera just missed.
        inset_x = CELL_WIDTH / (2.0 * _OUTSIDE)
        inset_y = CELL_HEIGHT / (2.0 * _OUTSIDE)
        frame = QRectF(
            CELL_WIDTH / 2.0 - inset_x, CELL_HEIGHT / 2.0 - inset_y, inset_x * 2, inset_y * 2
        )
        pen = QPen(COLOURS["border"], 1.0, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(frame)

        for stroke in self.preview.strokes:
            self._paint_stroke(painter, stroke)

        border = QColor(COLOURS[_STATUS_COLOUR[self.preview.measured_status]])
        painter.setPen(QPen(border, 3.0 if self.isSelected() else 2.0))
        painter.drawRect(self.boundingRect().adjusted(1.0, 1.0, -1.0, -1.0))

        painter.setPen(QPen(COLOURS["text_dim"], 1.0))
        painter.drawText(
            QRectF(6.0, CELL_HEIGHT - 18.0, CELL_WIDTH - 12.0, 14.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self.preview.label,
        )

    def _paint_stroke(self, painter: QPainter, stroke: Stroke) -> None:
        """Paint one projected polyline.

        Args:
            painter: The painter.
            stroke: The polyline.
        """
        name, width = _STROKE_COLOUR[stroke.kind]
        painter.setPen(QPen(COLOURS[name], width))

        if stroke.kind is StrokeKind.FRAME_CENTRE:
            x, y = _to_cell(0.0, 0.0)
            painter.drawLine(int(x - 5), int(y), int(x + 5), int(y))
            painter.drawLine(int(x), int(y - 5), int(x), int(y + 5))
            return

        if len(stroke.points) < _MIN_STROKE_POINTS:
            return
        path = QPainterPath()
        drawing = False
        for u, v in stroke.points:
            if abs(u) > _OUTSIDE or abs(v) > _OUTSIDE:
                drawing = False
                continue
            x, y = _to_cell(u, v)
            if drawing:
                path.lineTo(x, y)
            else:
                path.moveTo(x, y)
                drawing = True
        if not path.isEmpty():
            painter.drawPath(path)


class CameraInspector(QWidget):
    """A grid of camera previews, the checks behind them, and a Copy button."""

    copy_requested = Signal()
    save_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build an empty inspector.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._report: DiagnosticReport | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self.summary = QLabel("Run a session to check its cameras.")
        self.summary.setWordWrap(True)
        header.addWidget(self.summary, 1)

        self.copy_button = QPushButton("Copy report")
        self.copy_button.setToolTip(
            "Put the report on the clipboard, ready to paste into a conversation."
        )
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self.copy_requested.emit)
        header.addWidget(self.copy_button)

        self.save_button = QPushButton("Save contact sheet")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_requested.emit)
        header.addWidget(self.save_button)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.view = QGraphicsView()
        self.view.setScene(QGraphicsScene(self))
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.view.setBackgroundBrush(COLOURS["background"])
        self.view.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.view.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.view.scene().selectionChanged.connect(self._on_selected)
        splitter.addWidget(self.view)

        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText("Select a camera to see every check behind it.")
        splitter.addWidget(self.detail)
        splitter.setSizes([2, 1])
        layout.addWidget(splitter, 1)

    def show_report(self, report: DiagnosticReport, previews: Sequence[Preview]) -> None:
        """Display a diagnosis.

        Args:
            report: The diagnosis.
            previews: One preview per camera, in the same order.
        """
        self._report = report
        scene = self.view.scene()
        scene.clear()
        for index, preview in enumerate(previews):
            scene.addItem(_PreviewItem(preview, index))
        scene.setSceneRect(scene.itemsBoundingRect())

        tally = report.tally()
        self.summary.setText(
            f"{len(report.cameras)} cameras — "
            f"{report.measured_status.value} on what could be measured, "
            f"{report.status.value} overall. "
            f"{tally[CheckStatus.FAIL]} failing, {tally[CheckStatus.WARN]} warning. "
            f"{len(report.blind_spots)} thing(s) could not be checked."
        )
        self.copy_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.detail.setPlainText(_overview(report))

    def show_unavailable(self, reason: str) -> None:
        """Say why there is nothing to show.

        A blank tab is indistinguishable from a tab full of passing cameras at a
        glance, and a person who came here to verify something would draw the
        wrong conclusion from either. Stating the reason costs one line and
        removes the ambiguity.

        Args:
            reason: What is missing, in words.
        """
        self._report = None
        self.view.scene().clear()
        self.summary.setText(reason)
        self.detail.setPlainText(reason)
        self.copy_button.setEnabled(False)
        self.save_button.setEnabled(False)

    def clear(self) -> None:
        """Empty the inspector."""
        self._report = None
        self.view.scene().clear()
        self.summary.setText("Run a session to check its cameras.")
        self.detail.clear()
        self.copy_button.setEnabled(False)
        self.save_button.setEnabled(False)

    def _on_selected(self) -> None:
        """Show every check for the selected camera."""
        if self._report is None:
            return
        selected = self.view.scene().selectedItems()
        if not selected:
            self.detail.setPlainText(_overview(self._report))
            return
        item = selected[0]
        if not isinstance(item, _PreviewItem):
            return
        for camera in self._report.cameras:
            if camera.shot_index == item.preview.shot_index:
                self.detail.setPlainText(_camera_detail(camera))
                return


def _to_cell(u: float, v: float) -> tuple[float, float]:
    """Map normalised frame coordinates onto a cell.

    Args:
        u: Horizontal position, ``-1`` at the frame's left edge.
        v: Vertical position, ``-1`` at the frame's bottom edge.

    Returns:
        ``(x, y)`` in cell coordinates, with the vertical axis flipped because
        Qt's grows downward and the frame's grows upward.
    """
    return (u / _OUTSIDE + 1.0) * 0.5 * CELL_WIDTH, (1.0 - v / _OUTSIDE) * 0.5 * CELL_HEIGHT


def _overview(report: DiagnosticReport) -> str:
    """Summarise a report for the detail panel.

    Args:
        report: The diagnosis.

    Returns:
        Plain text.
    """
    lines = ["Every check, across every camera:", ""]
    lines.extend(
        f"  {name:<30} "
        + "  ".join(
            f"{status.value} {counts[status]}"
            for status in (
                CheckStatus.PASS,
                CheckStatus.WARN,
                CheckStatus.FAIL,
                CheckStatus.UNKNOWN,
            )
            if counts[status]
        )
        for name, counts in report.check_tally().items()
    )
    if report.blind_spots:
        lines.extend(["", "What could not be checked:", ""])
        lines.extend(f"  - {gap}" for gap in report.blind_spots)
    if report.assumptions:
        lines.extend(["", "What this report assumes:", ""])
        lines.extend(f"  - {line}" for line in report.assumptions)
    return "\n".join(lines)


def _camera_detail(camera: CameraDiagnostic) -> str:
    """Render every check for one camera.

    Args:
        camera: The diagnostic.

    Returns:
        Plain text.
    """
    heading = (
        f"Shot {camera.shot_index} — {camera.rig.value}, "
        f"{camera.size.name.lower()}, {camera.side.value}"
    )
    when = (
        f"{float(camera.start):.1f}s to {float(camera.end):.1f}s, "
        f"cars {', '.join(str(car) for car in camera.subjects)}"
    )
    where = (
        f"camera at ({camera.position.x:.1f}, {camera.position.y:.1f}, "
        f"{camera.position.z:.1f}), {camera.fov:.0f} degree lens"
    )
    lines = [heading, when, where, ""]
    for check in camera.checks:
        value = "not measured" if check.value is None else f"{check.value:.4g} {check.unit}".strip()
        lines.append(f"[{check.status.value.upper():<7}] {check.name}: {value}")
        lines.append(f"          {check.detail}")
        lines.append(f"          threshold: {check.threshold}")
        lines.append("")
    return "\n".join(lines)
