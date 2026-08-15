"""The timeline: shots as blocks, events as ticks, with a draggable playhead.

**One item per shot, not per frame.** A sixty-shot edit is sixty items, which the
scene graph handles comfortably. Adding an item per telemetry sample would be the
same mistake the track map avoids.

The ruler is drawn in ``drawBackground`` rather than as items. It is decoration
that never needs hit-testing, and giving it items would put thousands of ticks
into the scene graph for no benefit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QRect, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from assetto_ai_director.domain.events import EventKind
from assetto_ai_director.presentation.gui.theme import COLOURS

if TYPE_CHECKING:
    from assetto_ai_director.domain.events import Event
    from assetto_ai_director.domain.shot import Shot
    from assetto_ai_director.domain.timeline import Timeline

__all__ = ["TimelineView"]

PIXELS_PER_SECOND = 24.0
SHOT_TOP = 26.0
SHOT_HEIGHT = 44.0
EVENT_TOP = 4.0
EVENT_HEIGHT = 16.0

_MAX_RULER_TICKS = 40.0
"""Most ticks the ruler draws before choosing a coarser spacing.

Beyond this the labels overlap into an unreadable smear.
"""

_RIG_COLOURS = {
    "roadside": COLOURS["corner"],
    "static": COLOURS["corner"].darker(120),
    "follow": COLOURS["accent"],
    "drone": COLOURS["overtake"],
    "helicopter": COLOURS["overtake"].darker(130),
    "orbital": COLOURS["overtake"].lighter(120),
    "bonnet": COLOURS["drift"],
    "interior": COLOURS["drift"].darker(120),
    "roof": COLOURS["drift"].lighter(115),
    "wheel": COLOURS["risk"],
    "suspension": COLOURS["risk"].darker(120),
    "spline": COLOURS["quiet"],
}

_EVENT_COLOURS = {
    EventKind.DRIFT_START: COLOURS["drift"],
    EventKind.DRIFT_TRANSITION: COLOURS["drift"].lighter(130),
    EventKind.SPIN: COLOURS["risk"],
    EventKind.APEX: COLOURS["corner"],
    EventKind.OVERTAKE_COMPLETE: COLOURS["overtake"],
    EventKind.BRAKING_POINT: COLOURS["text_dim"],
}


class TimelineView(QGraphicsView):
    """Shows a directed edit as blocks on a time axis."""

    scrubbed = Signal(float)
    shot_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty timeline.

        Args:
            parent: The parent widget.
        """
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setBackgroundBrush(COLOURS["background"])
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setFixedHeight(int(SHOT_TOP + SHOT_HEIGHT + 24))

        self._duration = 0.0
        self._playhead: QGraphicsRectItem | None = None
        self._shots: list[tuple[int, QGraphicsRectItem]] = []

    def show_timeline(self, timeline: Timeline, events: tuple[Event, ...] = ()) -> None:
        """Display an edit.

        Args:
            timeline: The directed timeline.
            events: Detected events, drawn as ticks above the shots.
        """
        scene = self.scene()
        scene.clear()
        self._playhead = None
        self._shots = []
        self._duration = float(timeline.end)

        for event in events:
            self._add_event(scene, event)
        for shot in timeline.shots:
            self._add_shot(scene, shot)

        scene.setSceneRect(
            0.0, 0.0, max(self._duration * PIXELS_PER_SECOND, 1.0), SHOT_TOP + SHOT_HEIGHT + 8
        )
        self._add_playhead(scene)

    def _add_event(self, scene: QGraphicsScene, event: Event) -> None:
        """Draw one event tick.

        Args:
            scene: The scene.
            event: The event.
        """
        colour = _EVENT_COLOURS.get(event.kind, COLOURS["quiet"])
        x = float(event.start) * PIXELS_PER_SECOND
        width = max(float(event.duration) * PIXELS_PER_SECOND, 2.0)
        item = scene.addRect(
            x, EVENT_TOP, width, EVENT_HEIGHT, QPen(Qt.PenStyle.NoPen), QBrush(colour)
        )
        item.setOpacity(0.75)
        item.setToolTip(f"{event.kind.value} · {float(event.focus):.1f}s")

    def _add_shot(self, scene: QGraphicsScene, shot: Shot) -> None:
        """Draw one shot block.

        Args:
            scene: The scene.
            shot: The shot.
        """
        candidate = shot.candidate
        colour = _RIG_COLOURS.get(candidate.rig.value, COLOURS["surface_high"])
        x = float(shot.start) * PIXELS_PER_SECOND
        width = max(float(shot.duration) * PIXELS_PER_SECOND - 1.0, 2.0)

        item = scene.addRect(
            x,
            SHOT_TOP,
            width,
            SHOT_HEIGHT,
            QPen(COLOURS["border"], 1.0),
            QBrush(colour.darker(160)),
        )
        because = candidate.anchor.kind.value if candidate.anchor else "cover"
        item.setToolTip(
            f"shot {shot.index}\n"
            f"{candidate.rig.value} · {candidate.size.name.lower()} · {candidate.side.value}\n"
            f"{float(shot.duration):.2f}s · score {float(candidate.score):.3f}\n"
            f"because: {because}"
        )
        self._shots.append((shot.index, item))

    def _add_playhead(self, scene: QGraphicsScene) -> None:
        """Add the playhead marker.

        Args:
            scene: The scene.
        """
        self._playhead = scene.addRect(
            0.0,
            0.0,
            1.5,
            SHOT_TOP + SHOT_HEIGHT,
            QPen(Qt.PenStyle.NoPen),
            QBrush(COLOURS["text"]),
        )
        self._playhead.setZValue(20.0)

    def mark_time(self, moment: float) -> None:
        """Move the playhead.

        Args:
            moment: Time in seconds from session start.
        """
        if self._playhead is not None:
            self._playhead.setRect(moment * PIXELS_PER_SECOND, 0.0, 1.5, SHOT_TOP + SHOT_HEIGHT)

    def drawBackground(self, painter: QPainter, rect: QRectF | QRect) -> None:
        """Draw the time ruler behind the items.

        Args:
            painter: The painter.
            rect: The exposed region.
        """
        super().drawBackground(painter, rect)
        if self._duration <= 0.0:
            return

        painter.setPen(QPen(COLOURS["border"], 1.0))
        step = self._ruler_step()
        moment = 0.0
        while moment <= self._duration:
            x = moment * PIXELS_PER_SECOND
            painter.drawLine(int(x), 0, int(x), int(SHOT_TOP + SHOT_HEIGHT))
            painter.setPen(QPen(COLOURS["text_dim"], 1.0))
            painter.drawText(int(x) + 3, 12, f"{int(moment)}s")
            painter.setPen(QPen(COLOURS["border"], 1.0))
            moment += step

    def _ruler_step(self) -> float:
        """Choose a tick spacing that keeps labels legible.

        Returns:
            The spacing in seconds.
        """
        for step in (5.0, 10.0, 30.0, 60.0, 300.0):
            if self._duration / step <= _MAX_RULER_TICKS:
                return step
        return 600.0

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Scrub to the clicked moment and select any shot under it.

        Args:
            event: The mouse event.
        """
        scene_point = self.mapToScene(event.position().toPoint())
        moment = max(0.0, scene_point.x() / PIXELS_PER_SECOND)
        self.mark_time(moment)
        self.scrubbed.emit(moment)

        for index, item in self._shots:
            if item.rect().contains(scene_point):
                self.shot_selected.emit(index)
                break
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Scrub while the left button is held.

        Args:
            event: The mouse event.
        """
        if event.buttons() & Qt.MouseButton.LeftButton:
            scene_point = self.mapToScene(event.position().toPoint())
            moment = max(0.0, scene_point.x() / PIXELS_PER_SECOND)
            self.mark_time(moment)
            self.scrubbed.emit(moment)
        super().mouseMoveEvent(event)


def rig_colour(rig: str) -> QColor:
    """Look up the colour a rig is drawn in.

    Exposed so the shot inspector can match the timeline's colouring rather than
    inventing its own.

    Args:
        rig: The rig's name.

    Returns:
        Its colour.
    """
    return _RIG_COLOURS.get(rig, COLOURS["surface_high"])
