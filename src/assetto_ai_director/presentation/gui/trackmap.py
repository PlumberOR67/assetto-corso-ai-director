"""The track map: the driven line, coloured by what the analysis found.

**One graphics item, not fifty thousand.** The obvious implementation gives each
telemetry sample its own ``QGraphicsItem``, and it is the standard way to make Qt
unusable — a session is tens of thousands of samples, and the scene graph's
per-item bookkeeping dominates everything. This is a single item that paints
arrays directly with ``QPainter``.

**Hover uses a spatial index, not Qt hit-testing.** Routing pointer events
through the scene graph to find the nearest sample is the same problem again.
A KD-tree over an ``(n, 2)`` array answers it in microseconds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QStyleOptionGraphicsItem,
    QWidget,
)
from scipy.spatial import cKDTree

from assetto_ai_director.presentation.gui.theme import COLOURS

if TYPE_CHECKING:
    from assetto_ai_director.domain.timeline import Timeline

__all__ = ["TrackMapView"]

_MARGIN = 30.0
_TRACE_WIDTH = 2.2
_HOVER_RADIUS_SCENE = 18.0
_MIN_TRACE_POINTS = 2
"""Fewest samples that make a line rather than a dot."""


class _TraceItem(QGraphicsItem):
    """Paints the whole driven line as one item.

    Attributes:
        points: Ground-plane positions, shape ``(n, 2)``.
        intensity: Per-sample drift intensity in ``[0, 1]``, used for colour.
    """

    def __init__(self, points: np.ndarray, intensity: np.ndarray) -> None:
        """Store the trace.

        Args:
            points: Ground-plane positions, shape ``(n, 2)``.
            intensity: Per-sample intensity in ``[0, 1]``.
        """
        super().__init__()
        self.points = points
        self.intensity = intensity
        self._bounds = self._compute_bounds()
        self._segments = self._build_segments()

    def boundingRect(self) -> QRectF:
        """Return the item's bounds in scene coordinates.

        Returns:
            The bounding rectangle.
        """
        return self._bounds

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint the trace.

        Args:
            painter: The painter.
            option: Style options, unused.
            widget: The widget being painted on, unused.
        """
        del option, widget
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for colour, path in self._segments:
            pen = QPen(colour, _TRACE_WIDTH)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path)

    def _compute_bounds(self) -> QRectF:
        """Compute the trace's extent.

        Returns:
            The bounding rectangle, padded so the pen does not clip.
        """
        if self.points.shape[0] == 0:
            return QRectF(0.0, 0.0, 1.0, 1.0)
        low = self.points.min(axis=0)
        high = self.points.max(axis=0)
        return QRectF(
            float(low[0]) - _MARGIN,
            float(low[1]) - _MARGIN,
            float(high[0] - low[0]) + 2 * _MARGIN,
            float(high[1] - low[1]) + 2 * _MARGIN,
        )

    def _build_segments(self) -> list[tuple[QColor, QPainterPath]]:
        """Group the trace into a few paths by intensity band.

        One path per colour band rather than per sample: a handful of
        ``drawPath`` calls instead of tens of thousands of ``drawLine`` calls,
        which is the difference between a map that appears instantly and one that
        stutters on every pan.

        Returns:
            ``(colour, path)`` pairs, quiet stretches first so hot ones draw over
            them.
        """
        if self.points.shape[0] < _MIN_TRACE_POINTS:
            return []

        bands = [
            (0.0, 0.15, COLOURS["quiet"]),
            (0.15, 0.45, COLOURS["corner"]),
            (0.45, 1.01, COLOURS["drift"]),
        ]
        segments: list[tuple[QColor, QPainterPath]] = []

        for low, high, colour in bands:
            mask = (self.intensity >= low) & (self.intensity < high)
            path = QPainterPath()
            drawing = False
            for index in range(self.points.shape[0] - 1):
                if not mask[index]:
                    drawing = False
                    continue
                point = QPointF(float(self.points[index][0]), float(self.points[index][1]))
                if drawing:
                    path.lineTo(point)
                else:
                    path.moveTo(point)
                    drawing = True
            if not path.isEmpty():
                segments.append((colour, path))
        return segments


class TrackMapView(QGraphicsView):
    """A top-down view of the driven line, with hover and shot highlighting."""

    hovered = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty map.

        Args:
            parent: The parent widget.
        """
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setBackgroundBrush(COLOURS["background"])
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        self._points: np.ndarray = np.zeros((0, 2))
        self._time: np.ndarray = np.zeros(0)
        self._tree: cKDTree | None = None
        self._marker: QGraphicsEllipseItem | None = None

    def show_session(self, position: np.ndarray, time: np.ndarray, sideslip: np.ndarray) -> None:
        """Display a car's driven line.

        Args:
            position: World positions, shape ``(n, 3)``.
            time: Time per frame, seconds.
            sideslip: Sideslip per frame, radians, used to colour the trace.
        """
        scene = self.scene()
        scene.clear()
        self._marker = None

        # Qt's y axis points down; the world's z is "forward". Negating keeps the
        # map oriented the way a person expects rather than mirrored.
        self._points = np.stack([position[:, 0], -position[:, 2]], axis=1)
        self._time = time
        self._tree = cKDTree(self._points) if self._points.shape[0] else None

        intensity = np.clip(np.degrees(np.abs(sideslip)) / 35.0, 0.0, 1.0)
        item = _TraceItem(self._points, intensity)
        scene.addItem(item)
        scene.setSceneRect(item.boundingRect())
        self.fitInView(item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def highlight_shots(self, timeline: Timeline) -> None:
        """Mark where each shot's camera sits.

        Args:
            timeline: The directed timeline.
        """
        scene = self.scene()
        for shot in timeline.shots:
            keyframe = shot.path.keyframes[len(shot.path.keyframes) // 2]
            position = keyframe.basis.position
            radius = 2.4
            dot = scene.addEllipse(
                position.x - radius,
                -position.z - radius,
                radius * 2,
                radius * 2,
                QPen(COLOURS["accent"], 1.0),
                COLOURS["accent_dim"],
            )
            dot.setToolTip(
                f"shot {shot.index} · {shot.candidate.rig.value} · {float(shot.start):.1f}s"
            )

    def mark_time(self, moment: float) -> None:
        """Move the playhead marker to a moment.

        Args:
            moment: Time in seconds from session start.
        """
        if self._time.size == 0:
            return
        index = int(np.clip(np.searchsorted(self._time, moment), 0, self._points.shape[0] - 1))
        point = self._points[index]
        radius = 4.0

        if self._marker is None:
            self._marker = self.scene().addEllipse(
                0.0, 0.0, radius * 2, radius * 2, QPen(COLOURS["text"], 1.5), COLOURS["accent"]
            )
            self._marker.setZValue(10.0)
        self._marker.setRect(
            float(point[0]) - radius, float(point[1]) - radius, radius * 2, radius * 2
        )

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Report the moment nearest the pointer.

        Args:
            event: The mouse event.
        """
        super().mouseMoveEvent(event)
        if self._tree is None:
            return
        scene_point = self.mapToScene(event.position().toPoint())
        distance, index = self._tree.query([scene_point.x(), scene_point.y()])
        if distance <= _HOVER_RADIUS_SCENE and 0 <= index < self._time.shape[0]:
            self.hovered.emit(float(self._time[int(index)]))

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom around the pointer.

        Args:
            event: The wheel event.
        """
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scale(factor, factor)
