"""The track itself: a centreline, how wide it is, and where a point sits on it.

Everything the camera work could not answer came down to not having this. Where
a camera stands relative to the road, whether it is above the surface, and — the
one that started the work — what normalised position Assetto Corsa will actually
switch a trackside camera at, all need a description of the track rather than a
description of one car's drive through it.

**Provenance travels with the geometry.** A centreline read from the track's own
AI line is a different kind of fact from one averaged out of the cars' paths, and
a check that cannot tell them apart will report a lane's worth of error as a
camera fault. :class:`TrackSpline` therefore carries how it was obtained, and
whether it knows the road's width at all, so a consumer can refuse to answer
rather than answer badly.

**Projection is nearest-segment, not nearest-vertex.** A spline is sampled every
few metres; snapping to the nearest sample quantises lateral offset into steps of
that size, which is the same magnitude as the thing being measured. Projecting
onto the segments recovers a continuous answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from assetto_ai_director.domain.units import Metres, NormalisedPosition

__all__ = ["SplineSource", "TrackPoint", "TrackSpline"]

_MIN_POINTS = 2
"""Fewest samples that describe a line rather than a place."""

_EPSILON = 1e-9


class SplineSource(StrEnum):
    """Where a centreline came from, because it changes what it can be used for.

    Attributes:
        AI_LINE: Read from the track's own ``fast_lane.ai``. The racing line the
            game itself uses, with the road's width recorded alongside it.
        DRIVEN: Averaged from the cars' paths in a session. Always available and
            never exact — it is a line somebody drove, which sits wherever they
            chose to place the car, and it says nothing about how wide the road
            is. Good enough to answer "roughly where along the track", useless
            for "is this camera on the tarmac".
    """

    AI_LINE = "ai_line"
    DRIVEN = "driven"

    @property
    def is_measured(self) -> bool:
        """Whether this centreline came from the track rather than from a drive.

        Returns:
            ``True`` for the AI line.
        """
        return self is SplineSource.AI_LINE


@dataclass(frozen=True, slots=True)
class TrackPoint:
    """Where a world position sits relative to the track.

    Attributes:
        position: Normalised progress along the spline, in ``[0, 1)``. The same
            quantity Assetto Corsa switches trackside cameras on.
        lateral: Signed distance from the centreline in metres, positive to one
            side and negative to the other. Which side is which follows the same
            left-handed convention as the rest of the codebase, and no check
            relies on the sign alone.
        elevation: Height of the centreline at that point, in metres. A camera
            below this is below the road.
        half_width: Distance from the centreline to the edge of the road on the
            side the point is on, in metres, or ``None`` when the spline does not
            know. ``None`` is not zero and must not be treated as it.
        distance: Distance along the spline in metres, for turning a difference
            in :attr:`position` into something a person can judge.
    """

    position: NormalisedPosition
    lateral: Metres
    elevation: Metres
    half_width: Metres | None
    distance: Metres

    @property
    def is_on_road(self) -> bool | None:
        """Whether the point lies between the road's edges.

        Returns:
            ``True`` or ``False`` when the width is known, otherwise ``None`` —
            which callers must report as unmeasured rather than as a pass.
        """
        if self.half_width is None:
            return None
        return abs(float(self.lateral)) <= float(self.half_width)


@dataclass(frozen=True, slots=True)
class TrackSpline:
    """A track's centreline, with whatever else is known about it.

    Attributes:
        points: Centreline vertices in world space, shape ``(n, 3)``, y up.
        source: How it was obtained.
        left_width: Distance to the left edge at each vertex, shape ``(n,)``, or
            ``None`` when unknown.
        right_width: Distance to the right edge at each vertex, or ``None``.
        closed: Whether the line returns to its start, making it a circuit.
            A point-to-point course does not, and normalised position must not
            wrap on one.
    """

    points: np.ndarray
    source: SplineSource
    left_width: np.ndarray | None = None
    right_width: np.ndarray | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        """Validate the geometry.

        Raises:
            ValueError: If the line has too few points, is not ``(n, 3)``, or
                carries widths that do not match it point for point.
        """
        if self.points.ndim != _MIN_POINTS or self.points.shape[1] != 3:  # noqa: PLR2004
            message = f"A track spline needs an (n, 3) array, got {self.points.shape}."
            raise ValueError(message)
        if self.points.shape[0] < _MIN_POINTS:
            message = f"A track spline needs at least {_MIN_POINTS} points."
            raise ValueError(message)
        for name, width in (("left_width", self.left_width), ("right_width", self.right_width)):
            if width is not None and width.shape[0] != self.points.shape[0]:
                message = (
                    f"{name} has {width.shape[0]} values for "
                    f"{self.points.shape[0]} points; they must correspond."
                )
                raise ValueError(message)

    @property
    def knows_width(self) -> bool:
        """Whether the road's width is available.

        Returns:
            ``True`` when both edges are recorded.
        """
        return self.left_width is not None and self.right_width is not None

    @property
    def length(self) -> Metres:
        """Total length of the centreline.

        Returns:
            The distance from start to finish along the line, in metres.
        """
        return Metres(float(self._cumulative[-1]))

    @property
    def _cumulative(self) -> np.ndarray:
        """Distance along the line at each vertex.

        Returns:
            Cumulative distance, shape ``(n,)``, starting at zero.
        """
        steps = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        return np.concatenate(([0.0], np.cumsum(steps)))

    def project(self, position: np.ndarray) -> TrackPoint:
        """Find where a world point sits relative to the track.

        Projects onto the nearest *segment* rather than snapping to the nearest
        vertex: a spline sampled every few metres would otherwise quantise the
        lateral offset into steps of the same size as the offsets being
        measured.

        Args:
            position: A world point, shape ``(3,)``, y up.

        Returns:
            Where it sits.
        """
        starts = self.points[:-1]
        ends = self.points[1:]
        segments = ends - starts

        # Ground plane only. A camera nine metres up a bank is still beside the
        # same piece of road, and including elevation would push its projection
        # along the track rather than out from it.
        flat_segments = segments[:, [0, 2]]
        flat_offsets = position[[0, 2]] - starts[:, [0, 2]]
        lengths = np.einsum("ij,ij->i", flat_segments, flat_segments)
        travel = np.divide(
            np.einsum("ij,ij->i", flat_offsets, flat_segments),
            np.maximum(lengths, _EPSILON),
        )
        travel = np.clip(travel, 0.0, 1.0)

        closest = starts[:, [0, 2]] + flat_segments * travel[:, None]
        gaps = np.linalg.norm(position[[0, 2]] - closest, axis=1)
        index = int(np.argmin(gaps))
        along = float(travel[index])

        cumulative = self._cumulative
        distance = float(cumulative[index] + along * (cumulative[index + 1] - cumulative[index]))
        total = float(cumulative[-1])

        tangent = flat_segments[index]
        norm = float(np.linalg.norm(tangent))
        if norm < _EPSILON:
            signed = 0.0
        else:
            # Cross product in the ground plane, matching the codebase's
            # left-handed arrangement of x and z.
            offset = position[[0, 2]] - closest[index]
            signed = float(tangent[0] * offset[1] - tangent[1] * offset[0]) / norm

        elevation = float(
            self.points[index][1] + along * (self.points[index + 1][1] - self.points[index][1])
        )
        return TrackPoint(
            position=NormalisedPosition(distance / total if total > 0.0 else 0.0),
            lateral=Metres(signed),
            elevation=Metres(elevation),
            half_width=self._half_width(index, along, signed),
            distance=Metres(distance),
        )

    def _half_width(self, index: int, along: float, signed: float) -> Metres | None:
        """Interpolate the road's half width on the side a point sits.

        Args:
            index: Which segment the point projected onto.
            along: How far along that segment, in ``[0, 1]``.
            signed: The point's signed lateral offset, which picks the side.

        Returns:
            The half width in metres, or ``None`` when the spline does not know.
        """
        if self.left_width is None or self.right_width is None:
            return None
        edge = self.left_width if signed >= 0.0 else self.right_width
        return Metres(float(edge[index] + along * (edge[index + 1] - edge[index])))
