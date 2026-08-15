"""Estimating where along the track something happened.

Assetto Corsa's trackside cameras switch on **normalised track position** — a
number in ``[0, 1]`` that the game derives from the track's own spline. A
directed timeline knows only **time**. Converting between the two is the whole
problem, and this module holds the two estimates the project can make without
the track's AI line, side by side, because their disagreement is the measurement
that matters.

* :func:`time_fraction_spans` divides the timeline's *duration*. That is what
  ``cameras.ini`` export has always written. It is exact only when the car
  covers ground at a constant rate, which no car does: every braking zone
  stretches the time axis relative to the distance axis, and the error
  accumulates along the lap.
* :func:`distance_profile` integrates the car's own path length. Closer to the
  truth, and still not the truth — it measures the distance *this car* covered,
  so a wide line through a corner reads as further along the track than a tight
  one, and two cars diverge.

Neither is the track's normalised position. That comes from ``fast_lane.ai``, and
:func:`position_spans` is what uses it — each camera claiming the stretch of
track nearest to where it actually stands. The two estimates above are kept as
the fallback for a session whose track is not installed, and because their
disagreement is what the camera diagnostic measures.

All of it lives here rather than inside the exporter so that the code writing the
spans and the code checking them cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.domain.track import SplineSource, TrackSpline

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "DistanceProfile",
    "distance_profile",
    "driven_spline",
    "position_spans",
    "time_fraction_spans",
]

LAP_RETURN_METRES = 60.0
"""How close a car must come to its start to be considered to have completed a lap."""

MIN_LAP_METRES = 400.0
"""Distance a car must cover before returning counts as a lap rather than a wobble."""

_MIN_SAMPLES = 2
"""Frames needed before a path has any length to normalise over."""

_MIN_SPAN_SECONDS = 1e-6
_MIN_STEP = 1e-4
"""Smallest gap between consecutive span boundaries.

Two cameras sharing a boundary exactly would give the later one an empty range
the game can never select.
"""


@dataclass(frozen=True, slots=True)
class DistanceProfile:
    """A car's progress along the track, measured by its own path length.

    Attributes:
        fraction: Normalised progress per frame, in ``[0, 1]``. Wraps on a closed
            circuit and is monotonic on a point-to-point course.
        lap_length: The distance one traversal covered, in metres. This is what
            converts a disagreement in fraction into a disagreement in metres,
            which is the form a person can judge.
        closed: Whether the path returned to its start, making it a circuit.
            A point-to-point course does not revisit positions, so wrapping its
            fraction would send the final frame back to the start line.
    """

    fraction: np.ndarray
    lap_length: float
    closed: bool


def distance_profile(position: np.ndarray) -> DistanceProfile:
    """Estimate normalised track position from a car's own path.

    A lap is detected as a return to within :data:`LAP_RETURN_METRES` of the
    start after covering at least :data:`MIN_LAP_METRES`; distance is then
    normalised over that lap. With no lap detected the whole path is treated as
    one traversal, which is correct for a point-to-point course and wrong for a
    session that never closes.

    Args:
        position: World positions, shape ``(n, 3)``, y up.

    Returns:
        The profile. A path too short to have length yields all zeros and a lap
        length of zero rather than a division by it.
    """
    count = position.shape[0]
    if count < _MIN_SAMPLES:
        return DistanceProfile(np.zeros(count, dtype=np.float64), 0.0, closed=False)

    ground = position[:, [0, 2]]
    steps = np.linalg.norm(np.diff(ground, axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(steps)))

    to_start = np.linalg.norm(ground - ground[0], axis=1)
    returned = np.flatnonzero((to_start < LAP_RETURN_METRES) & (distance > MIN_LAP_METRES))
    closed = returned.size > 0
    lap_length = float(distance[returned[0]]) if closed else float(distance[-1])
    if lap_length <= 0.0:
        return DistanceProfile(np.zeros(count, dtype=np.float64), 0.0, closed=closed)

    fraction = distance / lap_length
    wrapped = np.mod(fraction, 1.0) if closed else np.clip(fraction, 0.0, 1.0)
    return DistanceProfile(np.asarray(wrapped, dtype=np.float64), lap_length, closed=closed)


def driven_spline(position: np.ndarray, *, spacing: float = 5.0) -> TrackSpline | None:
    """Build a centreline from one car's path through the session.

    The fallback for a machine with no Assetto Corsa installation, or a track
    that ships no AI line. **It is a line somebody drove**, not the middle of the
    road: it sits wherever the driver put the car, so it is off by up to a lane
    through every corner, and it says nothing at all about how wide the road is.

    Marked :attr:`~assetto_ai_director.domain.track.SplineSource.DRIVEN` so a
    check can refuse to answer a question this cannot support. Good enough for
    "roughly how far along the track"; useless for "is this camera on the
    tarmac", which is exactly why width comes back as unknown rather than as a
    number.

    Args:
        position: World positions, shape ``(n, 3)``, y up.
        spacing: Roughly how far apart to place the resulting vertices, in
            metres. Resampling matters: raw telemetry is unevenly spaced in
            distance, dense under braking and sparse on a straight, and a
            projection onto that lands unevenly for no reason to do with the
            track.

    Returns:
        The centreline, or ``None`` when the path is too short to describe one.
    """
    if position.shape[0] < _MIN_SAMPLES:
        return None

    steps = np.linalg.norm(np.diff(position[:, [0, 2]], axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(steps)))
    total = float(distance[-1])
    if total < spacing * _MIN_SAMPLES:
        return None

    wanted = np.arange(0.0, total, spacing)
    resampled = np.stack(
        [np.interp(wanted, distance, position[:, axis]) for axis in range(3)], axis=1
    )
    if resampled.shape[0] < _MIN_SAMPLES:
        return None

    gap = float(np.linalg.norm(resampled[-1, [0, 2]] - resampled[0, [0, 2]]))
    return TrackSpline(
        points=resampled,
        source=SplineSource.DRIVEN,
        closed=gap < LAP_RETURN_METRES and total > MIN_LAP_METRES,
    )


def position_spans(positions: Sequence[float]) -> list[tuple[float, float]]:
    """Divide ``[0, 1]`` between cameras that each know where they stand.

    The replacement for :func:`time_fraction_spans` once a centreline exists.
    Each camera claims the stretch of track nearer to it than to its neighbours,
    which is what a person means by "this camera covers that corner" and what
    Assetto Corsa's switching does with the result.

    The spans are contiguous by construction, because a gap leaves a stretch of
    track with no camera assigned and the view falls back unpredictably.

    Args:
        positions: Normalised track position per camera, in ascending order.

    Returns:
        ``(in_point, out_point)`` per camera, contiguous and covering ``[0, 1]``.
    """
    if not positions:
        return []

    boundaries = [0.0]
    boundaries.extend((earlier + later) * 0.5 for earlier, later in pairwise(positions))
    boundaries.append(1.0)

    # Two shots of the same corner would otherwise share a boundary exactly, and
    # the later one would get a range the game can never select.
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index], boundaries[index - 1] + _MIN_STEP)
    scale = boundaries[-1]
    if scale > 1.0:
        boundaries = [value / scale for value in boundaries]

    return [(boundaries[i], boundaries[i + 1]) for i in range(len(positions))]


def time_fraction_spans(
    start: float, end: float, shots: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Divide ``[0, 1]`` between shots in proportion to when they happen.

    **This is a time axis wearing a distance axis's clothes.** It is what the
    ``cameras.ini`` exporter has always written into ``IN_POINT``/``OUT_POINT``,
    and it is correct only for a car moving at a constant rate over exactly one
    traversal. It is kept, unchanged, because changing it and the diagnostic in
    the same breath would leave nothing to measure the change against.

    The spans are contiguous by construction — each shot's out point is the
    next one's in point — because a gap would leave a stretch of track with no
    camera assigned and the view falls back unpredictably.

    Args:
        start: When the timeline begins, in seconds.
        end: When it ends.
        shots: ``(start, end)`` per shot, in the order they should be laid out.

    Returns:
        ``(in_point, out_point)`` per shot, contiguous and covering ``[0, 1]``.
    """
    if not shots:
        return []

    span = max(end - start, _MIN_SPAN_SECONDS)
    boundaries = [0.0]
    for earlier, later in pairwise(shots):
        midpoint = (earlier[1] + later[0]) * 0.5
        boundaries.append(min(1.0, max(0.0, (midpoint - start) / span)))
    boundaries.append(1.0)

    # Enforce monotonicity: a non-increasing pair would produce a camera the
    # game can never select.
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index], boundaries[index - 1] + _MIN_STEP)
    scale = boundaries[-1]
    if scale > 1.0:
        boundaries = [value / scale for value in boundaries]

    return [(boundaries[i], boundaries[i + 1]) for i in range(len(shots))]
