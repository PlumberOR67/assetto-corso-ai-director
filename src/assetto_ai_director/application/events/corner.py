"""Corner detection: where the road turns, and where the apex falls.

Corners are segmented from **path curvature** rather than from steering input,
because steering angle is not among the channels this project can decode from a
replay — and because curvature is the more honest signal anyway. It describes
what the car actually did, not what the driver asked for.

**The apex is the point of maximum curvature**, chosen over the alternatives
after weighing them:

* *Minimum distance to the inside edge* is the textbook definition, and needs
  track boundary geometry this project does not have.
* *Minimum speed* is easy but wrong on any corner taken with a lift rather than a
  brake, and on downhill corners where the car accelerates through the apex —
  which describes most of a touge run.
* *Maximum lateral acceleration* conflates the corner's geometry with how hard
  the car was driven, so the same corner moves its apex depending on pace.

Maximum curvature is a property of the line the car took, is well defined on
every corner, and needs nothing but position and heading.
"""

from __future__ import annotations

import numpy as np

from assetto_ai_director.application.features.kinematics import (
    curvature_by_arc_length,
    longitudinal_acceleration,
)
from assetto_ai_director.domain.events import Event, EventKind, EventSeverity
from assetto_ai_director.domain.units import Seconds

__all__ = [
    "CORNER_ENTRY_CURVATURE",
    "CORNER_EXIT_CURVATURE",
    "detect_corners",
]

CORNER_ENTRY_CURVATURE = 0.020
"""Curvature at which a corner begins, in reciprocal metres — a 50 m radius.

Chosen relative to a real reference point rather than picked to taste: TUM's
racing stack uses 0.12 rad/m as a racing car's curvature limit, an 8.3 m radius.
A 50 m radius is comfortably a corner and comfortably not a straight.
"""

CORNER_EXIT_CURVATURE = 0.010
"""Curvature at which a corner ends — a 100 m radius.

Half the entry threshold, forming the hysteresis band. Without it, a long
constant-radius sweeper whose curvature sits near a single cutoff fragments into
a stutter of short corners.
"""

MIN_CORNER_SECONDS = 0.5
"""Shortest span accepted as a corner. Below this it is a kink or a correction."""

MIN_CORNER_SPEED_KMH = 15.0
"""Speed below which curvature is not treated as a corner.

A car manoeuvring in a pit box traces very tight curvature that has nothing to do
with the road.
"""

BRAKING_THRESHOLD = -3.0
"""Longitudinal acceleration, m/s^2, taken as meaningful braking.

Roughly 0.3 g. Below this magnitude the car is coasting or lifting, which is a
different driving act and reads differently on camera.
"""

_KMH_PER_MPS = 3.6


def detect_corners(
    car_id: int,
    position: np.ndarray,
    heading: np.ndarray,
    speed: np.ndarray,
    *,
    time: np.ndarray,
    sample_rate: float,
) -> list[Event]:
    """Segment a lap into corners and mark their apexes and braking points.

    Args:
        car_id: The car being analysed.
        position: World positions, shape ``(n, 3)``.
        heading: Heading per frame, radians.
        speed: Ground speed per frame, metres per second.
        time: Time per frame, seconds from session start.
        sample_rate: Frames per second.

    Returns:
        Corner entry, apex and exit events, plus a braking point where one is
        found, in time order.
    """
    del heading  # Curvature is a property of the path, not of where the car points.
    curvature = curvature_by_arc_length(position)
    magnitude = np.abs(curvature)
    acceleration = longitudinal_acceleration(speed, sample_rate)
    fast_enough = speed * _KMH_PER_MPS >= MIN_CORNER_SPEED_KMH

    active = _schmitt(magnitude, CORNER_ENTRY_CURVATURE, CORNER_EXIT_CURVATURE) & fast_enough
    min_frames = max(round(MIN_CORNER_SECONDS * sample_rate), 1)

    events: list[Event] = []
    for start, end in _spans(active):
        if end - start < min_frames:
            continue
        segment = magnitude[start:end]
        apex_index = start + int(np.argmax(segment))
        peak_curvature = float(segment.max())
        radius = 1.0 / peak_curvature if peak_curvature > 0.0 else float("inf")

        detail = {
            "peak_curvature": peak_curvature,
            "min_radius_m": radius,
            "entry_speed_kmh": float(speed[start] * _KMH_PER_MPS),
            "apex_speed_kmh": float(speed[apex_index] * _KMH_PER_MPS),
            "direction": float(np.sign(curvature[apex_index])),
        }
        severity = EventSeverity.from_ratio(peak_curvature / CORNER_ENTRY_CURVATURE)

        events.append(
            Event(
                kind=EventKind.CORNER_ENTRY,
                start=Seconds(float(time[start])),
                end=Seconds(float(time[apex_index])),
                subjects=(car_id,),
                severity=severity,
                peak=Seconds(float(time[start])),
                detail=detail,
            )
        )
        events.append(
            Event(
                kind=EventKind.APEX,
                start=Seconds(float(time[start])),
                end=Seconds(float(time[min(end, len(time) - 1)])),
                subjects=(car_id,),
                severity=severity,
                peak=Seconds(float(time[apex_index])),
                detail=detail,
            )
        )
        events.append(
            Event(
                kind=EventKind.CORNER_EXIT,
                start=Seconds(float(time[apex_index])),
                end=Seconds(float(time[min(end, len(time) - 1)])),
                subjects=(car_id,),
                severity=severity,
                peak=Seconds(float(time[min(end, len(time) - 1)])),
                detail=detail,
            )
        )

        braking = _braking_point(acceleration, start, apex_index, sample_rate)
        if braking is not None:
            events.append(
                Event(
                    kind=EventKind.BRAKING_POINT,
                    start=Seconds(float(time[braking])),
                    end=Seconds(float(time[apex_index])),
                    subjects=(car_id,),
                    severity=EventSeverity.from_ratio(
                        abs(float(acceleration[braking])) / abs(BRAKING_THRESHOLD)
                    ),
                    peak=Seconds(float(time[braking])),
                    detail={"peak_deceleration": float(acceleration[braking])},
                )
            )

    events.sort(key=lambda event: event.start)
    return events


def _braking_point(
    acceleration: np.ndarray, start: int, apex: int, sample_rate: float
) -> int | None:
    """Locate the onset of braking before a corner.

    Searches a window preceding the corner as well as the corner's own entry
    phase, because braking normally begins on the straight before the road starts
    to turn — looking only inside the detected corner would consistently report
    the braking point late.

    Args:
        acceleration: Longitudinal acceleration per frame.
        start: Index where the corner begins.
        apex: Index of the apex.
        sample_rate: Frames per second.

    Returns:
        The index of peak deceleration, or ``None`` if the car never braked
        meaningfully.
    """
    lookback = max(round(2.0 * sample_rate), 1)
    window_start = max(0, start - lookback)
    window = acceleration[window_start:apex]
    if window.size == 0:
        return None
    peak = int(np.argmin(window))
    if window[peak] > BRAKING_THRESHOLD:
        return None

    # The *onset*, not the peak. The event is named BRAKING_POINT and documented
    # as the moment deceleration begins, which is the editorially interesting
    # instant — a camera cut on peak deceleration arrives after the driver has
    # already committed. Walk back from the peak to the first frame of the
    # continuous braking phase that contains it.
    onset = peak
    while onset > 0 and window[onset - 1] <= BRAKING_THRESHOLD:
        onset -= 1
    return window_start + onset


def _schmitt(signal: np.ndarray, high: float, low: float) -> np.ndarray:
    """Apply a Schmitt trigger: on above ``high``, off below ``low``.

    Args:
        signal: The samples.
        high: Entry threshold.
        low: Exit threshold.

    Returns:
        A boolean mask of the active state.
    """
    state = np.zeros(signal.shape[0], dtype=bool)
    active = False
    for index in range(signal.shape[0]):
        active = signal[index] >= low if active else signal[index] >= high
        state[index] = active
    return state


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous runs of ``True``.

    Args:
        mask: A boolean array.

    Returns:
        ``(start, end)`` index pairs, end-exclusive.
    """
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]
