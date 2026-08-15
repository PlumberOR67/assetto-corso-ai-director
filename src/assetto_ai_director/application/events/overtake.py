"""Overtake detection from relative position between cars.

An overtake is a **change in order**, not merely two cars being close. The test
is therefore a sign change in the *longitudinal* gap — the separation projected
onto the direction of travel — while the cars are near enough for the pass to be
a real interaction rather than a lapping event two corners apart.

Projecting onto the direction of travel is what makes this robust. Raw distance
between two cars shrinks and grows constantly through corners without anyone
passing anyone; only the along-track component encodes who is ahead.

**Lapping is excluded by proximity, not by lap counting.** Lap number is not
among the channels decodable from the replay record, so a car being lapped cannot
be distinguished from a car being raced by counting laps. The distance gate does
the work instead: cars a straight apart do not produce an event even as their
along-track order changes.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from assetto_ai_director.domain.events import Event, EventKind, EventSeverity
from assetto_ai_director.domain.units import Seconds

__all__ = ["MAX_OVERTAKE_DISTANCE", "detect_overtakes"]

MAX_OVERTAKE_DISTANCE = 25.0
"""Metres. Cars further apart than this are not considered to be interacting.

Roughly five car lengths. Wide enough to include a pass completed on a straight,
narrow enough to exclude two cars on opposite sides of a hairpin whose along-track
order flips for reasons of geometry rather than racing.
"""

ATTEMPT_DISTANCE = 15.0
"""Metres. Closing inside this counts as an overtake attempt even if it fails."""

MIN_OVERTAKE_SPEED_KMH = 30.0
"""Speed below which an order change is manoeuvring, not racing."""

APPROACH_SECONDS = 1.5
"""How long before and after the pass the event should span, so a camera has
time to establish the move rather than arriving as it completes."""

MIN_GAP_SWING_METRES = 4.0
"""How far the along-track gap must swing either side of a crossing.

Roughly a car length. Below this the two cars are running level and the gap is
wobbling, not reversing — reporting each wobble as a completed pass would fill a
nose-to-tail battle with dozens of phantom overtakes.
"""

_KMH_PER_MPS = 3.6
_MIN_SEPARATION = 1e-6


def detect_overtakes(
    positions: dict[int, np.ndarray],
    headings: dict[int, np.ndarray],
    speeds: dict[int, np.ndarray],
    time: np.ndarray,
    sample_rate: float,
) -> list[Event]:
    """Find passes between every pair of cars.

    Args:
        positions: World positions per car id, each shape ``(n, 3)``.
        headings: Heading per car id, each shape ``(n,)``.
        speeds: Ground speed per car id, each shape ``(n,)``.
        time: Time per frame, seconds from session start.
        sample_rate: Frames per second.

    Returns:
        Completed overtakes, in time order. Each event's subjects are ordered
        ``(overtaker, overtaken)``.
    """
    events: list[Event] = []
    pad = max(round(APPROACH_SECONDS * sample_rate), 1)
    frame_count = time.shape[0]

    for first, second in combinations(sorted(positions), 2):
        gap, distance = _relative(positions[first], positions[second], headings[first], frame_count)
        # Every mask must span the same window `_relative` truncated to, or the
        # combination broadcasts two different lengths and raises. Cars can have
        # unequal decoded frame counts, so this cannot be assumed away.
        limit = gap.shape[0]
        close = distance <= MAX_OVERTAKE_DISTANCE
        moving = (speeds[first][:limit] * _KMH_PER_MPS >= MIN_OVERTAKE_SPEED_KMH) & (
            speeds[second][:limit] * _KMH_PER_MPS >= MIN_OVERTAKE_SPEED_KMH
        )
        eligible = close & moving

        # A bare sign change counts sub-metre wobble between two cars running
        # level as a completed change of order, and two cars nose to tail
        # produce dozens of those a lap. Hysteresis requires the gap to reach a
        # real magnitude on both sides of the crossing, so a pass has to be
        # committed before it is reported.
        crossings = _committed_crossings(gap, MIN_GAP_SWING_METRES)

        for index in crossings:
            if not eligible[index]:
                continue
            start = max(0, index - pad)
            end = min(limit - 1, index + pad)
            # Positive gap means `second` is ahead of `first` along the
            # direction of travel; a flip from positive to negative therefore
            # means `first` has passed `second`.
            overtaker, overtaken = (first, second) if gap[index] < 0 else (second, first)
            separation = float(distance[index])
            events.append(
                Event(
                    kind=EventKind.OVERTAKE_COMPLETE,
                    start=Seconds(float(time[start])),
                    end=Seconds(float(time[end])),
                    subjects=(overtaker, overtaken),
                    confidence=float(min(1.0, max(0.3, 1.0 - separation / MAX_OVERTAKE_DISTANCE))),
                    severity=EventSeverity.from_ratio(
                        max(1.0, ATTEMPT_DISTANCE / max(separation, 1.0))
                    ),
                    peak=Seconds(float(time[index])),
                    detail={
                        "separation_m": separation,
                        "speed_kmh": float(speeds[overtaker][index] * _KMH_PER_MPS),
                    },
                )
            )

    events.sort(key=lambda event: event.start)
    return _deduplicate(events)


def _committed_crossings(gap: np.ndarray, swing: float) -> list[int]:
    """Find sign changes in the gap that are flanked by a real magnitude.

    A crossing counts only if the gap reached ``swing`` metres of the opposite
    sign before it and after it — that is, one car was genuinely behind and then
    genuinely ahead, rather than the two trading millimetres.

    Args:
        gap: Along-track separation per frame, signed.
        swing: Magnitude the gap must reach on each side, metres.

    Returns:
        Frame indices of committed crossings, in order.
    """
    sign = np.sign(gap)
    candidates = np.flatnonzero((sign[:-1] * sign[1:]) < 0) + 1
    if candidates.size == 0:
        return []

    committed: list[int] = []
    for raw in candidates:
        index = int(raw)
        before = gap[:index]
        after = gap[index:]
        if before.size == 0 or after.size == 0:
            continue
        # Look back to the last time the gap was decisively on one side, and
        # forward to the first time it is decisively on the other.
        was_behind = np.flatnonzero(np.abs(before) >= swing)
        becomes_ahead = np.flatnonzero(np.abs(after) >= swing)
        if was_behind.size == 0 or becomes_ahead.size == 0:
            continue
        if np.sign(before[was_behind[-1]]) == np.sign(after[becomes_ahead[0]]):
            continue
        committed.append(index)
    return committed


def _relative(
    leader: np.ndarray, follower: np.ndarray, heading: np.ndarray, frame_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute along-track gap and absolute distance between two cars.

    Args:
        leader: First car's positions, shape ``(n, 3)``.
        follower: Second car's positions, shape ``(n, 3)``.
        heading: First car's heading, shape ``(n,)``.
        frame_count: Frames to consider.

    Returns:
        ``(gap, distance)``. ``gap`` is the separation projected onto the first
        car's direction of travel, positive when the second car is ahead.
    """
    limit = min(frame_count, leader.shape[0], follower.shape[0], heading.shape[0])
    delta = follower[:limit, [0, 2]] - leader[:limit, [0, 2]]
    forward = np.stack([np.sin(heading[:limit]), np.cos(heading[:limit])], axis=1)
    gap = np.einsum("ij,ij->i", delta, forward)
    distance = np.linalg.norm(delta, axis=1)
    return gap, np.maximum(distance, _MIN_SEPARATION)


def _deduplicate(events: list[Event]) -> list[Event]:
    """Collapse repeated crossings between the same pair of cars.

    Two cars running nose to tail cross the zero gap repeatedly as they jockey.
    Each crossing is technically an order change, but reporting twenty of them
    for one battle would swamp the timeline; the strongest is kept per overlap.

    Args:
        events: Detected overtakes, in time order.

    Returns:
        The deduplicated list.
    """
    kept: list[Event] = []
    for event in events:
        pair = frozenset(event.subjects)
        replaced = False
        for index, existing in enumerate(kept):
            if frozenset(existing.subjects) == pair and existing.overlaps(event):
                if event.confidence > existing.confidence:
                    kept[index] = event
                replaced = True
                break
        if not replaced:
            kept.append(event)
    return kept
