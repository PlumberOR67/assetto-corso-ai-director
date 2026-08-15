"""Drift detection: sustained slides, and the transitions between them.

**Thresholds are borrowed, not invented.** The established Assetto Corsa drift
app begins scoring at 10 degrees of slip above 20 km/h, treats 90-120 degrees as
a high-multiplier slide and anything past 120 as a spin. Those numbers reflect
years of community consensus about what counts as a drift in this specific game,
which is worth far more than a threshold picked to make the output look tidy.

**Hysteresis, not a bare threshold.** A single cutoff on a noisy signal produces
a burst of on-off-on detections whenever the angle hovers near it — one long
slide arrives as fifteen fragments, and a director would cut fifteen times. A
Schmitt trigger fixes this: entry needs a higher angle than exit, so once a drift
starts it takes a real reduction to end it. The gap between the thresholds is the
noise immunity.

**What is not detected, and why.** Countersteer requires steering angle, which
this project cannot yet decode from the replay record. It is tempting to infer it
from front-wheel geometry, but steering rotates the wheels about the kingpin axis
and moves their *centres* by millimetres — far below suspension travel. Any
"countersteer" derived that way would be noise wearing a label, so the detector
does not emit it.
"""

from __future__ import annotations

import numpy as np

from assetto_ai_director.application.features.kinematics import yaw_rate
from assetto_ai_director.domain.events import Event, EventKind, EventSeverity
from assetto_ai_director.domain.units import Seconds

__all__ = [
    "DRIFT_ENTRY_DEGREES",
    "DRIFT_EXIT_DEGREES",
    "MIN_DRIFT_SPEED_KMH",
    "SPIN_DEGREES",
    "detect_drifts",
    "detect_transitions",
]

DRIFT_ENTRY_DEGREES = 10.0
"""Sideslip at which a drift is considered to have begun.

The threshold the established Assetto Corsa drift-scoring app uses to start
scoring. Reused rather than re-derived: it encodes community consensus about what
reads as a drift in this game.
"""

DRIFT_EXIT_DEGREES = 6.0
"""Sideslip at which a drift is considered over.

Lower than the entry threshold on purpose. The gap is the hysteresis band that
stops one long slide fragmenting into a burst of detections as the angle wanders
across a single cutoff.
"""

SPIN_DEGREES = 120.0
"""Sideslip past which the car is spinning rather than drifting.

Also from the drift app, which zeroes the score multiplier here. A spin is still
worth filming — it is just not a drift, and conflating them would let a botched
corner outrank a committed slide.
"""

HIGH_ANGLE_DEGREES = 90.0
"""Sideslip above which a slide is exceptional. The drift app's 1.5x band."""

MIN_DRIFT_SPEED_KMH = 20.0
"""Speed below which slip angle is not treated as a drift.

Also from the drift app. Without it, a car rotating in a gravel trap at walking
pace scores as a spectacular slide.
"""

MIN_DRIFT_SECONDS = 0.35
"""Shortest span accepted as a drift.

Below this it is a twitch or a kerb strike, not a technique — and no camera move
can be built around it anyway.
"""

MIN_TRANSITION_YAW_RATE = 0.35
"""Yaw rate magnitude, radians per second, required either side of a transition.

A transition is a *deliberate* reversal of rotation. Requiring real rotational
speed on both sides rejects the sign flips that noise produces whenever yaw rate
passes through zero on a straight.
"""

TRANSITION_WINDOW_SECONDS = 0.6
"""How far either side of a sign change to look for the required yaw rate."""

_KMH_PER_MPS = 3.6


def detect_drifts(
    car_id: int,
    sideslip: np.ndarray,
    speed: np.ndarray,
    time: np.ndarray,
    sample_rate: float,
) -> list[Event]:
    """Find sustained slides.

    Args:
        car_id: The car being analysed.
        sideslip: Sideslip angle per frame, radians.
        speed: Ground speed per frame, metres per second.
        time: Time per frame, seconds from session start.
        sample_rate: Frames per second.

    Returns:
        One :attr:`~assetto_ai_director.domain.events.EventKind.DRIFT_START`
        event per slide, spanning the whole slide, with its peak angle marked.
        Spins are reported separately as
        :attr:`~assetto_ai_director.domain.events.EventKind.SPIN`.
    """
    degrees = np.degrees(np.abs(sideslip))
    fast_enough = speed * _KMH_PER_MPS >= MIN_DRIFT_SPEED_KMH

    active = _schmitt(degrees, DRIFT_ENTRY_DEGREES, DRIFT_EXIT_DEGREES) & fast_enough
    min_frames = max(round(MIN_DRIFT_SECONDS * sample_rate), 1)

    events: list[Event] = []
    for start, end in _spans(active):
        if end - start < min_frames:
            continue
        segment = degrees[start:end]
        peak_index = start + int(np.argmax(segment))
        peak_angle = float(segment.max())

        kind = EventKind.SPIN if peak_angle >= SPIN_DEGREES else EventKind.DRIFT_START
        events.append(
            Event(
                kind=kind,
                start=Seconds(float(time[start])),
                end=Seconds(float(time[min(end, len(time) - 1)])),
                subjects=(car_id,),
                confidence=_confidence(peak_angle),
                severity=EventSeverity.from_ratio(peak_angle / DRIFT_ENTRY_DEGREES),
                peak=Seconds(float(time[peak_index])),
                detail={
                    "peak_angle_deg": peak_angle,
                    "mean_angle_deg": float(segment.mean()),
                    "peak_speed_kmh": float(speed[start:end].max() * _KMH_PER_MPS),
                },
            )
        )
    return events


def detect_transitions(
    car_id: int,
    sideslip: np.ndarray,
    heading: np.ndarray,
    speed: np.ndarray,
    *,
    time: np.ndarray,
    sample_rate: float,
) -> list[Event]:
    """Find linked transitions — the flick from one slide directly into another.

    The signature move of touge driving, and the most cinematic thing a car can
    do: the car is sideways one way, rotates through neutral, and is immediately
    sideways the other way without straightening.

    Detected as a sign change in yaw rate that is flanked by genuine rotation in
    both directions, while the car is actually sliding. The flanking requirement
    is what separates a deliberate reversal from yaw rate wandering through zero
    on a straight.

    Args:
        car_id: The car being analysed.
        sideslip: Sideslip angle per frame, radians.
        heading: Heading per frame, radians.
        speed: Ground speed per frame, metres per second.
        time: Time per frame, seconds.
        sample_rate: Frames per second.

    Returns:
        One event per transition.
    """
    rate = yaw_rate(heading, sample_rate)
    degrees = np.degrees(np.abs(sideslip))
    fast_enough = speed * _KMH_PER_MPS >= MIN_DRIFT_SPEED_KMH
    # The car must be genuinely drifting, not merely above the hysteresis floor.
    # Using the exit threshold here admitted "transitions" at 8 degrees of slip,
    # which is a steering correction, not a linked drift.
    sliding = degrees >= DRIFT_ENTRY_DEGREES

    window = max(round(TRANSITION_WINDOW_SECONDS * sample_rate), 1)
    sign = np.sign(rate)
    crossings = np.flatnonzero((sign[:-1] * sign[1:]) < 0) + 1

    events: list[Event] = []
    for raw_index in crossings:
        # Narrowed once here: numpy integers propagate through min/max into a
        # union type that cannot be used as an index.
        index = int(raw_index)
        before = rate[max(0, index - window) : index]
        after = rate[index : index + window]
        if before.size == 0 or after.size == 0:
            continue
        # Real rotation on both sides, in *opposite* directions.
        #
        # Testing only that |yaw rate| exceeds the threshold somewhere in each
        # window does not do this: a window containing a single strong rotation
        # passes regardless of which way it turned, so a car that rotates hard
        # one way, settles, and rotates hard the same way again registers as a
        # transition. Taking the signed extreme on the side it actually rotated
        # is what makes the test directional.
        before_extreme = before.max() if abs(before.max()) >= abs(before.min()) else before.min()
        after_extreme = after.max() if abs(after.max()) >= abs(after.min()) else after.min()
        if abs(before_extreme) < MIN_TRANSITION_YAW_RATE:
            continue
        if abs(after_extreme) < MIN_TRANSITION_YAW_RATE:
            continue
        if np.sign(before_extreme) == np.sign(after_extreme):
            continue
        span = slice(max(0, index - window), min(len(degrees), index + window))
        if not (sliding[span].any() and fast_enough[index]):
            continue
        # A spin also reverses yaw rate, but it is the opposite of a linked
        # transition: one is control, the other is its loss. Reporting a spin as
        # a transition would let a botched corner masquerade as the signature
        # move of touge driving.
        if degrees[span].max() >= SPIN_DEGREES:
            continue

        magnitude = float(max(np.abs(before).max(), np.abs(after).max()))
        events.append(
            Event(
                kind=EventKind.DRIFT_TRANSITION,
                start=Seconds(float(time[max(0, index - window)])),
                end=Seconds(float(time[min(len(time) - 1, index + window)])),
                subjects=(car_id,),
                confidence=min(1.0, magnitude / (MIN_TRANSITION_YAW_RATE * 3.0)),
                severity=EventSeverity.from_ratio(magnitude / MIN_TRANSITION_YAW_RATE),
                peak=Seconds(float(time[index])),
                detail={
                    "peak_yaw_rate": magnitude,
                    "sideslip_deg": float(degrees[span].max()),
                },
            )
        )
    return _merge_close(events, sample_rate)


def _schmitt(signal: np.ndarray, high: float, low: float) -> np.ndarray:
    """Apply a Schmitt trigger: turn on above ``high``, off below ``low``.

    Between the thresholds the output holds its previous state, which is what
    gives the detector its noise immunity.

    Args:
        signal: The samples.
        high: Entry threshold.
        low: Exit threshold.

    Returns:
        A boolean mask of the active state.
    """
    above_high = signal >= high
    above_low = signal >= low

    # Vectorised hysteresis: within each run of "above low", the state is on
    # from the first "above high" onward.
    state = np.zeros(signal.shape[0], dtype=bool)
    active = False
    for index in range(signal.shape[0]):
        active = bool(above_low[index]) if active else bool(above_high[index])
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


def _confidence(peak_angle: float) -> float:
    """Score how confidently a slide is a drift.

    Angles barely past the entry threshold are marginal and are discounted, so
    downstream scoring can prefer unambiguous slides without a second threshold.

    Args:
        peak_angle: Peak sideslip in degrees.

    Returns:
        Confidence in ``[0, 1]``.
    """
    if peak_angle >= HIGH_ANGLE_DEGREES:
        return 1.0
    span = HIGH_ANGLE_DEGREES - DRIFT_ENTRY_DEGREES
    return float(min(1.0, max(0.3, (peak_angle - DRIFT_ENTRY_DEGREES) / span + 0.3)))


def _merge_close(events: list[Event], sample_rate: float) -> list[Event]:
    """Merge transitions that overlap in time.

    A single flick can produce several yaw-rate zero crossings as the rotation
    settles. Reporting each as its own transition would triple-count one move.

    Args:
        events: Detected transitions, in time order.
        sample_rate: Frames per second.

    Returns:
        The merged list.
    """
    del sample_rate
    if not events:
        return []
    merged = [events[0]]
    for event in events[1:]:
        last = merged[-1]
        if event.start <= last.end:
            keep = last if last.confidence >= event.confidence else event
            merged[-1] = Event(
                kind=keep.kind,
                start=last.start,
                end=max(last.end, event.end),
                subjects=keep.subjects,
                confidence=max(last.confidence, event.confidence),
                severity=keep.severity,
                peak=keep.peak,
                detail=keep.detail,
            )
        else:
            merged.append(event)
    return merged
