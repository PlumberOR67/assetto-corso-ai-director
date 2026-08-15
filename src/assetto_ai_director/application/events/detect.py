"""Run every detector over a session and collect the results.

The single entry point the director calls. Detectors stay independent of each
other — a drift detector that consulted the corner detector would be much harder
to reason about and impossible to test in isolation — so this module simply runs
them all and merges the output in time order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.application.events.corner import detect_corners
from assetto_ai_director.application.events.drift import detect_drifts, detect_transitions
from assetto_ai_director.application.events.overtake import detect_overtakes
from assetto_ai_director.application.ports.progress import Stage
from assetto_ai_director.domain.errors import OperationCancelledError
from assetto_ai_director.domain.events import Event, EventKind

if TYPE_CHECKING:
    from collections.abc import Mapping

    from assetto_ai_director.application.ports.cancellation import CancelToken
    from assetto_ai_director.application.ports.progress import ProgressReporter

__all__ = ["CarSignals", "DetectionResult", "detect_all"]


@dataclass(frozen=True, slots=True)
class CarSignals:
    """The decoded signals one car contributes to detection.

    Attributes:
        car_id: The car's index.
        position: World positions, shape ``(n, 3)``.
        heading: Heading per frame, radians.
        speed: Ground speed per frame, metres per second.
        sideslip: Sideslip angle per frame, radians.
    """

    car_id: int
    position: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    sideslip: np.ndarray


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Everything found in a session.

    Attributes:
        events: All events, in time order.
        notes: Honest caveats about what could not be detected, carried through
            to the timeline so a user is told what the analysis was blind to
            rather than left to assume it saw everything.
    """

    events: tuple[Event, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def of_kind(self, *kinds: EventKind) -> tuple[Event, ...]:
        """Filter events by kind.

        Args:
            *kinds: The kinds to keep.

        Returns:
            The matching events, in time order.
        """
        wanted = set(kinds)
        return tuple(event for event in self.events if event.kind in wanted)

    def involving(self, car_id: int) -> tuple[Event, ...]:
        """Filter events by participating car.

        Args:
            car_id: The car of interest.

        Returns:
            The matching events, in time order.
        """
        return tuple(event for event in self.events if event.involves(car_id))

    @property
    def counts(self) -> dict[EventKind, int]:
        """Tally events by kind.

        Returns:
            A mapping of kind to occurrence count.
        """
        tally: dict[EventKind, int] = {}
        for event in self.events:
            tally[event.kind] = tally.get(event.kind, 0) + 1
        return tally


UNDETECTABLE_NOTE = (
    "Not detected from this source: countersteer, throttle and brake application, "
    "gear changes, contact and off-track excursions. Those channels have not been "
    "located in the replay record, so no detector claims them."
)

MIN_ACTIVE_DISTANCE_METRES = 50.0
"""Distance a car must cover before it is analysed at all.

Replays routinely contain entries that never move: an online session with empty
grid slots records every slot, parked at the origin, for the whole session. One
real fixture has fifteen such entries against two moving cars.

They are excluded rather than analysed, for two reasons. Analysing them is pure
waste — pairwise overtake detection is quadratic in car count, so fifteen parked
cars cost far more than they contribute. And they are not silent: a stationary car
still jitters, and jitter run through an ``atan2`` produces heading noise that a
detector can mistake for events.
"""


def _active_cars(signals: Mapping[int, CarSignals]) -> tuple[dict[int, CarSignals], list[int]]:
    """Split cars into those that actually drove and those that never moved.

    Args:
        signals: Decoded signals per car id.

    Returns:
        ``(active, skipped_ids)``.
    """
    active: dict[int, CarSignals] = {}
    skipped: list[int] = []
    for car_id, car in signals.items():
        ground = car.position[:, [0, 2]]
        distance = float(np.linalg.norm(np.diff(ground, axis=0), axis=1).sum())
        if distance >= MIN_ACTIVE_DISTANCE_METRES:
            active[car_id] = car
        else:
            skipped.append(car_id)
    return active, skipped


def detect_all(
    signals: Mapping[int, CarSignals],
    time: np.ndarray,
    sample_rate: float,
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> DetectionResult:
    """Run every detector over a session.

    Args:
        signals: Decoded signals per car id.
        time: Time per frame, seconds from session start.
        sample_rate: Frames per second.
        progress: Optional reporter. This function owns the whole
            :attr:`Stage.DETECT` phase — start, one tick per car, finish —
            because the total is the number of cars that *moved*, which is not
            knowable until :func:`_active_cars` has run. A caller supplying the
            total from outside would have to duplicate that filter and would
            over-count a grid full of parked AI.
        cancel: Optional cancellation token, polled once per car.

    Returns:
        All detected events plus notes on what could not be detected.

    Raises:
        OperationCancelledError: If cancellation is requested part-way through.
    """
    events: list[Event] = []
    active, skipped = _active_cars(signals)

    if progress is not None:
        progress.start(str(Stage.DETECT), len(active))

    for car in active.values():
        if cancel is not None and cancel.cancelled():
            raise OperationCancelledError(str(Stage.DETECT))
        events.extend(detect_drifts(car.car_id, car.sideslip, car.speed, time, sample_rate))
        events.extend(
            detect_transitions(
                car.car_id,
                car.sideslip,
                car.heading,
                car.speed,
                time=time,
                sample_rate=sample_rate,
            )
        )
        events.extend(
            detect_corners(
                car.car_id,
                car.position,
                car.heading,
                car.speed,
                time=time,
                sample_rate=sample_rate,
            )
        )
        # After the work, not before: `advance` means "this much is done", and a
        # bar that reaches its total while the last car is still being analysed
        # is the same lie as the one this whole change exists to fix.
        if progress is not None:
            progress.advance()

    if len(active) > 1:
        events.extend(
            detect_overtakes(
                {car_id: car.position for car_id, car in active.items()},
                {car_id: car.heading for car_id, car in active.items()},
                {car_id: car.speed for car_id, car in active.items()},
                time,
                sample_rate,
            )
        )

    notes = [UNDETECTABLE_NOTE]
    if skipped:
        notes.append(
            f"Skipped {len(skipped)} car(s) that never moved "
            f"(ids {', '.join(str(car_id) for car_id in sorted(skipped))}) — "
            f"empty grid slots, not participants."
        )

    events.sort(key=lambda event: (event.start, event.kind.value))
    if progress is not None:
        progress.finish()
    return DetectionResult(tuple(events), tuple(notes))
