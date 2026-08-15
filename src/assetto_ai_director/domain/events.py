"""Driving events: the things a director might want to point a camera at.

An event is a *typed, timestamped span with subjects*. It is deliberately not a
score. "A drift happened here, involving car 3, from t=41.2 to t=44.8, and the
detector is fairly confident" is a factual claim about the session. Whether that
drift is worth a shot is a separate, style-dependent judgement made downstream.

Keeping those apart matters: the same session must be able to yield an Initial D
cut that lingers on the drift and a Best Motoring cut that ignores it in favour
of a clean connected sequence, without re-running detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from assetto_ai_director.domain.units import Seconds

__all__ = ["Event", "EventKind", "EventSeverity"]

_MODERATE_RATIO = 1.5
"""Multiples of the detection threshold at which an event stops being marginal."""

_STRONG_RATIO = 3.0
"""Multiples of the detection threshold at which an event becomes unmistakable."""


class EventKind(StrEnum):
    """What kind of thing happened.

    Attributes:
        CORNER_ENTRY: Turn-in — steering rate rises while the car is braking.
        APEX: The point of closest approach to the inside of a corner.
        CORNER_EXIT: Throttle reapplication after the apex.
        BRAKING_POINT: The onset of meaningful deceleration.
        TRAIL_BRAKING: Brake pressure carried past turn-in and released
            gradually — a technique worth showing because it is visible in the
            car's attitude.
        DRIFT_START: Sideslip crosses the on-threshold and is sustained.
        DRIFT_SUSTAIN: A continuing high-sideslip span.
        DRIFT_END: Sideslip falls back below the off-threshold.
        DRIFT_TRANSITION: A linked transition — yaw rate changes sign while
            sideslip stays large. The signature move of touge driving, and the
            single most cinematic thing a car can do.
        COUNTERSTEER: Steering opposes sideslip. Correlates with drift but is
            detected separately, since it also happens during a save.
        OVERTAKE_ATTEMPT: One car closes to within striking distance of another.
        OVERTAKE_COMPLETE: The order between two cars actually changes.
        DEFENCE: A car holds position against a closing attacker.
        CONTACT: Damage state changes — the cars touched.
        OFF_TRACK: Wheels leave the track surface.
        NEAR_LIMIT: Normalised slip reaches the grip limit and stays there.
        SPIN: Sideslip exceeds the point of no return.
        GEAR_CHANGE: A shift.
        REDLINE: Engine speed near maximum.
    """

    CORNER_ENTRY = "corner_entry"
    APEX = "apex"
    CORNER_EXIT = "corner_exit"
    BRAKING_POINT = "braking_point"
    TRAIL_BRAKING = "trail_braking"

    DRIFT_START = "drift_start"
    DRIFT_SUSTAIN = "drift_sustain"
    DRIFT_END = "drift_end"
    DRIFT_TRANSITION = "drift_transition"
    COUNTERSTEER = "countersteer"

    OVERTAKE_ATTEMPT = "overtake_attempt"
    OVERTAKE_COMPLETE = "overtake_complete"
    DEFENCE = "defence"

    CONTACT = "contact"
    OFF_TRACK = "off_track"
    NEAR_LIMIT = "near_limit"
    SPIN = "spin"

    GEAR_CHANGE = "gear_change"
    REDLINE = "redline"

    @property
    def is_instantaneous(self) -> bool:
        """Whether this kind marks a moment rather than a span.

        Instantaneous events still carry a span — a camera needs time around the
        moment — but the span is a window chosen by the detector, not a measured
        duration.

        Returns:
            ``True`` for point-like events.
        """
        return self in {
            EventKind.APEX,
            EventKind.BRAKING_POINT,
            EventKind.CONTACT,
            EventKind.GEAR_CHANGE,
            EventKind.DRIFT_START,
            EventKind.DRIFT_END,
            EventKind.OVERTAKE_COMPLETE,
        }

    @property
    def is_multi_car(self) -> bool:
        """Whether this kind inherently involves more than one car.

        Multi-car events are the ones that justify two-subject framing, which is
        a different and much harder camera problem than following one car.

        Returns:
            ``True`` for events defined by a relationship between cars.
        """
        return self in {
            EventKind.OVERTAKE_ATTEMPT,
            EventKind.OVERTAKE_COMPLETE,
            EventKind.DEFENCE,
            EventKind.CONTACT,
        }

    @property
    def is_dramatic(self) -> bool:
        """Whether this kind reads as drama rather than technique.

        Used as a coarse prior when a style weights spectacle over craft.

        Returns:
            ``True`` for events an audience reads as risk or incident.
        """
        return self in {
            EventKind.CONTACT,
            EventKind.SPIN,
            EventKind.OFF_TRACK,
            EventKind.NEAR_LIMIT,
            EventKind.OVERTAKE_COMPLETE,
            EventKind.DRIFT_TRANSITION,
        }


class EventSeverity(StrEnum):
    """How pronounced an instance of an event was.

    Attributes:
        SUBTLE: Detected, but only just past threshold.
        MODERATE: Clearly present.
        STRONG: Unmistakable.
    """

    SUBTLE = "subtle"
    MODERATE = "moderate"
    STRONG = "strong"

    @classmethod
    def from_ratio(cls, ratio: float) -> EventSeverity:
        """Classify an intensity expressed as a multiple of the threshold.

        Args:
            ratio: Measured magnitude divided by the detection threshold. A value
                of 1.0 sits exactly on the threshold.

        Returns:
            The severity band.
        """
        if ratio < _MODERATE_RATIO:
            return cls.SUBTLE
        if ratio < _STRONG_RATIO:
            return cls.MODERATE
        return cls.STRONG


@dataclass(frozen=True, slots=True)
class Event:
    """Something that happened in the session.

    Attributes:
        kind: What happened.
        start: When it began, in seconds from session start.
        end: When it ended. Equals ``start`` only for zero-length markers;
            detectors normally pad instantaneous events into a usable window.
        subjects: The cars involved, ordered by importance — the first is the
            one a camera should favour.
        confidence: How sure the detector is, in ``[0, 1]``. Below 1.0 whenever
            a threshold was marginal or an input channel was degraded, so
            downstream scoring can discount uncertain events rather than treating
            every detection as equally solid.
        severity: How pronounced the instance was.
        peak: When the event was most intense. Distinct from the midpoint: a
            drift's peak angle is usually well after its start, and that is where
            a cut wants to land.
        detail: Detector-specific numbers, e.g. ``{"max_sideslip_deg": 38.4}``.
            Free-form by design so a detector can record what it measured without
            forcing every measurement into the shared schema.
    """

    kind: EventKind
    start: Seconds
    end: Seconds
    subjects: tuple[int, ...]
    confidence: float = 1.0
    severity: EventSeverity = EventSeverity.MODERATE
    peak: Seconds | None = None
    detail: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the span and confidence.

        Raises:
            ValueError: If the span runs backwards, no subject is given, or the
                confidence is outside ``[0, 1]``.
        """
        if self.end < self.start:
            message = f"Event {self.kind} ends ({self.end}) before it starts ({self.start})."
            raise ValueError(message)
        if not self.subjects:
            message = f"Event {self.kind} has no subject car."
            raise ValueError(message)
        if not 0.0 <= self.confidence <= 1.0:
            message = f"Event confidence must be in [0, 1], got {self.confidence}."
            raise ValueError(message)

    @property
    def duration(self) -> Seconds:
        """How long the event lasted.

        Returns:
            The span length in seconds.
        """
        return Seconds(self.end - self.start)

    @property
    def midpoint(self) -> Seconds:
        """The centre of the event's span.

        Returns:
            The midpoint in seconds.
        """
        return Seconds((self.start + self.end) * 0.5)

    @property
    def focus(self) -> Seconds:
        """The single best instant to build a shot around.

        Returns:
            The peak if the detector found one, otherwise the midpoint.
        """
        return self.peak if self.peak is not None else self.midpoint

    @property
    def primary_subject(self) -> int:
        """The car a camera should favour.

        Returns:
            The first subject's car id.
        """
        return self.subjects[0]

    def overlaps(self, other: Event) -> bool:
        """Test whether two events share any time.

        Args:
            other: The event to compare against.

        Returns:
            ``True`` if the spans intersect.
        """
        return self.start < other.end and other.start < self.end

    def involves(self, car_id: int) -> bool:
        """Test whether a car takes part in this event.

        Args:
            car_id: The car to look for.

        Returns:
            ``True`` if the car is among the subjects.
        """
        return car_id in self.subjects
