"""Per-frame features shared by every scorer.

Computed once per session and handed to all five scorers, because most of them
need the same derived signals and recomputing curvature five times over a 68 000
frame session is pure waste.

**Session-relative normalisation is the important idea here.** Speed is scored
against this session's own distribution rather than an absolute number, so the
score means the same thing on a mountain pass and on a high-speed circuit. A
fixed "150 km/h is fast" threshold would rate an entire touge run as slow and an
entire Le Mans lap as maximal, and in both cases the director would lose its
ability to discriminate between moments — which is the only thing a score is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.application.features.kinematics import (
    curvature_by_arc_length,
    jerk,
    lateral_acceleration,
    longitudinal_acceleration,
    yaw_rate,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from assetto_ai_director.application.events.detect import CarSignals

__all__ = ["CarFeatures", "SessionFeatures", "compute_features"]

_LOW_PERCENTILE = 10.0
_HIGH_PERCENTILE = 95.0
"""Percentiles anchoring the speed score.

Not the minimum and maximum: those are single samples, and one spin or one
stationary moment would rescale the whole session. The 10th and 95th are stable
against outliers while still spanning the range that actually varies.
"""

_MIN_MOVING_SPEED = 2.0
_DEGENERATE_SCALE = 1e-6
"""Below this a normalisation anchor is indistinguishable from zero, and dividing
by it would turn numerical noise into full-scale scores."""


@dataclass(frozen=True, slots=True)
class CarFeatures:
    """Derived per-frame signals for one car.

    Attributes:
        car_id: The car's index.
        speed: Ground speed, metres per second.
        sideslip: Sideslip angle, radians.
        curvature: Path curvature, reciprocal metres.
        yaw_rate: Yaw rate, radians per second.
        longitudinal_g: Longitudinal acceleration, m/s^2.
        lateral_g: Lateral acceleration, m/s^2.
        jerk: Rate of change of acceleration, m/s^3.
        position: World positions, shape ``(n, 3)``.
        heading: Heading, radians.
    """

    car_id: int
    speed: np.ndarray
    sideslip: np.ndarray
    curvature: np.ndarray
    yaw_rate: np.ndarray
    longitudinal_g: np.ndarray
    lateral_g: np.ndarray
    jerk: np.ndarray
    position: np.ndarray
    heading: np.ndarray


@dataclass(frozen=True, slots=True)
class SessionFeatures:
    """Derived signals for every car, plus session-wide normalisation anchors.

    Attributes:
        cars: Features per car id.
        time: Time per frame, seconds from session start.
        sample_rate: Frames per second.
        speed_low: Speed at the low percentile, metres per second.
        speed_high: Speed at the high percentile, metres per second.
        jerk_scale: Jerk magnitude at the high percentile, used to normalise the
            Flow score against this session rather than an absolute figure.
    """

    cars: dict[int, CarFeatures]
    time: np.ndarray
    sample_rate: float
    speed_low: float
    speed_high: float
    jerk_scale: float

    @property
    def frame_count(self) -> int:
        """How many frames the session spans.

        Returns:
            The frame count.
        """
        return int(self.time.shape[0])

    def normalised_speed(self, speed: np.ndarray) -> np.ndarray:
        """Rescale speed into ``[0, 1]`` against this session's own range.

        Args:
            speed: Speed samples, metres per second.

        Returns:
            The normalised values, clipped to ``[0, 1]``.
        """
        span = self.speed_high - self.speed_low
        if span <= 0.0:
            return np.zeros_like(speed)
        return np.clip((speed - self.speed_low) / span, 0.0, 1.0)


def compute_features(
    signals: Mapping[int, CarSignals], time: np.ndarray, sample_rate: float
) -> SessionFeatures:
    """Derive every per-frame signal the scorers need.

    Args:
        signals: Decoded signals per car id.
        time: Time per frame, seconds from session start.
        sample_rate: Frames per second.

    Returns:
        The computed features and normalisation anchors.
    """
    cars: dict[int, CarFeatures] = {}
    for car_id, car in signals.items():
        curvature = curvature_by_arc_length(car.position)
        longitudinal = longitudinal_acceleration(car.speed, sample_rate)
        cars[car_id] = CarFeatures(
            car_id=car_id,
            speed=car.speed,
            sideslip=car.sideslip,
            curvature=curvature,
            yaw_rate=yaw_rate(car.heading, sample_rate),
            longitudinal_g=longitudinal,
            lateral_g=lateral_acceleration(car.speed, curvature),
            jerk=jerk(longitudinal, sample_rate),
            position=car.position,
            heading=car.heading,
        )

    speed_low, speed_high = _speed_anchors(cars)
    return SessionFeatures(
        cars=cars,
        time=time,
        sample_rate=sample_rate,
        speed_low=speed_low,
        speed_high=speed_high,
        jerk_scale=_jerk_scale(cars),
    )


def _speed_anchors(cars: dict[int, CarFeatures]) -> tuple[float, float]:
    """Find the speed percentiles that anchor the Speed score.

    Only moving frames contribute. A session that begins on the grid would
    otherwise have its low anchor pinned at zero, compressing every real speed
    difference into the top of the range.

    Args:
        cars: Features per car.

    Returns:
        ``(low, high)`` speeds in metres per second.
    """
    # Filter first, then test. Checking whether `cars` is non-empty is not the
    # same question: a session where every car is parked yields a list of empty
    # arrays, and np.concatenate refuses an empty sequence.
    samples = [
        moving for car in cars.values() if (moving := car.speed[car.speed > _MIN_MOVING_SPEED]).size
    ]
    if not samples:
        return 0.0, 1.0
    moving_speeds = np.concatenate(samples)
    return (
        float(np.percentile(moving_speeds, _LOW_PERCENTILE)),
        float(np.percentile(moving_speeds, _HIGH_PERCENTILE)),
    )


def _jerk_scale(cars: dict[int, CarFeatures]) -> float:
    """Find the jerk magnitude that counts as maximally unsettled this session.

    Args:
        cars: Features per car.

    Returns:
        The high-percentile absolute jerk, or ``1.0`` when there is nothing to
        measure, so downstream division stays safe.
    """
    # Moving frames only, matching _speed_anchors. Pooling parked cars pulls the
    # anchor toward zero — a session with fifteen empty grid slots is mostly
    # zero jerk — which inflates every real car's normalised jerk and collapses
    # the Flow score for the cars that actually drove.
    samples = [
        moving
        for car in cars.values()
        if (moving := np.abs(car.jerk)[car.speed > _MIN_MOVING_SPEED]).size
    ]
    if not samples:
        return 1.0
    scale = float(np.percentile(np.concatenate(samples), _HIGH_PERCENTILE))
    return scale if scale > _DEGENERATE_SCALE else 1.0
