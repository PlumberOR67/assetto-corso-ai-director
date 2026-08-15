"""The five moment scores.

Each answers a different question about an instant, and they are deliberately
orthogonal — a moment can be fast and dull, or slow and beautiful, and a director
that collapsed those into one number would have nothing left to choose with.

**Two scores are limited by what this source provides, and say so.**

*Risk* would ideally combine grip usage, track-edge proximity and damage. None of
those channels have been located in the replay record, so Risk here is built from
what is real: proximity to other cars weighted by closing rate, and lateral jerk.
That is a narrower Risk than the name suggests, and the limitation is reported in
the timeline notes rather than hidden behind a confident-looking number.

*Cinematic* is the most interpretive of the five, and the only one that is a claim
about composition rather than about driving. It rewards moments that *photograph*
well — cars at a framable separation, headings spread enough to read as distinct
shapes, elevation change to put something in the background — which is not the
same as moments that are dramatic. A drift on a flat car park and the same drift
against a falling mountain road score identically on Drift and differently here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.application.events.drift import (
    DRIFT_ENTRY_DEGREES,
    MIN_DRIFT_SPEED_KMH,
    SPIN_DEGREES,
)
from assetto_ai_director.application.ports.progress import NullProgress, Stage
from assetto_ai_director.domain.errors import OperationCancelledError
from assetto_ai_director.domain.scoring import ScoreVector
from assetto_ai_director.domain.units import Score

if TYPE_CHECKING:
    from assetto_ai_director.application.ports.cancellation import CancelToken
    from assetto_ai_director.application.ports.progress import ProgressReporter
    from assetto_ai_director.application.scoring.features import CarFeatures, SessionFeatures

__all__ = [
    "RISK_LIMITATION_NOTE",
    "score_car",
    "score_session",
]

RISK_LIMITATION_NOTE = (
    "Risk is computed from car proximity, closing rate and lateral jerk only. "
    "Grip usage, track-edge proximity and damage are not available from this "
    "source, so a lone car running wide scores no risk however close it came."
)

# --- Drift ------------------------------------------------------------------

DRIFT_REFERENCE_DEGREES = 45.0
"""Sideslip treated as a maximal drift.

Sits between the published automated-drifting targets — Stanford's steady-state
work targets 40 degrees, Toyota Research's Formula Drift policy reaches 63 — so a
committed slide scores near the top without a spin being required to get there.
"""

SUSTAIN_WINDOW_SECONDS = 1.2
"""How long a slide must hold to score fully.

Drift is a sustained state, not an instant. A car that flicks past 30 degrees for
three frames on a kerb has not drifted, and scoring it as though it had would put
a camera on a bump.
"""

# --- Risk -------------------------------------------------------------------

PROXIMITY_REFERENCE_METRES = 12.0
"""Separation at which two cars are as close as this score cares about."""

PROXIMITY_MAX_METRES = 40.0
"""Beyond this, cars are not interacting and contribute no risk."""

CLOSING_REFERENCE_MPS = 8.0
"""Closing rate treated as maximal. Distance alone is not danger — two cars
holding station a metre apart is a formation, not a moment."""

LATERAL_JERK_REFERENCE = 30.0
"""Lateral jerk, m/s^3, treated as maximally violent."""

# --- Cinematic --------------------------------------------------------------

FRAMABLE_MIN_METRES = 3.0
FRAMABLE_MAX_METRES = 30.0
"""Separation band within which two cars can share a frame legibly.

Closer and they overlap into one shape; further and either the framing goes so
wide that both become specks, or one leaves frame.
"""

_IDEAL_HEADING_SPREAD = np.pi * 0.5
"""Heading difference that reads best on screen: a quarter turn.

Enough divergence that the two cars are separate shapes rather than one, without
being so opposed that they are leaving each other's frame.
"""

ELEVATION_WINDOW_SECONDS = 3.0
ELEVATION_REFERENCE_METRES = 8.0
"""Height change over the window treated as maximally scenic.

Elevation is what separates a mountain pass from a car park on screen, and it is
directly measurable from the position channel.
"""

_KMH_PER_MPS = 3.6
_MIN_SAMPLES_FOR_GRADIENT = 2

_BASE_SHARE = 0.4
_URGENCY_SHARE = 0.6
"""How proximity risk splits between "they are close" and "they are converging".

Weighted toward convergence: two cars a metre apart holding station for a whole
straight is a formation, not a moment.
"""

_VIOLENCE_WEIGHT = 0.7
"""How much a violent direction change counts against real car-to-car proximity.

Below one, because a lone car being thrown about is less dangerous than the same
car being thrown about next to someone else.
"""


def score_session(
    features: SessionFeatures,
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> dict[int, list[ScoreVector]]:
    """Score every frame of every car.

    Args:
        features: Precomputed session features.
        progress: Optional reporter. Owns the whole :attr:`Stage.SCORE` phase.
        cancel: Optional cancellation token, polled once per car.

    Returns:
        Per-car lists of score vectors, one per frame.

    Raises:
        OperationCancelledError: If cancellation is requested part-way through.
    """
    reporter: ProgressReporter = progress if progress is not None else NullProgress()
    reporter.start(str(Stage.SCORE), len(features.cars))
    scored: dict[int, list[ScoreVector]] = {}
    for car_id in features.cars:
        if cancel is not None and cancel.cancelled():
            raise OperationCancelledError(str(Stage.SCORE))
        scored[car_id] = score_car(features, car_id)
        reporter.advance()
    reporter.finish()
    return scored


def score_car(features: SessionFeatures, car_id: int) -> list[ScoreVector]:
    """Score every frame for one car.

    Args:
        features: Precomputed session features.
        car_id: The car to score.

    Returns:
        One score vector per frame.
    """
    car = features.cars[car_id]
    drift = drift_score(car, features.sample_rate)
    speed = speed_score(car, features)
    risk = risk_score(car, features)
    flow = flow_score(car, features)
    cinematic = cinematic_score(car, features)

    # Bounded by the car's own arrays, not by the session time base. A car with
    # fewer decoded frames than the session — which happens whenever cars are
    # decoded independently — would otherwise be indexed past its end.
    count = int(drift.shape[0])
    return [
        ScoreVector(
            drift=Score(float(drift[i])),
            speed=Score(float(speed[i])),
            risk=Score(float(risk[i])),
            flow=Score(float(flow[i])),
            cinematic=Score(float(cinematic[i])),
        )
        for i in range(count)
    ]


def drift_score(car: CarFeatures, sample_rate: float) -> np.ndarray:
    """Score how committed a slide is, per frame.

    Combines angle, sustain and speed. Each is necessary: angle alone rewards a
    kerb flick, sustain alone rewards a long gentle slip, and without the speed
    gate a car rotating in a gravel trap at walking pace outscores a committed
    slide through a fast sweeper.

    A spin scores *down* rather than up. It is dramatic, but it is not a drift,
    and letting a botched corner top the drift score would send the director's
    best camera to the wrong moment.

    Args:
        car: The car's features.
        sample_rate: Frames per second.

    Returns:
        Scores in ``[0, 1]``, shape ``(n,)``.
    """
    degrees = np.degrees(np.abs(car.sideslip))
    angle = np.clip(
        (degrees - DRIFT_ENTRY_DEGREES) / (DRIFT_REFERENCE_DEGREES - DRIFT_ENTRY_DEGREES),
        0.0,
        1.0,
    )
    # Past the spin threshold the car is no longer drifting; taper to zero rather
    # than letting the angle term keep climbing.
    spinning = degrees >= SPIN_DEGREES
    angle = np.where(spinning, 0.0, angle)

    window = max(round(SUSTAIN_WINDOW_SECONDS * sample_rate), 1)
    sustain = _moving_mean((degrees >= DRIFT_ENTRY_DEGREES).astype(np.float64), window)

    fast_enough = (car.speed * _KMH_PER_MPS >= MIN_DRIFT_SPEED_KMH).astype(np.float64)
    return np.clip(angle * (0.4 + 0.6 * sustain) * fast_enough, 0.0, 1.0)


def speed_score(car: CarFeatures, features: SessionFeatures) -> np.ndarray:
    """Score speed relative to this session's own distribution.

    Args:
        car: The car's features.
        features: Session features, carrying the normalisation anchors.

    Returns:
        Scores in ``[0, 1]``, shape ``(n,)``.
    """
    return features.normalised_speed(car.speed)


def risk_score(car: CarFeatures, features: SessionFeatures) -> np.ndarray:
    """Score how close to disaster a moment is.

    See :data:`RISK_LIMITATION_NOTE` — this is a narrower Risk than the name
    implies, because the channels that would carry the rest are not available.

    Proximity is gated on closing rate. Two cars holding station a metre apart
    for a whole straight is a formation, not a risk; the same gap closing at
    8 m/s is a moment.

    Args:
        car: The car's features.
        features: Session features, for the other cars' positions.

    Returns:
        Scores in ``[0, 1]``, shape ``(n,)``.
    """
    proximity = _proximity_risk(car, features)
    lateral_jerk = np.abs(_gradient(car.lateral_g)) * features.sample_rate
    violence = np.clip(lateral_jerk / LATERAL_JERK_REFERENCE, 0.0, 1.0)
    # Gated on movement, like every other score. Without this, two cars parked on
    # adjacent grid slots register maximum proximity and a standing start scores
    # as the most dangerous moment of the session.
    moving = (car.speed * _KMH_PER_MPS >= MIN_DRIFT_SPEED_KMH).astype(np.float64)
    combined = np.clip(np.maximum(proximity, violence * _VIOLENCE_WEIGHT) * moving, 0.0, 1.0)
    return np.asarray(combined, dtype=np.float64)


def flow_score(car: CarFeatures, features: SessionFeatures) -> np.ndarray:
    """Score how clean and connected the driving is.

    The inverse of jerk: smooth, committed driving has low jerk, while hesitation
    and correction spike it long before they show up in speed. This is the score
    that values a beautifully linked sequence of corners, and it is what
    television-style coverage is actually looking for.

    Args:
        car: The car's features.
        features: Session features, for the jerk normalisation anchor.

    Returns:
        Scores in ``[0, 1]``, shape ``(n,)``.
    """
    window = max(round(2.0 * features.sample_rate), 1)
    smoothed = _moving_mean(np.abs(car.jerk), window)
    unsettled = np.clip(smoothed / features.jerk_scale, 0.0, 1.0)
    moving = (car.speed * _KMH_PER_MPS >= MIN_DRIFT_SPEED_KMH).astype(np.float64)
    return np.clip((1.0 - unsettled) * moving, 0.0, 1.0)


def cinematic_score(car: CarFeatures, features: SessionFeatures) -> np.ndarray:
    """Score how well a moment photographs.

    Distinct from how dramatic it is. Three measurable contributions: a second car
    at a separation where both fit legibly in one frame, headings spread enough
    that the two read as different shapes rather than one blur, and elevation
    change in the recent path — which is what makes a mountain pass look like a
    mountain pass rather than a road.

    Args:
        car: The car's features.
        features: Session features, for the other cars.

    Returns:
        Scores in ``[0, 1]``, shape ``(n,)``.
    """
    companion, heading_spread = _companion_framing(car, features)
    window = max(round(ELEVATION_WINDOW_SECONDS * features.sample_rate), 1)
    elevation = _elevation_change(car.position[:, 1], window)
    scenery = np.clip(elevation / ELEVATION_REFERENCE_METRES, 0.0, 1.0)

    moving = (car.speed * _KMH_PER_MPS >= MIN_DRIFT_SPEED_KMH).astype(np.float64)
    composed = 0.5 * companion + 0.2 * heading_spread + 0.3 * scenery
    return np.clip(composed * moving, 0.0, 1.0)


def _proximity_risk(car: CarFeatures, features: SessionFeatures) -> np.ndarray:
    """Compute risk from the nearest other car and how fast it is closing.

    Args:
        car: The car being scored.
        features: Session features, for the other cars.

    Returns:
        Scores in ``[0, 1]``, shape ``(n,)``.
    """
    others = [other for other in features.cars.values() if other.car_id != car.car_id]
    if not others:
        return np.zeros(car.speed.shape[0], dtype=np.float64)

    nearest = np.full(car.speed.shape[0], np.inf)
    for other in others:
        limit = min(car.position.shape[0], other.position.shape[0])
        distance = np.linalg.norm(
            car.position[:limit, [0, 2]] - other.position[:limit, [0, 2]], axis=1
        )
        nearest[:limit] = np.minimum(nearest[:limit], distance)

    nearest = np.where(np.isfinite(nearest), nearest, PROXIMITY_MAX_METRES)
    closeness = np.clip(
        (PROXIMITY_MAX_METRES - nearest) / (PROXIMITY_MAX_METRES - PROXIMITY_REFERENCE_METRES),
        0.0,
        1.0,
    )
    # Closing rate is the derivative of separation; negative means converging.
    closing = -_gradient(nearest) * features.sample_rate
    urgency = np.clip(closing / CLOSING_REFERENCE_MPS, 0.0, 1.0)
    weighted = np.clip(closeness * (_BASE_SHARE + _URGENCY_SHARE * urgency), 0.0, 1.0)
    return np.asarray(weighted, dtype=np.float64)


def _companion_framing(
    car: CarFeatures, features: SessionFeatures
) -> tuple[np.ndarray, np.ndarray]:
    """Score whether a second car sits at a framable distance, and how distinct it looks.

    Args:
        car: The car being scored.
        features: Session features, for the other cars.

    Returns:
        ``(framability, heading_spread)``, each in ``[0, 1]``.
    """
    count = car.speed.shape[0]
    others = [other for other in features.cars.values() if other.car_id != car.car_id]
    if not others:
        return np.zeros(count), np.zeros(count)

    best = np.zeros(count)
    spread = np.zeros(count)
    for other in others:
        limit = min(car.position.shape[0], other.position.shape[0])
        distance = np.linalg.norm(
            car.position[:limit, [0, 2]] - other.position[:limit, [0, 2]], axis=1
        )
        # A triangular band: peaks in the middle of the framable range.
        centre = (FRAMABLE_MIN_METRES + FRAMABLE_MAX_METRES) * 0.5
        half = (FRAMABLE_MAX_METRES - FRAMABLE_MIN_METRES) * 0.5
        framable = np.clip(1.0 - np.abs(distance - centre) / half, 0.0, 1.0)

        difference = np.abs(
            (car.heading[:limit] - other.heading[:limit] + np.pi) % (2 * np.pi) - np.pi
        )
        # Peaks at a quarter turn and falls away again. A monotone ramp would
        # award its maximum to head-on and tail-on pairs — exactly the case the
        # score should reject, since cars pointing opposite ways are separating
        # and will not hold a frame together for the length of a shot.
        distinct = np.clip(
            1.0 - np.abs(difference - _IDEAL_HEADING_SPREAD) / _IDEAL_HEADING_SPREAD, 0.0, 1.0
        )

        improved = framable > best[:limit]
        best[:limit] = np.where(improved, framable, best[:limit])
        spread[:limit] = np.where(improved, distinct, spread[:limit])
    return best, spread


def _elevation_change(elevation: np.ndarray, window: int) -> np.ndarray:
    """Measure absolute height change over a centred window.

    Centred, matching :func:`_moving_mean` and the rest of the pipeline. A
    trailing window would date the scenery term half a window in the past — 1.5 s
    at the default — so the score would peak after the descent it describes, and
    a camera cut on it would arrive late.

    Args:
        elevation: Height per frame, metres.
        window: Window length in frames.

    Returns:
        Absolute change over the window, shape ``(n,)``.
    """
    count = elevation.shape[0]
    if count <= window or window < 1:
        return np.zeros(count)
    half = window // 2
    indices = np.arange(count)
    behind = np.clip(indices - half, 0, count - 1)
    ahead = np.clip(indices + (window - half), 0, count - 1)
    return np.asarray(np.abs(elevation[ahead] - elevation[behind]), dtype=np.float64)


def _gradient(signal: np.ndarray) -> np.ndarray:
    """Differentiate a signal, tolerating arrays too short for ``np.gradient``.

    ``np.gradient`` raises below two samples. A one-frame session is degenerate
    but reachable — a caller asking for a very short window, or a replay trimmed
    to its final instant — and crashing there would be a worse answer than
    "nothing changed".

    Args:
        signal: The samples.

    Returns:
        The per-sample derivative, or zeros when the signal is too short.
    """
    if signal.shape[0] < _MIN_SAMPLES_FOR_GRADIENT:
        return np.zeros_like(signal, dtype=np.float64)
    return np.asarray(np.gradient(signal), dtype=np.float64)


def _moving_mean(signal: np.ndarray, window: int) -> np.ndarray:
    """Compute a centred moving average.

    Centred rather than trailing so the result is zero-phase, matching the rest
    of the pipeline: a trailing average would delay every score relative to the
    event that caused it.

    Args:
        signal: The samples.
        window: Window length in frames.

    Returns:
        The smoothed signal, same shape as the input.
    """
    if window <= 1 or signal.shape[0] < window:
        return signal.astype(np.float64)
    kernel = np.ones(window) / window
    padded = np.pad(signal, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")
