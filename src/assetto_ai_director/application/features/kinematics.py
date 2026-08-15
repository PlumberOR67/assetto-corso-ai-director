"""Derived signals: smoothing, derivatives, curvature and yaw rate.

**Phase lag is the enemy here.** Everything downstream is about *when* something
happened — when a drift began, when the car reached the apex, when to cut. A
filter that shifts the signal in time shifts every event with it, and the error
is invisible because the output still looks like clean driving data.

So every filter in this module is zero-phase. Smoothing uses a Savitzky-Golay
filter, which fits a local polynomial and evaluates it at the centre of its own
window, introducing no delay. Derivatives use the same filter's analytic
derivative rather than differencing a smoothed signal, which would amplify noise
twice.

**Curvature is computed by arc length, not by second derivatives.** The textbook
formula ``kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)`` is exact for a smooth
curve and useless on sampled data, because the second derivative of a noisy
signal is dominated by the noise. Production racing-line software instead
measures heading change over a fixed *distance* — TUM's ``trajectory_planning_helpers``
uses a 2 m window for curvature — which is stable, has a physical meaning, and
behaves the same at 30 km/h and 200 km/h. That is what is implemented here.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

__all__ = [
    "CURVATURE_WINDOW_METRES",
    "arc_length",
    "curvature_by_arc_length",
    "longitudinal_acceleration",
    "path_bearing",
    "smooth",
    "wrap",
    "yaw_rate",
]

CURVATURE_WINDOW_METRES = 2.0
"""Arc-length window for curvature, in metres.

Matches the default used by TUM's ``trajectory_planning_helpers`` for their
autonomous racing stack. Short enough to resolve a hairpin, long enough that
float32 position quantisation does not dominate.
"""

HEADING_WINDOW_METRES = 1.0
"""Arc-length window for heading change, in metres. TUM's default."""

_MIN_POLY_ORDER = 2
_MIN_SAMPLES = 5
_MIN_SAMPLES_FOR_BEARING = 2
"""Frames needed before a direction of travel exists at all."""

_EPSILON = 1e-9


def wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles into ``(-pi, pi]``.

    Every angular difference must pass through this. Heading crosses the branch
    cut several times a lap, and an unwrapped difference produces a spurious
    ``2*pi`` spike that reads downstream as a violent, instantaneous rotation —
    indistinguishable from a real flick.

    Args:
        angle: Angles in radians, any magnitude.

    Returns:
        The wrapped angles.
    """
    return np.asarray((angle + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float64)


def smooth(signal: np.ndarray, window: int, order: int = 2) -> np.ndarray:
    """Zero-phase smoothing via a Savitzky-Golay filter.

    Args:
        signal: The samples to smooth.
        window: Window length in samples. Forced odd and clamped to the signal
            length, since the filter requires both.
        order: Polynomial order.

    Returns:
        The smoothed signal, unchanged if it is too short to filter.
    """
    length = signal.shape[0]
    if length < _MIN_SAMPLES:
        return signal.astype(np.float64)
    window = min(max(window | 1, order + _MIN_POLY_ORDER | 1), length - (1 - length % 2))
    if window <= order:
        return signal.astype(np.float64)
    return np.asarray(savgol_filter(signal, window, order), dtype=np.float64)


def derivative(signal: np.ndarray, dt: float, window: int, order: int = 2) -> np.ndarray:
    """Zero-phase first derivative via a Savitzky-Golay filter.

    Uses the filter's analytic derivative rather than differencing a smoothed
    signal. Differencing after smoothing amplifies whatever noise the smoothing
    left behind; differentiating the fitted polynomial does not.

    Args:
        signal: The samples.
        dt: Sample spacing in seconds.
        window: Window length in samples.
        order: Polynomial order. Must exceed the derivative order.

    Returns:
        The derivative per second.
    """
    length = signal.shape[0]
    if length < _MIN_SAMPLES:
        return np.zeros(length, dtype=np.float64)
    window = min(max(window | 1, order + _MIN_POLY_ORDER | 1), length - (1 - length % 2))
    if window <= order:
        return np.zeros(length, dtype=np.float64)
    return np.asarray(savgol_filter(signal, window, order, deriv=1, delta=dt), dtype=np.float64)


def yaw_rate(heading: np.ndarray, sample_rate: float, window_seconds: float = 0.2) -> np.ndarray:
    """Compute yaw rate from a heading signal.

    Heading is unwrapped before differentiating. Differentiating a wrapped signal
    produces a large spike at every branch-cut crossing, which a transition
    detector would read as a violent flick — precisely the event it is looking
    for, so the failure would be silent and plausible.

    Args:
        heading: Heading per frame, radians.
        sample_rate: Frames per second.
        window_seconds: Smoothing window in seconds.

    Returns:
        Yaw rate in radians per second.
    """
    unwrapped = np.unwrap(heading)
    window = max(round(window_seconds * sample_rate), _MIN_SAMPLES)
    return derivative(unwrapped, 1.0 / sample_rate, window)


def arc_length(position: np.ndarray) -> np.ndarray:
    """Cumulative distance travelled along the ground plane.

    Args:
        position: World positions, shape ``(n, 3)``, y up.

    Returns:
        Cumulative distance in metres, shape ``(n,)``, starting at zero.
    """
    ground = position[:, [0, 2]]
    steps = np.linalg.norm(np.diff(ground, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(steps)))


def path_bearing(position: np.ndarray) -> np.ndarray:
    """Compute the direction the car is travelling, from its positions alone.

    Distinct from chassis heading, and the distinction is exactly what this
    project measures. During a drift the car points one way and moves another;
    their difference is sideslip. Road curvature is a property of the *path*, so
    it must be derived from where the car went, not from where it was pointing —
    otherwise a drift through a straight would register as a corner.

    Args:
        position: World positions, shape ``(n, 3)``.

    Returns:
        Unwrapped bearings in radians, shape ``(n,)``.
    """
    count = position.shape[0]
    if count < _MIN_SAMPLES_FOR_BEARING:
        return np.zeros(count, dtype=np.float64)
    ground = position[:, [0, 2]]
    delta = np.gradient(ground, axis=0)
    bearing = np.arctan2(delta[:, 0], delta[:, 1])
    return np.asarray(np.unwrap(bearing), dtype=np.float64)


def curvature_by_arc_length(
    position: np.ndarray, window_metres: float = CURVATURE_WINDOW_METRES
) -> np.ndarray:
    """Compute signed path curvature over a fixed arc-length window.

    Curvature is bearing change per unit distance, measured across a window of
    constant *length* rather than constant time, so the same corner yields the
    same curvature whether taken at 40 or 140 km/h.

    **The window is interpolated, not snapped to samples.** Locating the window
    edges with a sample search looks equivalent and is not: once the car travels
    further per frame than half the window, the nearest sample within the window
    is the frame itself, the window collapses to zero length, and curvature is
    silently forced to zero. At 33 Hz with a 2 m window that happens above
    exactly 120 km/h — so the original implementation reported no curvature at
    all on the fastest sixth of a real touge run, and injected a saturated
    discontinuity in every downstream signal at the moment the car crossed the
    threshold. Interpolating the unwrapped bearing at the exact arc lengths keeps
    the window the requested length at any speed.

    Args:
        position: World positions, shape ``(n, 3)``.
        window_metres: Arc length spanned by the measurement.

    Returns:
        Curvature in reciprocal metres, shape ``(n,)``. Zero only where the car
        genuinely has not moved.
    """
    count = position.shape[0]
    if count < _MIN_SAMPLES:
        return np.zeros(count, dtype=np.float64)

    distance = arc_length(position)
    bearing = path_bearing(position)
    half = window_metres * 0.5

    total = float(distance[-1])
    if total <= _EPSILON:
        return np.zeros(count, dtype=np.float64)

    # Clamp the sample points into the travelled range so the ends measure a
    # shorter but still non-degenerate window instead of extrapolating.
    behind = np.clip(distance - half, 0.0, total)
    ahead = np.clip(distance + half, 0.0, total)

    turned = np.interp(ahead, distance, bearing) - np.interp(behind, distance, bearing)
    span = ahead - behind

    result = np.zeros(count, dtype=np.float64)
    valid = span > _EPSILON
    result[valid] = turned[valid] / span[valid]
    return result


def longitudinal_acceleration(
    speed: np.ndarray, sample_rate: float, window_seconds: float = 0.25
) -> np.ndarray:
    """Compute acceleration along the direction of travel.

    Args:
        speed: Ground speed per frame, metres per second.
        sample_rate: Frames per second.
        window_seconds: Smoothing window in seconds.

    Returns:
        Acceleration in metres per second squared. Negative under braking.
    """
    window = max(round(window_seconds * sample_rate), _MIN_SAMPLES)
    return derivative(speed, 1.0 / sample_rate, window)


def lateral_acceleration(speed: np.ndarray, curvature: np.ndarray) -> np.ndarray:
    """Compute lateral acceleration from speed and path curvature.

    Uses ``a = v^2 * kappa``, the centripetal relation. Derived from geometry
    rather than differentiated from position twice, which would square the noise.

    Args:
        speed: Ground speed, metres per second.
        curvature: Path curvature, reciprocal metres.

    Returns:
        Lateral acceleration in metres per second squared, signed by turn
        direction.
    """
    return np.asarray(speed**2 * curvature, dtype=np.float64)


def jerk(acceleration: np.ndarray, sample_rate: float, window_seconds: float = 0.3) -> np.ndarray:
    """Compute the rate of change of acceleration.

    The basis of the Flow score: smooth, connected driving has low jerk, while
    hesitation and correction show up as spikes long before they show up in speed.

    Args:
        acceleration: Acceleration per frame.
        sample_rate: Frames per second.
        window_seconds: Smoothing window in seconds.

    Returns:
        Jerk in metres per second cubed.
    """
    window = max(round(window_seconds * sample_rate), _MIN_SAMPLES)
    return derivative(acceleration, 1.0 / sample_rate, window)
