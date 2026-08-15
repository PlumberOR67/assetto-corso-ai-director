"""Smoothing a camera's motion so it looks operated rather than computed.

**The formula, and why this one.** A camera that snaps to its ideal position
every frame reproduces every twitch of the car it is following, and the result
looks like a spreadsheet rather than a shot. The standard game-camera answer is
an exponential approach to the target::

    x += (target - x) * (1 - exp(-dt / tau))

``tau`` is the time constant: after ``tau`` seconds the camera has closed about
63% of the gap. The exponential form matters. The naive version everyone writes
first, ``x += (target - x) * k``, is **frame-rate dependent** — the same ``k``
smooths twice as hard at 66 Hz as at 33 Hz — so an edit generated from a
high-quality replay would move differently from the same edit generated from a
low-quality one, for no reason connected to the driving.

Angles are smoothed on the unit circle rather than as scalars. Interpolating
between 179 and -179 degrees numerically sweeps the whole way round; doing it as
a two-component vector takes the one-degree path a camera operator would.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["smooth_angles", "smooth_towards", "smooth_track"]

_EPSILON = 1e-9
_MIN_SAMPLES = 2
"""Fewest samples for which smoothing or clamping means anything."""


def _alpha(dt: float, tau: float) -> float:
    """Compute the per-step blend factor for a time constant.

    Args:
        dt: Time step in seconds.
        tau: Time constant in seconds. Zero or less disables smoothing.

    Returns:
        The fraction of the remaining gap to close this step, in ``[0, 1]``.
    """
    if tau <= _EPSILON or dt <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / tau)


def smooth_towards(current: float, target: float, dt: float, tau: float) -> float:
    """Move one scalar toward a target, frame-rate independently.

    Args:
        current: Present value.
        target: Value being approached.
        dt: Time step in seconds.
        tau: Time constant in seconds.

    Returns:
        The updated value.
    """
    return current + (target - current) * _alpha(dt, tau)


def smooth_track(targets: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Smooth a whole trajectory with an exponential follow.

    Applied forward then backward, which doubles the effective order and — more
    importantly — makes the filter **zero-phase**. A one-directional follow lags
    its target by roughly ``tau``, so a camera would arrive at each corner late by
    a fixed amount, and the lag would be invisible in the numbers while obvious
    on screen.

    Args:
        targets: Ideal positions, shape ``(n,)`` or ``(n, d)``.
        dt: Time step in seconds.
        tau: Time constant in seconds. Zero disables smoothing.

    Returns:
        The smoothed trajectory, same shape as the input.
    """
    if tau <= _EPSILON or targets.shape[0] < _MIN_SAMPLES:
        return np.asarray(targets, dtype=np.float64)

    # Halved because the pass is applied twice; two sequential first-order
    # follows of tau/2 approximate one of tau without the phase lag.
    alpha = _alpha(dt, tau * 0.5)
    forward = _sweep(np.asarray(targets, dtype=np.float64), alpha)
    backward = _sweep(forward[::-1], alpha)[::-1]
    return np.asarray(backward, dtype=np.float64)


def _sweep(values: np.ndarray, alpha: float) -> np.ndarray:
    """Run a single exponential follow along the leading axis.

    Args:
        values: The trajectory.
        alpha: Per-step blend factor.

    Returns:
        The followed trajectory.
    """
    result = np.empty_like(values)
    result[0] = values[0]
    for index in range(1, values.shape[0]):
        result[index] = result[index - 1] + (values[index] - result[index - 1]) * alpha
    return result


def smooth_angles(angles: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Smooth a sequence of angles without sweeping the long way round.

    Args:
        angles: Angles in radians, shape ``(n,)``.
        dt: Time step in seconds.
        tau: Time constant in seconds.

    Returns:
        The smoothed angles, wrapped to ``(-pi, pi]``.
    """
    if tau <= _EPSILON or angles.shape[0] < _MIN_SAMPLES:
        return np.asarray(angles, dtype=np.float64)
    # Smooth the direction as a vector: the short way round falls out of the
    # geometry instead of needing a special case at the branch cut.
    components = np.stack([np.sin(angles), np.cos(angles)], axis=1)
    smoothed = smooth_track(components, dt, tau)
    return np.asarray(np.arctan2(smoothed[:, 0], smoothed[:, 1]), dtype=np.float64)


def clamp_speed(path: np.ndarray, dt: float, max_speed: float) -> np.ndarray:
    """Limit how fast a camera may travel between samples.

    A drone that outruns what its notional aircraft could do is the detail that
    breaks the illusion a style depends on — the shot stops reading as a drone
    and starts reading as a cheat.

    Args:
        path: Positions, shape ``(n, 3)``.
        dt: Time step in seconds.
        max_speed: Speed limit in metres per second. Non-positive disables it.

    Returns:
        The clamped path.
    """
    if max_speed <= 0.0 or path.shape[0] < _MIN_SAMPLES:
        return np.asarray(path, dtype=np.float64)
    limit = max_speed * dt
    result = np.array(path, dtype=np.float64, copy=True)
    for index in range(1, result.shape[0]):
        step = result[index] - result[index - 1]
        distance = float(np.linalg.norm(step))
        if distance > limit and distance > _EPSILON:
            result[index] = result[index - 1] + step * (limit / distance)
    return result
