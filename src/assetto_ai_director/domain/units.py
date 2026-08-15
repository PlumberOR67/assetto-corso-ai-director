"""Physical units as distinct types.

Every scalar quantity in the domain is a :class:`typing.NewType` over ``float``
rather than a bare ``float``. This costs nothing at runtime and lets mypy catch
the single most common class of bug in telemetry code: mixing up units.

Passing a value in km/h where m/s is expected, or degrees where radians are
expected, is a silent 3.6x or 57x error that produces plausible-looking output.
That is precisely the failure mode this project exists to avoid, so the type
system is made to refuse it.

Canonical internal representation, used everywhere unless a docstring says
otherwise:

* distances in **metres**
* speeds in **metres per second**
* angles in **radians**
* time in **seconds** relative to session start
* field of view in **vertical degrees** (matching CSP's ``GrabbedCamera.fov``)

Conversion happens only at layer boundaries — a reader converting a file's units
on the way in, or an exporter converting on the way out — never in the middle of
an analysis.
"""

from __future__ import annotations

import math
from typing import Final, NewType

__all__ = [
    "DEG_PER_RAD",
    "MPS_PER_KMH",
    "Degrees",
    "FovDegrees",
    "Grams",
    "Hertz",
    "KilometresPerHour",
    "Metres",
    "MetresPerSecond",
    "MetresPerSecondCubed",
    "MetresPerSecondSquared",
    "Newtons",
    "NormalisedPosition",
    "Radians",
    "RadiansPerSecond",
    "RadiansPerSecondSquared",
    "ReciprocalMetres",
    "Rpm",
    "Score",
    "Seconds",
    "degrees_to_radians",
    "kmh_to_mps",
    "mps_to_kmh",
    "radians_to_degrees",
    "wrap_pi",
]

# --- Linear -----------------------------------------------------------------
Metres = NewType("Metres", float)
MetresPerSecond = NewType("MetresPerSecond", float)
MetresPerSecondSquared = NewType("MetresPerSecondSquared", float)
MetresPerSecondCubed = NewType("MetresPerSecondCubed", float)
"""Jerk. The derivative of acceleration; the basis of the Flow score."""

KilometresPerHour = NewType("KilometresPerHour", float)
"""Only for presentation and for reproducing thresholds published in km/h."""

# --- Angular ----------------------------------------------------------------
Radians = NewType("Radians", float)
Degrees = NewType("Degrees", float)
RadiansPerSecond = NewType("RadiansPerSecond", float)
RadiansPerSecondSquared = NewType("RadiansPerSecondSquared", float)

FovDegrees = NewType("FovDegrees", float)
"""Vertical field of view in degrees.

Vertical, because that is what CSP's ``ac.GrabbedCamera.fov`` is documented to
be. Assetto Corsa's ``cameras.ini`` uses ``MIN_FOV``/``MAX_FOV`` whose axis is
**not** documented anywhere we could find; the exporter therefore treats the
conversion as an explicit, flagged assumption rather than silently equating them.
"""

# --- Time and rate ----------------------------------------------------------
Seconds = NewType("Seconds", float)
Hertz = NewType("Hertz", float)

# --- Other physical ---------------------------------------------------------
Rpm = NewType("Rpm", float)
Newtons = NewType("Newtons", float)
Grams = NewType("Grams", float)
ReciprocalMetres = NewType("ReciprocalMetres", float)
"""Curvature, kappa = 1/R. Zero on a straight; 0.12 is roughly a racing car's limit."""

# --- Dimensionless ----------------------------------------------------------
NormalisedPosition = NewType("NormalisedPosition", float)
"""Position along the track spline in ``[0, 1)``. Wraps at the start/finish line."""

Score = NewType("Score", float)
"""A normalised score in ``[0, 1]``. Comparable across sessions and tracks."""

# --- Conversion factors -----------------------------------------------------
MPS_PER_KMH: Final = 1.0 / 3.6
DEG_PER_RAD: Final = 180.0 / math.pi


def kmh_to_mps(value: KilometresPerHour) -> MetresPerSecond:
    """Convert kilometres per hour to metres per second.

    Args:
        value: Speed in km/h.

    Returns:
        The same speed in m/s.
    """
    return MetresPerSecond(value * MPS_PER_KMH)


def mps_to_kmh(value: MetresPerSecond) -> KilometresPerHour:
    """Convert metres per second to kilometres per hour.

    Args:
        value: Speed in m/s.

    Returns:
        The same speed in km/h.
    """
    return KilometresPerHour(value / MPS_PER_KMH)


def radians_to_degrees(value: Radians) -> Degrees:
    """Convert radians to degrees.

    Args:
        value: Angle in radians.

    Returns:
        The same angle in degrees.
    """
    return Degrees(value * DEG_PER_RAD)


def degrees_to_radians(value: Degrees) -> Radians:
    """Convert degrees to radians.

    Args:
        value: Angle in degrees.

    Returns:
        The same angle in radians.
    """
    return Radians(value / DEG_PER_RAD)


def wrap_pi(angle: Radians) -> Radians:
    """Wrap an angle into ``(-pi, pi]``.

    Every angular difference in this codebase must pass through this function.
    Sideslip is the difference between path tangent and heading, and both wrap
    independently; subtracting them without wrapping produces a spurious ``2*pi``
    spike every time either crosses the branch cut. Those spikes look exactly
    like a violent drift initiation to an event detector.

    Args:
        angle: An angle in radians, of any magnitude.

    Returns:
        The equivalent angle in ``(-pi, pi]``.

    Examples:
        >>> round(wrap_pi(Radians(3 * math.pi / 2)), 6)
        -1.570796
        >>> round(wrap_pi(Radians(math.pi)), 6)
        3.141593
    """
    wrapped = math.remainder(angle, math.tau)
    # math.remainder maps exactly -pi to -pi; the convention here is (-pi, pi].
    if wrapped == -math.pi:
        return Radians(math.pi)
    return Radians(wrapped)
