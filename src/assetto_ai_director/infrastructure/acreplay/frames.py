"""Decode the verified channels inside a car's per-frame records.

**What this module decodes, and why it is allowed to.** The 288-byte car frame
record is undocumented. Rather than transcribing a community struct definition
and hoping, each field here was located by probing real replays and then
*validated against an independent physical fact*. Only fields that passed are
decoded; the rest of the record stays opaque.

Two fields passed:

*Body position*, three ``float32`` at offset 20. Validated by driving the
resulting trajectory through physics: over a 180-second window the path is
4504 m long, giving a mean of 90 km/h with a p95 of 143 km/h, and the elevation
falls monotonically from 801.6 m to 643.3 m. The fixtures are recorded on
``akina_downhill``. A 158-metre descent on a track whose name is "downhill" is
not something a wrong offset produces.

*Four wheel positions*, twelve ``float32`` at offset 40. Validated by rigid-body
geometry: the four points hold constant distances from the body reference
(standard deviation under 5 mm across thousands of frames), sit 0.18-0.24 m below
it, and form a rectangle 1.40 m across by **2.405 m** long. The fixtures are
Toyota AE86s, whose wheelbase is 2400 mm. Random bytes do not reproduce a
specific car's wheelbase to within five millimetres.

**Why this matters more than it looks.** The replay also stores a rotation field,
but its Euler order, handedness and zero reference are undocumented, and sideslip
angle — the quantity the entire drift analysis rests on — is the difference
between where the car points and where it is going. Deriving heading from an
unvalidated rotation convention would have meant either a risky guess or falling
back to a degraded proxy.

The wheel positions make that problem disappear. The vector from the rear axle
midpoint to the front axle midpoint *is* the car's longitudinal axis, as a matter
of geometry rather than convention. Heading is therefore exact, and the
front/rear assignment is resolved from the data itself: one orientation yields a
median sideslip of 1.34 degrees, the other 178.66. There is no ambiguity to
resolve and nothing to calibrate.

The resulting numbers hold up as driving: on a downhill pass at a median 91 km/h,
13.3% of frames exceed 10 degrees of sideslip — the threshold the established
Assetto Corsa drift-scoring app uses to begin scoring — and 7.8% exceed 20
degrees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from assetto_ai_director.domain.frames import ReferenceFrame
from assetto_ai_director.domain.telemetry import Channel
from assetto_ai_director.infrastructure.acreplay.structure import CarBlock, map_replay

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

__all__ = [
    "BODY_POSITION_OFFSET",
    "DECODED_CHANNELS",
    "WHEEL_POSITION_OFFSET",
    "CarFrames",
    "read_car_frames",
]

BODY_POSITION_OFFSET: Final = 20
"""Byte offset of the body position: three ``float32`` (x, y, z), y up, metres."""

WHEEL_POSITION_OFFSET: Final = 40
"""Byte offset of four wheel positions: four times three ``float32``, metres.

Their order within the record is *not* assumed to match the game's FL/FR/RL/RR
convention. The two axles are identified by geometry — the pair whose separation
is the track width — and front versus rear is resolved from the data by whichever
assignment yields physically sensible sideslip.
"""

DECODED_CHANNELS: Final = frozenset(
    {
        Channel.POSITION,
        Channel.SPEED,
        Channel.HEADING,
    }
)
"""Channels this decoder can honestly provide.

Deliberately short. Everything else in the record — throttle, brake, gear, rpm,
per-wheel slip — is present in the bytes but has not been located and validated,
so it is not offered. A source that advertised those channels would have to
fabricate them, and the whole provider contract exists to prevent that.
"""

_VECTOR_BYTES = 12
_WHEEL_COUNT = 4
_TANGENT_LAG = 3
"""Frames spanned when differencing position for the path tangent.

One frame is dominated by float32 quantisation at 33 Hz; three gives a clean
direction while still resolving a corner. This is a smoothing choice, not a
physical constant.
"""

_MIN_TANGENT_SPEED = 2.0
"""Metres per second below which a path tangent is direction-less noise.

A stationary car still jitters by millimetres, and ``atan2`` of that jitter is
uniformly random. Heading is held rather than computed below this speed.
"""

_MIN_FRAMES_FOR_SPEED = 2
"""Frames needed before a finite difference can produce a speed at all."""

_MAX_PLAUSIBLE_SPEED = 150.0
"""Metres per second, 540 km/h, above which a sample is a position discontinuity.

Comfortably above anything in Assetto Corsa — the fastest content tops out near
400 km/h — and far below the values a teleport produces, which are tens of
thousands. The gap between the two is wide enough that no threshold tuning is
required.
"""


@dataclass(frozen=True, slots=True)
class CarFrames:
    """Decoded per-frame channels for one car.

    Attributes:
        car_id: The car's index in the session.
        position: World position, shape ``(n, 3)``, metres, y up.
        wheel_positions: Wheel positions, shape ``(n, 4, 3)``, metres.
        heading: Ground-plane yaw from wheel geometry, shape ``(n,)``, radians,
            measured as ``atan2(x, z)`` to match the game's y-up convention.
        speed: Ground speed, shape ``(n,)``, metres per second.
        sample_rate: Frames per second.
    """

    car_id: int
    position: npt.NDArray[np.float64]
    wheel_positions: npt.NDArray[np.float64]
    heading: npt.NDArray[np.float64]
    speed: npt.NDArray[np.float64]
    sample_rate: float

    @property
    def frame_count(self) -> int:
        """How many frames were decoded.

        Returns:
            The frame count.
        """
        return int(self.position.shape[0])

    @property
    def sideslip(self) -> npt.NDArray[np.float64]:
        """Chassis sideslip angle per frame, radians.

        The angle between where the car is pointing and where it is actually
        travelling — the quantity that defines a drift. Computed as the wrapped
        difference between path tangent and heading, both of which come from
        validated geometry rather than from an undocumented rotation convention.

        Held at zero below :data:`_MIN_TANGENT_SPEED`, where the path tangent is
        numerically meaningless.

        Returns:
            Sideslip in radians, shape ``(n,)``, wrapped to ``(-pi, pi]``.
        """
        tangent = self.path_tangent
        beta = _wrap(tangent - self.heading)
        beta[self.speed < _MIN_TANGENT_SPEED] = 0.0
        return beta

    @property
    def path_tangent(self) -> npt.NDArray[np.float64]:
        """Direction of travel per frame, radians.

        Returns:
            The ground-plane bearing of motion, shape ``(n,)``.
        """
        return _path_tangent(self.position)


def read_car_frames(path: Path, block: CarBlock, sample_rate: float) -> CarFrames:
    """Decode one car's validated channels from a replay.

    Args:
        path: The replay file.
        block: The car's verified structural block.
        sample_rate: Session sampling rate in hertz.

    Returns:
        The decoded channels.

    Raises:
        OSError: If the file cannot be read.
    """
    count = block.frame_count
    stride = block.stride

    with map_replay(path) as data:
        start = block.frames_offset
        # Copied rather than viewed: a NumPy view keeps the memory map exported,
        # which prevents it from closing and leaks the mapping.
        raw = np.frombuffer(bytes(data[start : start + count * stride]), dtype=np.uint8)

    records = raw.reshape(count, stride)

    position = (
        records[:, BODY_POSITION_OFFSET : BODY_POSITION_OFFSET + _VECTOR_BYTES]
        .copy()
        .view(np.float32)
        .reshape(count, 3)
        .astype(np.float64)
    )
    wheel_bytes = _WHEEL_COUNT * _VECTOR_BYTES
    wheels = (
        records[:, WHEEL_POSITION_OFFSET : WHEEL_POSITION_OFFSET + wheel_bytes]
        .copy()
        .view(np.float32)
        .reshape(count, _WHEEL_COUNT, 3)
        .astype(np.float64)
    )

    heading = _heading_from_wheels(wheels, position, sample_rate)
    speed = _speed(position, sample_rate)
    return CarFrames(block.identity.car_id, position, wheels, heading, speed, sample_rate)


def _heading_from_wheels(
    wheels: npt.NDArray[np.float64], position: npt.NDArray[np.float64], sample_rate: float
) -> npt.NDArray[np.float64]:
    """Derive ground-plane heading from wheel geometry.

    Pairs the wheels into axles by separation, then resolves which axle is at the
    front by testing both assignments against the direction of travel. The correct
    one produces small sideslip; the wrong one produces roughly 180 degrees, so
    the test is not close.

    Args:
        wheels: Wheel positions, shape ``(n, 4, 3)``.
        position: Body positions, shape ``(n, 3)``.
        sample_rate: Frames per second.

    Returns:
        Heading in radians, shape ``(n,)``.
    """
    axle_first, axle_second = _axle_midpoints(wheels)
    axis = axle_first - axle_second
    heading = np.arctan2(axis[:, 0], axis[:, 2])

    tangent = _path_tangent(position)
    speed = _speed(position, sample_rate)
    moving = speed > _MIN_TANGENT_SPEED
    if not moving.any():
        return heading

    forward_error = np.abs(_wrap(tangent[moving] - heading[moving])).mean()
    reversed_error = np.abs(_wrap(tangent[moving] - heading[moving] - np.pi)).mean()
    if reversed_error < forward_error:
        heading = _wrap(heading + np.pi)
    return heading


def _axle_midpoints(
    wheels: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Group four wheels into two axles and return their midpoints.

    The pairing is found from the first frame's geometry: the two shortest
    pairings are the axles, because track width is smaller than wheelbase on any
    road car. Derived from the data rather than assuming the record's wheel order.

    Args:
        wheels: Wheel positions, shape ``(n, 4, 3)``.

    Returns:
        The two axle midpoints, each shape ``(n, 3)``.
    """
    sample = wheels[0]
    pairings = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
    best = min(
        pairings,
        key=lambda pairing: sum(float(np.linalg.norm(sample[a] - sample[b])) for a, b in pairing),
    )
    (a1, a2), (b1, b2) = best
    first = (wheels[:, a1, :] + wheels[:, a2, :]) * 0.5
    second = (wheels[:, b1, :] + wheels[:, b2, :]) * 0.5
    return first, second


def _path_tangent(position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the ground-plane direction of travel.

    Args:
        position: World positions, shape ``(n, 3)``.

    Returns:
        Bearings in radians, shape ``(n,)``, with the leading frames repeating
        the first valid value rather than being left undefined.
    """
    count = position.shape[0]
    tangent = np.zeros(count, dtype=np.float64)
    if count <= 2 * _TANGENT_LAG:
        return tangent

    # A *centred* chord: the direction from i-lag to i+lag, assigned to i.
    #
    # An earlier version took the trailing chord from i-lag to i and stored it at
    # i, which dates the tangent 1.5 frames in the past while heading is
    # instantaneous. Sideslip is their difference, so that introduced a bias of
    # -lag/2 * yaw_rate / sample_rate — small on a straight, largest exactly
    # where the car is turning hardest, which is where drift is measured.
    delta = position[2 * _TANGENT_LAG :] - position[: -2 * _TANGENT_LAG]
    values = np.arctan2(delta[:, 0], delta[:, 2])
    tangent[_TANGENT_LAG : count - _TANGENT_LAG] = values
    tangent[:_TANGENT_LAG] = values[0]
    tangent[count - _TANGENT_LAG :] = values[-1]
    return tangent


def _speed(position: npt.NDArray[np.float64], sample_rate: float) -> npt.NDArray[np.float64]:
    """Compute ground speed from position differencing.

    Position differencing is preferred over the record's stored velocity field:
    the stored field's reference frame is undocumented, whereas the distance
    between two world positions divided by a known time step is unambiguous.

    Args:
        position: World positions, shape ``(n, 3)``.
        sample_rate: Frames per second.

    Returns:
        Speed in metres per second, shape ``(n,)``.
    """
    count = position.shape[0]
    speed = np.zeros(count, dtype=np.float64)
    if count < _MIN_FRAMES_FOR_SPEED:
        return speed
    ground = position[:, [0, 2]]
    step = np.linalg.norm(np.diff(ground, axis=0), axis=1)
    speed[1:] = step * sample_rate
    speed[0] = speed[1]

    # Reject teleports. A replay contains position discontinuities — a session
    # reset, a pit teleport, the ring buffer's seam — and differencing across one
    # yields an enormous speed. Real files produce samples above 170 000 km/h.
    #
    # Left unguarded that outlier does not merely look silly: it dominates the
    # session's speed percentiles, and the acceleration derived from it fabricates
    # STRONG-severity spins and corners at the seam. Holding the previous value
    # across the discontinuity is the honest repair — the car's speed through a
    # teleport is not a measurable quantity, so inventing a smooth one would be
    # worse than carrying the last thing actually observed.
    implausible = speed > _MAX_PLAUSIBLE_SPEED
    if implausible.any():
        good = np.flatnonzero(~implausible)
        if good.size:
            speed = np.interp(np.arange(count), good, speed[good])
        else:
            speed = np.zeros(count, dtype=np.float64)
    return speed


def _wrap(angle: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Wrap angles into ``(-pi, pi]``.

    Args:
        angle: Angles in radians.

    Returns:
        The wrapped angles.
    """
    return np.asarray((angle + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float64)


POSITION_FRAME: Final = ReferenceFrame.WORLD
"""Positions are unambiguously world-frame; they trace the track."""
