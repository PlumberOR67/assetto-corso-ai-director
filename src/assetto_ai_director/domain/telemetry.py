"""The telemetry model: channels, capability declaration and columnar storage.

The central design decision of this module is that a telemetry source
**declares what it has** rather than promising a fixed struct.

That is not academic. Assetto Corsa's shared memory exposes one car and has no
per-wheel slip angle at all; the replay file on disk carries every car with
per-wheel slip angle, slip ratio, normalised slip and load. An interface with an
unconditional ``slip_angle`` field would force the shared-memory provider to
invent a number — and an invented zero is indistinguishable from a car driving
perfectly straight.

So :class:`ChannelAvailability` is part of the contract, consumers ask before
they read, and analyses degrade honestly instead of silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.domain.errors import ChannelUnavailableError
from assetto_ai_director.domain.frames import ReferenceFrame
from assetto_ai_director.domain.units import Hertz, Seconds

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

__all__ = [
    "CarTelemetry",
    "Channel",
    "ChannelAvailability",
    "SessionTelemetry",
]


class Channel(StrEnum):
    """A named telemetry channel.

    Membership here means "this project knows what this quantity is", not "every
    source provides it". Availability is per-source and declared separately.
    """

    # --- Kinematics (position is the one channel every source must have) ---
    POSITION = "position"
    """World-space position, metres. The irreducible minimum for a camera system."""

    VELOCITY = "velocity"
    """Stored velocity vector. Frame is source-dependent; see ``velocity_frame``."""

    SPEED = "speed"
    """Scalar ground speed, m/s. Derived from position differencing when absent."""

    ROTATION = "rotation"
    """Raw stored rotation. Euler order and handedness are source-dependent."""

    HEADING = "heading"
    """Validated ground-plane yaw, radians. Only present after calibration."""

    # --- Driver inputs ---
    THROTTLE = "throttle"
    BRAKE = "brake"
    STEER_ANGLE = "steer_angle"
    GEAR = "gear"
    RPM = "rpm"

    # --- Per-wheel (4-element arrays, ordered FL, FR, RL, RR) ---
    WHEEL_SLIP_ANGLE = "wheel_slip_angle"
    """Per-wheel tyre slip angle, radians. Present in replays, absent in shared memory."""

    WHEEL_SLIP_RATIO = "wheel_slip_ratio"
    WHEEL_ND_SLIP = "wheel_nd_slip"
    """Normalised slip: 1.0 is the grip limit. The basis of the Risk score."""

    WHEEL_LOAD = "wheel_load"
    WHEEL_ANGULAR_SPEED = "wheel_angular_speed"

    # --- Session state ---
    LAP = "lap"
    NORMALISED_POSITION = "normalised_position"
    """Progress along the track spline in ``[0, 1)``. Drives ``cameras.ini`` export."""

    LAP_TIME = "lap_time"
    DAMAGE = "damage"
    TYRES_OUT = "tyres_out"

    @property
    def is_per_wheel(self) -> bool:
        """Whether this channel stores four values per sample.

        Returns:
            ``True`` for per-wheel channels, which are stored as ``(n, 4)``
            arrays rather than ``(n,)``.
        """
        return self.value.startswith("wheel_")

    @property
    def is_vector(self) -> bool:
        """Whether this channel stores a 3-component vector per sample.

        Returns:
            ``True`` for channels stored as ``(n, 3)`` arrays.
        """
        return self in (Channel.POSITION, Channel.VELOCITY, Channel.ROTATION)


@dataclass(frozen=True, slots=True)
class ChannelAvailability:
    """What a telemetry source actually provides.

    Attributes:
        channels: The channels present.
        source_name: Human-readable source name, used in error messages so a
            user learns *which* provider lacks the data.
        velocity_frame: The frame :attr:`Channel.VELOCITY` is expressed in.
            Defaults to :attr:`ReferenceFrame.UNKNOWN` because no documentation
            establishes it for the replay format, and an unverified frame must
            not be silently trusted.
        rotation_calibrated: Whether the rotation convention has been validated
            against path tangent. Until it is, sideslip derived from heading is
            not trustworthy and the pipeline uses the rear-wheel slip proxy.
    """

    channels: frozenset[Channel]
    source_name: str
    velocity_frame: ReferenceFrame = ReferenceFrame.UNKNOWN
    rotation_calibrated: bool = False

    @classmethod
    def of(
        cls,
        source_name: str,
        channels: Iterable[Channel],
        *,
        velocity_frame: ReferenceFrame = ReferenceFrame.UNKNOWN,
        rotation_calibrated: bool = False,
    ) -> ChannelAvailability:
        """Build an availability declaration.

        Args:
            source_name: Human-readable name of the providing source.
            channels: The channels the source provides.
            velocity_frame: Frame of the velocity channel, if provided.
            rotation_calibrated: Whether rotation has been validated.

        Returns:
            The availability declaration.
        """
        return cls(frozenset(channels), source_name, velocity_frame, rotation_calibrated)

    def has(self, channel: Channel) -> bool:
        """Report whether a channel is available.

        Args:
            channel: The channel to test.

        Returns:
            ``True`` if the source provides it.
        """
        return channel in self.channels

    def has_all(self, *channels: Channel) -> bool:
        """Report whether every given channel is available.

        Args:
            *channels: The channels to test.

        Returns:
            ``True`` if all are provided.
        """
        return all(channel in self.channels for channel in channels)

    def require(self, *channels: Channel) -> None:
        """Assert that channels are available, raising if not.

        Args:
            *channels: The channels an analysis depends on.

        Raises:
            ChannelUnavailableError: On the first missing channel.
        """
        for channel in channels:
            if channel not in self.channels:
                raise ChannelUnavailableError(channel.value, self.source_name)

    @property
    def supports_true_sideslip(self) -> bool:
        """Whether real chassis sideslip can be computed from this source.

        Requires a validated heading. Without it the pipeline falls back to the
        rear-wheel slip-angle proxy, which is real measured data rather than a
        guess, but is a different quantity and is reported as degraded.

        Returns:
            ``True`` when heading is present and the rotation convention has been
            calibrated.
        """
        return self.rotation_calibrated and Channel.HEADING in self.channels

    @property
    def supports_drift_scoring(self) -> bool:
        """Whether drift can be scored at all, by any available route.

        Returns:
            ``True`` if either validated sideslip or per-wheel slip angle exists.
        """
        return self.supports_true_sideslip or Channel.WHEEL_SLIP_ANGLE in self.channels


@dataclass(frozen=True, slots=True)
class CarTelemetry:
    """One car's channels for a whole session, stored columnar.

    Arrays are shape ``(n,)`` for scalars, ``(n, 3)`` for vectors and ``(n, 4)``
    for per-wheel channels, where ``n`` is the session's frame count. All cars in
    a session share one time base, so index ``i`` is the same instant for every
    car — which is what makes multi-car analysis such as overtake detection a
    simple vectorised comparison.

    Attributes:
        car_id: Index of this car within the session.
        channels: Channel data keyed by channel.
    """

    car_id: int
    channels: dict[Channel, npt.NDArray[np.float64]]

    def get(self, channel: Channel) -> npt.NDArray[np.float64]:
        """Fetch a channel's array.

        Args:
            channel: The channel to fetch.

        Returns:
            The stored array.

        Raises:
            ChannelUnavailableError: If this car has no such channel.
        """
        try:
            return self.channels[channel]
        except KeyError:
            raise ChannelUnavailableError(channel.value, f"car {self.car_id}") from None

    @property
    def frame_count(self) -> int:
        """The number of samples in this car's channels.

        Returns:
            The sample count, or ``0`` if the car carries no channels.
        """
        if not self.channels:
            return 0
        return int(next(iter(self.channels.values())).shape[0])


@dataclass(frozen=True, slots=True)
class SessionTelemetry:
    """A complete session: every car, on one shared time base.

    Attributes:
        cars: Per-car telemetry, ordered by ``car_id``.
        sample_rate: Sampling rate in hertz.
        availability: What the originating source declared it provides.
        time: Monotonic time in seconds from session start, shape ``(n,)``.
    """

    cars: tuple[CarTelemetry, ...]
    sample_rate: Hertz
    availability: ChannelAvailability
    time: npt.NDArray[np.float64]

    @property
    def frame_count(self) -> int:
        """The number of samples on the shared time base.

        Returns:
            The sample count.
        """
        return int(self.time.shape[0])

    @property
    def duration(self) -> Seconds:
        """The wall-clock length of the session.

        Returns:
            Duration in seconds; ``0`` for an empty session.
        """
        if self.frame_count == 0:
            return Seconds(0.0)
        return Seconds(float(self.time[-1] - self.time[0]))

    @property
    def car_count(self) -> int:
        """The number of cars in the session.

        Returns:
            The car count.
        """
        return len(self.cars)

    def car(self, car_id: int) -> CarTelemetry:
        """Fetch one car's telemetry by id.

        Args:
            car_id: The car index.

        Returns:
            That car's telemetry.

        Raises:
            IndexError: If no car has that id.
        """
        for entry in self.cars:
            if entry.car_id == car_id:
                return entry
        message = f"No car with id {car_id}; session has {self.car_count} cars."
        raise IndexError(message)

    def index_at(self, moment: Seconds) -> int:
        """Find the sample index nearest a moment in time.

        Args:
            moment: Time in seconds from session start.

        Returns:
            The index of the nearest sample, clamped to the valid range.
        """
        if self.frame_count == 0:
            return 0
        position = int(np.searchsorted(self.time, moment))
        position = min(position, self.frame_count - 1)
        if position > 0 and abs(self.time[position - 1] - moment) <= abs(
            self.time[position] - moment
        ):
            return position - 1
        return position
