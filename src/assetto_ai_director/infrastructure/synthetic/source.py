"""A synthetic driving session with events at known times.

**Why this exists even though real replays are available.** A test that depends
on a 300 MB file the repository cannot contain is a test that does not run for
anyone else — the fixture suite skips silently on a fresh clone, and a
contributor gets a green build that verified far less than they think. This
provider gives the whole pipeline something to run on that is committed, fast,
and — crucially — has a *known answer*.

That last point is what makes it more than a stand-in. Real telemetry can only be
checked for plausibility: a detector that finds fifteen drifts on Akina is
probably right, but nothing proves it. Here the drifts are scripted, so a detector
can be asked whether it found *those* drifts, at *those* times.

**It is not a physics simulation and does not pretend to be.** The car follows a
procedural path at a procedural speed with procedurally scripted slip. It
produces signals with the right shape, magnitude and units for the analysis
layer, which is what the analysis layer needs. Anything asking whether a
manoeuvre is physically achievable must use a real replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.application.events.detect import CarSignals
from assetto_ai_director.domain.frames import ReferenceFrame
from assetto_ai_director.domain.identity import CarIdentity
from assetto_ai_director.domain.session import SessionMeta
from assetto_ai_director.domain.telemetry import Channel, ChannelAvailability
from assetto_ai_director.domain.units import Hertz, Seconds

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ScriptedDrift",
    "ScriptedOvertake",
    "SyntheticSession",
    "SyntheticSource",
]

DEFAULT_RATE = Hertz(33.3)
"""Matches the recording rate of the real fixtures, so timing behaves alike."""

_CORNER_PERIOD_SECONDS = 9.0
"""How often the synthetic road turns. Roughly a touge corner rhythm."""

_CORNER_RADIUS = 45.0
"""Radius of the synthetic corners, metres. Comfortably inside the detector's
corner threshold so segmentation has something unambiguous to find."""

_BASE_SPEED = 25.0
_SPEED_SWING = 8.0

_OVERTAKE_START_LAG = 18.0
"""How far behind an overtaker begins, metres.

Inside the detector's interaction range so the two cars count as racing, and far
enough that the gap has a decisive magnitude on both sides of the crossing —
which the overtake detector requires before it will call a pass committed.
"""


@dataclass(frozen=True, slots=True)
class ScriptedDrift:
    """A slide placed deliberately, so a detector can be asked to find it.

    Attributes:
        start: When it begins, seconds.
        duration: How long it lasts, seconds.
        peak_degrees: Peak sideslip angle.
        car_id: Which car slides.
    """

    start: Seconds
    duration: Seconds
    peak_degrees: float = 28.0
    car_id: int = 0

    @property
    def end(self) -> Seconds:
        """When the slide ends.

        Returns:
            The end time in seconds.
        """
        return Seconds(float(self.start) + float(self.duration))

    @property
    def peak(self) -> Seconds:
        """When the slide is at its widest angle.

        Returns:
            The peak time in seconds.
        """
        return Seconds(float(self.start) + float(self.duration) * 0.5)


@dataclass(frozen=True, slots=True)
class ScriptedOvertake:
    """A pass placed deliberately.

    Attributes:
        at: When the order actually changes, seconds.
        overtaker: The car that comes through.
        overtaken: The car that is passed.
    """

    at: Seconds
    overtaker: int = 1
    overtaken: int = 0


@dataclass(frozen=True, slots=True)
class SyntheticSession:
    """A generated session and the ground truth behind it.

    Attributes:
        signals: Decoded per-car signals, ready for the pipeline.
        meta: Session metadata.
        time: Time per frame, seconds.
        sample_rate: Frames per second.
        drifts: The slides that were scripted in.
        overtakes: The passes that were scripted in.
    """

    signals: dict[int, CarSignals]
    meta: SessionMeta
    time: np.ndarray
    sample_rate: Hertz
    drifts: tuple[ScriptedDrift, ...] = field(default_factory=tuple)
    overtakes: tuple[ScriptedOvertake, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> Seconds:
        """How long the session runs.

        Returns:
            Duration in seconds.
        """
        return Seconds(float(self.time[-1]) if self.time.size else 0.0)


class SyntheticSource:
    """Generates a session procedurally.

    Satisfies the ``TelemetrySource`` protocol structurally, and additionally
    exposes the ground truth through :meth:`generate`.
    """

    def __init__(
        self,
        *,
        seconds: float = 60.0,
        cars: int = 2,
        sample_rate: Hertz = DEFAULT_RATE,
        drifts: Sequence[ScriptedDrift] | None = None,
        overtakes: Sequence[ScriptedOvertake] | None = None,
        seed: int = 0,
    ) -> None:
        """Configure the generator.

        Args:
            seconds: Session length.
            cars: How many cars to generate.
            sample_rate: Frames per second.
            drifts: Slides to script in. A default pair is used when omitted.
            overtakes: Passes to script in. A default is used when omitted.
            seed: Seed for the small amount of noise added, so a generated
                session is reproducible.
        """
        self.seconds = seconds
        self.cars = max(1, cars)
        self.sample_rate = sample_rate
        self.seed = seed
        self.drifts = tuple(drifts) if drifts is not None else self._default_drifts()
        self.overtakes = (
            tuple(overtakes)
            if overtakes is not None
            else ((ScriptedOvertake(Seconds(seconds * 0.6)),) if self.cars > 1 else ())
        )

    @property
    def name(self) -> str:
        """A short identifier for this source.

        Returns:
            The source name.
        """
        return "synthetic"

    def describe(self) -> SessionMeta:
        """Read the session's metadata without generating telemetry.

        Returns:
            The session metadata.
        """
        return SessionMeta(
            track_id="synthetic",
            layout="loop",
            weather_id="clear",
            sample_rate=self.sample_rate,
            frame_count=self._frame_count,
            cars=tuple(
                CarIdentity(
                    car_id=index,
                    model="synthetic_car",
                    driver_name=f"driver_{index}",
                    is_player=index == 0,
                )
                for index in range(self.cars)
            ),
            source_name=self.name,
        )

    def availability(self) -> ChannelAvailability:
        """Declare which channels this source provides.

        Returns:
            The capability declaration. Deliberately the same set the replay
            reader offers, so nothing tested here passes only because the
            synthetic source is more generous than reality.
        """
        return ChannelAvailability.of(
            self.name,
            (Channel.POSITION, Channel.SPEED, Channel.HEADING),
            velocity_frame=ReferenceFrame.WORLD,
            rotation_calibrated=True,
        )

    def generate(self) -> SyntheticSession:
        """Generate the session and its ground truth.

        Returns:
            The session, with the scripted events recorded alongside.
        """
        count = self._frame_count
        rate = float(self.sample_rate)
        time = np.arange(count, dtype=np.float64) / rate

        signals: dict[int, CarSignals] = {}
        for car_id in range(self.cars):
            position, heading, speed = self._drive(car_id, time)
            sideslip = self._sideslip(car_id, time)
            signals[car_id] = CarSignals(car_id, position, heading, speed, sideslip)

        return SyntheticSession(
            signals=signals,
            meta=self.describe(),
            time=time,
            sample_rate=self.sample_rate,
            drifts=self.drifts,
            overtakes=self.overtakes,
        )

    def load(self, progress: object = None) -> SyntheticSession:
        """Generate the session, matching the ``TelemetrySource`` port's shape.

        Args:
            progress: Ignored; generation is instantaneous.

        Returns:
            The generated session.
        """
        del progress
        return self.generate()

    @property
    def _frame_count(self) -> int:
        """How many frames the configured session spans.

        Returns:
            The frame count.
        """
        return max(int(self.seconds * float(self.sample_rate)), 2)

    def _default_drifts(self) -> tuple[ScriptedDrift, ...]:
        """Build the slides used when the caller scripts none.

        Returns:
            Two slides of different magnitudes, spaced well apart.
        """
        return (
            ScriptedDrift(Seconds(self.seconds * 0.2), Seconds(3.0), 32.0, 0),
            ScriptedDrift(Seconds(self.seconds * 0.75), Seconds(2.0), 18.0, 0),
        )

    def _drive(self, car_id: int, time: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate one car's path, heading and speed.

        The road is a sequence of alternating corners of fixed radius joined by
        straights, which gives the corner detector unambiguous geometry to find
        without needing a real track.

        Args:
            car_id: Which car.
            time: Time per frame.

        Returns:
            ``(position, heading, speed)``.
        """
        rate = float(self.sample_rate)
        phase = 2.0 * math.pi * time / _CORNER_PERIOD_SECONDS

        speed = _BASE_SPEED + _SPEED_SWING * np.sin(phase * 0.5)
        speed += self._overtake_delta(car_id, time)
        speed = np.maximum(speed, 5.0)

        # Turn rate follows from the corner radius and the current speed, so the
        # geometry stays a genuine circle of that radius at any pace.
        turn_rate = np.sin(phase) * (speed / _CORNER_RADIUS)
        heading = np.cumsum(turn_rate) / rate

        step = speed / rate
        # An overtaker starts behind. Without this both cars begin level and the
        # along-track gap only ever moves one way, so there is no order change
        # for the detector to find — the scripted pass would never happen.
        lag = self._starting_lag(car_id)
        x = np.cumsum(np.sin(heading) * step) - math.sin(float(heading[0])) * lag
        z = np.cumsum(np.cos(heading) * step) - math.cos(float(heading[0])) * lag
        # A gentle descent, so the cinematic score's elevation term has signal.
        y = -time * 0.4

        lateral = 0.0 if car_id == 0 else 3.5 * car_id
        offset = np.stack(
            [np.cos(heading) * lateral, np.zeros_like(x), -np.sin(heading) * lateral], axis=1
        )
        position = np.stack([x, y, z], axis=1) + offset
        return position, _wrap(heading), speed

    def _starting_lag(self, car_id: int) -> float:
        """How far behind the pack a car begins, metres.

        Args:
            car_id: Which car.

        Returns:
            The lag. Non-zero only for cars scripted to overtake someone, so the
            pass has somewhere to come from.
        """
        return _OVERTAKE_START_LAG if any(o.overtaker == car_id for o in self.overtakes) else 0.0

    def _overtake_delta(self, car_id: int, time: np.ndarray) -> np.ndarray:
        """Add the speed difference that produces a scripted pass.

        Args:
            car_id: Which car.
            time: Time per frame.

        Returns:
            A speed offset per frame, metres per second.
        """
        delta = np.zeros_like(time)
        for overtake in self.overtakes:
            if car_id != overtake.overtaker:
                continue
            window = 6.0
            centre = float(overtake.at)
            delta += 6.0 * np.exp(-(((time - centre) / window) ** 2))
        return delta

    def _sideslip(self, car_id: int, time: np.ndarray) -> np.ndarray:
        """Generate sideslip, with the scripted slides written in.

        Args:
            car_id: Which car.
            time: Time per frame.

        Returns:
            Sideslip per frame, radians.
        """
        generator = np.random.default_rng(self.seed + car_id)
        # A little noise so detectors face a realistic signal rather than a
        # perfectly clean one that any threshold would separate.
        signal = generator.normal(0.0, math.radians(1.2), size=time.shape)

        for drift in self.drifts:
            if drift.car_id != car_id:
                continue
            start, end = float(drift.start), float(drift.end)
            inside = (time >= start) & (time <= end)
            if not inside.any():
                continue
            # A raised-cosine envelope: rises, holds, falls, so the peak lands in
            # the middle and the detector's hysteresis has real edges to find.
            local = (time[inside] - start) / max(end - start, 1e-6)
            envelope = 0.5 - 0.5 * np.cos(2.0 * math.pi * local)
            signal[inside] += math.radians(drift.peak_degrees) * envelope
        return signal


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles into ``(-pi, pi]``.

    Args:
        angle: Angles in radians.

    Returns:
        The wrapped angles.
    """
    return np.asarray((angle + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float64)
