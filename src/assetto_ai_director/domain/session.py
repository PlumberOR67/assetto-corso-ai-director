"""Session metadata: what was driven, where, and at what fidelity.

The most consequential field here is :attr:`SessionMeta.sample_rate`.

Assetto Corsa's replay recording quality is a user setting, and its levels map to
very different sampling rates. At the low end, a replay simply does not contain
enough temporal resolution to locate a braking point or the onset of a drift, and
no amount of clever analysis recovers information the file never stored.

So the rate is a first-class session property, surfaced in the GUI and checked
against a per-style floor. A director that silently produces a confident edit
from 8 Hz data would be lying about how much it knows.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from assetto_ai_director.domain.identity import CarIdentity
from assetto_ai_director.domain.units import Hertz, Seconds

__all__ = ["RecordingQuality", "SessionMeta"]

_UNUSABLE_BELOW_HZ = 10.0
"""Below this rate a 30 ms brake application can fall entirely between samples."""

_COARSE_BELOW_HZ = 20.0
"""Below this rate corner structure survives but event onsets do not."""

_GOOD_BELOW_HZ = 40.0
"""Above this rate there is enough resolution for slow motion and frame-accurate cuts."""


class RecordingQuality(IntEnum):
    """Qualitative bands for a session's temporal resolution.

    The bands are this project's own judgement about what each rate can support,
    not a Kunos classification. They exist so the GUI and CLI can say something
    more useful than a bare number.

    Attributes:
        UNUSABLE: Below roughly 10 Hz. Position is fine for a track map, but
            event timing is meaningless: a 30 ms brake application falls between
            samples.
        COARSE: Roughly 10-20 Hz. Corner-level structure survives; the precise
            instant of turn-in or drift initiation does not.
        GOOD: Roughly 20-40 Hz. Enough for reliable event detection and cutting.
        EXCELLENT: Above roughly 40 Hz. Enough for slow-motion ramps and
            frame-accurate cut placement.
    """

    UNUSABLE = 0
    COARSE = 1
    GOOD = 2
    EXCELLENT = 3

    @classmethod
    def from_rate(cls, rate: Hertz) -> RecordingQuality:
        """Classify a sampling rate into a quality band.

        Args:
            rate: The session's sampling rate in hertz.

        Returns:
            The band that rate falls into.
        """
        if rate < _UNUSABLE_BELOW_HZ:
            return cls.UNUSABLE
        if rate < _COARSE_BELOW_HZ:
            return cls.COARSE
        if rate < _GOOD_BELOW_HZ:
            return cls.GOOD
        return cls.EXCELLENT

    @property
    def supports_event_timing(self) -> bool:
        """Whether this band can locate events precisely enough to cut on them.

        Returns:
            ``True`` for :attr:`GOOD` and :attr:`EXCELLENT`.
        """
        return self >= RecordingQuality.GOOD

    @property
    def advice(self) -> str:
        """A one-line explanation to show the user.

        Returns:
            Plain-language guidance about what this rate can and cannot support.
        """
        match self:
            case RecordingQuality.UNUSABLE:
                return (
                    "Recording rate is too low for event timing. Raise Assetto "
                    "Corsa's replay quality and record again."
                )
            case RecordingQuality.COARSE:
                return (
                    "Recording rate resolves corners but not precise event "
                    "onsets. Cuts may land slightly late."
                )
            case RecordingQuality.GOOD:
                return "Recording rate is sufficient for reliable event detection."
            case RecordingQuality.EXCELLENT:
                return "Recording rate supports frame-accurate cutting and slow motion."


@dataclass(frozen=True, slots=True)
class SessionMeta:
    """Everything known about a session that is not per-frame telemetry.

    Attributes:
        track_id: Track folder name, e.g. ``pk_akina``.
        layout: Track layout name, e.g. ``akina_downhill``. Empty for tracks with
            a single layout. Load-bearing for ``cameras.ini`` export, whose data
            directory is layout-scoped.
        weather_id: Weather preset name, e.g. ``7_heavy_clouds``.
        sample_rate: Sampling rate in hertz.
        frame_count: Number of recorded frames.
        cars: Identity of every car in the session, ordered by ``car_id``.
        source_name: Which provider produced this session.
        wrapped: Whether the game's ring buffer overwrote the session start.
            When true, the recording begins mid-session and the earliest frames
            are the *most recent* ones in wall-clock terms; treating the file as
            a simple chronological sequence would splice the end onto the front.
    """

    track_id: str
    layout: str
    weather_id: str
    sample_rate: Hertz
    frame_count: int
    cars: tuple[CarIdentity, ...]
    source_name: str
    wrapped: bool = False

    @property
    def duration(self) -> Seconds:
        """The session's wall-clock length.

        Returns:
            Duration in seconds, derived from frame count and sample rate.
        """
        if self.sample_rate <= 0.0:
            return Seconds(0.0)
        return Seconds(self.frame_count / self.sample_rate)

    @property
    def quality(self) -> RecordingQuality:
        """The temporal-resolution band this session falls into.

        Returns:
            The recording quality band.
        """
        return RecordingQuality.from_rate(self.sample_rate)

    @property
    def car_count(self) -> int:
        """How many cars the session contains.

        Returns:
            The car count.
        """
        return len(self.cars)

    @property
    def track_label(self) -> str:
        """A display label combining track and layout.

        Returns:
            ``track/layout`` when a layout is set, otherwise just the track id.
        """
        return f"{self.track_id}/{self.layout}" if self.layout else self.track_id

    def player(self) -> CarIdentity | None:
        """Find the human-driven car, if there is one.

        Returns:
            The player's identity, or ``None`` for a fully AI session.
        """
        return next((car for car in self.cars if car.is_player), None)
