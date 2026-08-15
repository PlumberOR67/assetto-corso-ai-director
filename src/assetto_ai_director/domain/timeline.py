"""The timeline: the director's finished output.

This is the product. Everything else — the GUI, the ``cameras.ini`` exporter, the
MCP server — is a projection of this object.

A timeline **tiles** its horizon: consecutive shots meet exactly, with no gaps
and no overlaps. That is stronger than requiring the shots to be disjoint, and it
is deliberate. A cut is a continuous experience; a gap would mean a moment with
no camera assigned, which is not a thing an edit can contain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise

from assetto_ai_director.domain.session import SessionMeta
from assetto_ai_director.domain.shot import Shot
from assetto_ai_director.domain.units import Seconds

__all__ = ["Cut", "Timeline", "TransitionKind"]

SCHEMA_VERSION = 1
"""Version of the serialised timeline schema.

Bumped whenever the JSON shape changes incompatibly, so a consumer reading a file
written by a different version can refuse it rather than silently misinterpret it.
"""

_TILING_TOLERANCE = 1e-6
"""Seconds of slack allowed when checking that shots meet exactly.

Shot boundaries are computed in floating point from sample indices, so exact
equality is the wrong test. A microsecond is far below one frame at any
achievable recording rate.
"""


class TransitionKind(StrEnum):
    """How one shot gives way to the next.

    Attributes:
        CUT: An instantaneous change. The default, and the overwhelming majority
            of transitions in any of the reference styles.
        DISSOLVE: A cross-fade. Reads as a passage of time.
        WHIP: A fast pan that blurs through the change, disguising the cut.
        SPEED_RAMP: A change of playback rate across the boundary, usually into
            or out of slow motion.
        FADE: Through black. Reserved for the very start and end.
    """

    CUT = "cut"
    DISSOLVE = "dissolve"
    WHIP = "whip"
    SPEED_RAMP = "speed_ramp"
    FADE = "fade"

    @property
    def is_instantaneous(self) -> bool:
        """Whether the transition occupies no time.

        Returns:
            ``True`` only for a straight cut.
        """
        return self is TransitionKind.CUT


@dataclass(frozen=True, slots=True)
class Cut:
    """The boundary between two consecutive shots.

    Attributes:
        time: When the change happens, in seconds from session start.
        kind: How the change happens.
        duration: How long a non-instantaneous transition takes. Zero for a cut.
        from_index: Index of the outgoing shot, or ``None`` at the very start.
        to_index: Index of the incoming shot, or ``None`` at the very end.
        speed_factor: Playback rate after this boundary. ``1.0`` is real time;
            ``0.4`` is slow motion. Applied by exporters that can retime, and
            reported as unsupported by those that cannot.
    """

    time: Seconds
    kind: TransitionKind = TransitionKind.CUT
    duration: Seconds = Seconds(0.0)
    from_index: int | None = None
    to_index: int | None = None
    speed_factor: float = 1.0

    def __post_init__(self) -> None:
        """Validate the transition.

        Raises:
            ValueError: If a cut is given a duration, or the speed factor is not
                positive.
        """
        if self.kind.is_instantaneous and self.duration > 0.0:
            message = f"A {self.kind.value} transition cannot have duration {self.duration}."
            raise ValueError(message)
        if self.speed_factor <= 0.0:
            message = f"Speed factor must be positive, got {self.speed_factor}."
            raise ValueError(message)

    @property
    def is_slow_motion(self) -> bool:
        """Whether playback slows after this boundary.

        Returns:
            ``True`` if the speed factor is below real time.
        """
        return self.speed_factor < 1.0


@dataclass(frozen=True, slots=True)
class Timeline:
    """A complete directed edit.

    Attributes:
        shots: The shots, ordered by start time and tiling the horizon.
        cuts: The boundaries between them, one per shot transition.
        session: Metadata about the session this was directed from.
        style_name: Which style profile produced it.
        schema_version: Serialisation schema version.
        notes: Free-form annotations for the user — degraded channels, sample
            rate warnings, constraints that could not be satisfied. This is how
            the timeline stays honest about its own limitations instead of
            presenting every edit as equally well-founded.
    """

    shots: tuple[Shot, ...]
    cuts: tuple[Cut, ...]
    session: SessionMeta
    style_name: str
    schema_version: int = SCHEMA_VERSION
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def start(self) -> Seconds:
        """When the edit begins.

        Returns:
            The first shot's start, or zero for an empty timeline.
        """
        return self.shots[0].start if self.shots else Seconds(0.0)

    @property
    def end(self) -> Seconds:
        """When the edit ends.

        Returns:
            The last shot's end, or zero for an empty timeline.
        """
        return self.shots[-1].end if self.shots else Seconds(0.0)

    @property
    def duration(self) -> Seconds:
        """How long the edit runs.

        Returns:
            The span in seconds.
        """
        return Seconds(self.end - self.start)

    @property
    def shot_count(self) -> int:
        """How many shots the edit contains.

        Returns:
            The shot count.
        """
        return len(self.shots)

    @property
    def average_shot_length(self) -> Seconds:
        """The mean shot duration.

        The standard single-number summary of an edit's pace, and the primary
        quantity the style measurement harness extracts from reference footage.

        Returns:
            Mean duration in seconds, or zero for an empty timeline.
        """
        if not self.shots:
            return Seconds(0.0)
        return Seconds(sum(shot.duration for shot in self.shots) / len(self.shots))

    @property
    def shot_durations(self) -> tuple[Seconds, ...]:
        """Every shot's duration, in order.

        The input to the 1/f rhythm check: real edits have shot lengths that are
        pink-noise correlated rather than independent, and the sequence order is
        what carries that structure.

        Returns:
            The durations in shot order.
        """
        return tuple(shot.duration for shot in self.shots)

    def tiles_without_gaps(self, tolerance: float = _TILING_TOLERANCE) -> bool:
        """Verify that consecutive shots meet exactly.

        Args:
            tolerance: Permitted slack in seconds, to absorb floating-point
                error in boundaries computed from sample indices.

        Returns:
            ``True`` if every shot begins where the previous one ended.
        """
        return all(
            abs(later.start - earlier.end) <= tolerance for earlier, later in pairwise(self.shots)
        )

    def shot_at(self, moment: Seconds) -> Shot | None:
        """Find the shot covering a moment.

        Args:
            moment: The time of interest, in seconds from session start.

        Returns:
            The covering shot, or ``None`` if the moment lies outside the edit.
        """
        for shot in self.shots:
            if shot.start <= moment < shot.end:
                return shot
        if self.shots and moment == self.end:
            return self.shots[-1]
        return None

    def with_note(self, note: str) -> Timeline:
        """Return a copy carrying an extra annotation.

        Args:
            note: The annotation to append.

        Returns:
            A new timeline with the note added.
        """
        return Timeline(
            self.shots,
            self.cuts,
            self.session,
            self.style_name,
            self.schema_version,
            (*self.notes, note),
        )
