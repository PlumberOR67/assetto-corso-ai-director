"""Shots: the units the director actually schedules.

A :class:`ShotCandidate` is a *possibility* — this rig, this size, this side of
the line, over this span, worth roughly this much. The director generates many
and keeps few.

A :class:`Shot` is a candidate that survived scheduling and acquired a camera
path. The distinction matters because selection is a global optimisation: a
candidate's individual score is only one term, and a high-scoring candidate is
routinely rejected because of what it would do to the shots either side of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from assetto_ai_director.domain.camera import CameraPath, CameraSpec, RigKind
from assetto_ai_director.domain.events import Event
from assetto_ai_director.domain.units import Score, Seconds

__all__ = ["LineSide", "Shot", "ShotCandidate", "ShotSize"]

_JUMP_CUT_SIZE_STEPS = 2
"""How many steps apart two shot sizes must be to read as a deliberate change.

One step — a full shot to a medium — reads as an error rather than an edit.
"""

_TWO_SHOT_SUBJECTS = 2
"""Subjects required before a shot needs two-target framing."""


class ShotSize(IntEnum):
    """How large the subject sits in frame.

    Ordered from tightest to widest so relative size is a numeric comparison,
    which the jump-cut rule needs: two consecutive shots must differ markedly in
    size or angle, and "markedly" is easiest to express as a distance in this
    ordering.

    Attributes:
        EXTREME_CLOSE_UP: A detail — a tyre's contact patch, a shifting hand.
        CLOSE_UP: Part of the car fills the frame.
        MEDIUM: The whole car, tightly framed.
        FULL: The whole car with room around it.
        LONG: The car small within its surroundings.
        EXTREME_LONG: The landscape, with the car as an element in it.
    """

    EXTREME_CLOSE_UP = 0
    CLOSE_UP = 1
    MEDIUM = 2
    FULL = 3
    LONG = 4
    EXTREME_LONG = 5

    @property
    def is_tight(self) -> bool:
        """Whether this size reads as a close shot.

        Tight shots need establishing first — an audience shown a detail before
        they know where they are has to spend the shot working out the geography
        instead of watching the car.

        Returns:
            ``True`` for close-up and tighter.
        """
        return self <= ShotSize.CLOSE_UP

    @property
    def is_establishing(self) -> bool:
        """Whether this size can establish geography.

        Returns:
            ``True`` for long and wider.
        """
        return self >= ShotSize.LONG

    def differs_markedly_from(self, other: ShotSize) -> bool:
        """Test whether cutting between two sizes avoids a jump cut.

        The classical requirement is a *marked* difference in size, view or
        number of subjects. Two steps in this ordering is the threshold used
        here — a full shot cutting to a close-up reads as a deliberate move,
        whereas a full shot cutting to a medium reads as a glitch.

        Args:
            other: The size being cut to.

        Returns:
            ``True`` if the difference is large enough.
        """
        return abs(int(self) - int(other)) >= _JUMP_CUT_SIZE_STEPS


class LineSide(StrEnum):
    """Which side of the line of action the camera sits on.

    The line of action for a racing shot is the direction of travel. Crossing it
    between consecutive shots flips the car's apparent direction on screen, so a
    car travelling left-to-right suddenly travels right-to-left, and the audience
    reads it as a different car or a reversal.

    This is the one continuity rule the scheduler treats as an absolute
    prohibition rather than a cost.

    Attributes:
        LEFT: Camera on the left of the direction of travel.
        RIGHT: Camera on the right.
        NEUTRAL: On the line itself — head-on, tail-on, or directly overhead.
            Neutral shots are the legal way to cross: a camera may move to the
            far side *through* a neutral shot without a visible reversal.
    """

    LEFT = "left"
    RIGHT = "right"
    NEUTRAL = "neutral"

    def crosses_line_with(self, other: LineSide) -> bool:
        """Test whether cutting between two sides would reverse screen direction.

        Args:
            other: The side being cut to.

        Returns:
            ``True`` only when cutting directly between opposite sides. Either
            side to or from neutral is always legal.
        """
        if self is LineSide.NEUTRAL or other is LineSide.NEUTRAL:
            return False
        return self is not other


@dataclass(frozen=True, slots=True)
class ShotCandidate:
    """A shot the director could take, with what it would be worth.

    Attributes:
        start: When the shot would begin, in seconds from session start.
        end: When it would end.
        subjects: The cars it would frame, most important first. One subject is
            a simple framing problem; two require Toric-space solving.
        rig: Which camera behaviour.
        size: How large the subject would sit in frame.
        side: Which side of the line of action.
        score: The style-weighted value of the moment this covers.
        anchor: The event that motivated this candidate, if any. Candidates also
            exist for uneventful spans, because a cut has to tile the whole
            timeline and something must cover the quiet parts.
        spec: The resolved camera parameters.
    """

    start: Seconds
    end: Seconds
    subjects: tuple[int, ...]
    rig: RigKind
    size: ShotSize
    side: LineSide
    score: Score
    anchor: Event | None = None
    spec: CameraSpec | None = None

    def __post_init__(self) -> None:
        """Validate the candidate.

        Raises:
            ValueError: If the span is not positive or no subject is given.
        """
        if self.end <= self.start:
            message = f"Shot candidate must have positive duration, got {self.start}..{self.end}."
            raise ValueError(message)
        if not self.subjects:
            message = "Shot candidate must have at least one subject."
            raise ValueError(message)

    @property
    def duration(self) -> Seconds:
        """How long the shot would last.

        Returns:
            The span in seconds.
        """
        return Seconds(self.end - self.start)

    @property
    def primary_subject(self) -> int:
        """The car the shot is principally about.

        Returns:
            The first subject's car id.
        """
        return self.subjects[0]

    @property
    def is_two_shot(self) -> bool:
        """Whether the shot frames two cars at once.

        Returns:
            ``True`` for two-subject candidates.
        """
        return len(self.subjects) >= _TWO_SHOT_SUBJECTS

    def shares_subject_with(self, other: ShotCandidate) -> bool:
        """Test whether two candidates are about any of the same cars.

        Continuity rules apply between shots of the *same* subject. Cutting from
        one car to a different car is a change of subject, not a line crossing,
        and must not be penalised as one.

        Args:
            other: The candidate to compare against.

        Returns:
            ``True`` if the subject sets intersect.
        """
        return bool(set(self.subjects) & set(other.subjects))


@dataclass(frozen=True, slots=True)
class Shot:
    """A scheduled shot with a generated camera path.

    Attributes:
        candidate: The candidate that was selected.
        path: The camera's motion over the shot.
        index: Position in the final cut, counting from zero.
        smoothness_cost: The evaluated cost of this camera move. Computed after
            path generation, since it is a property of the trajectory rather than
            of the moment, and fed back as a penalty on the selection.
    """

    candidate: ShotCandidate
    path: CameraPath
    index: int
    smoothness_cost: float = 0.0

    @property
    def start(self) -> Seconds:
        """When the shot begins.

        Returns:
            The candidate's start time.
        """
        return self.candidate.start

    @property
    def end(self) -> Seconds:
        """When the shot ends.

        Returns:
            The candidate's end time.
        """
        return self.candidate.end

    @property
    def duration(self) -> Seconds:
        """How long the shot lasts.

        Returns:
            The span in seconds.
        """
        return self.candidate.duration
