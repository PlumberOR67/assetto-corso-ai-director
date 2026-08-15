"""The cost of cutting from one shot to the next.

These are the rules that separate an edit from a slideshow of good moments. They
come from He, Cohen and Salesin's *The Virtual Cinematographer* (SIGGRAPH '96),
which encoded them as hard constraints in a film-idiom state machine; here they
are transition costs in a dynamic program, following Galvane et al.'s
reformulation of continuity editing as an optimisation.

**Why any of this is necessary.** Greedily picking the highest-scoring moments
produces an edit that crosses the line of action and jump-cuts constantly,
because both faults are invisible to a per-moment score — they exist only in the
relationship between consecutive shots. The scoring layer cannot see them by
construction. This module is where they become expensive.

One rule is absolute rather than costly: crossing the line of action. Cut from
the left of a car to the right and it appears to reverse direction on screen,
which an audience reads as a different car or a different lap. There is no score
high enough to be worth that, so it is infinite rather than merely dear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from assetto_ai_director.domain.shot import ShotSize

if TYPE_CHECKING:
    from assetto_ai_director.domain.shot import ShotCandidate

__all__ = ["ContinuityRules", "transition_cost"]

IMPOSSIBLE = math.inf
"""Cost of a transition that must never be selected."""


@dataclass(frozen=True, slots=True)
class ContinuityRules:
    """How strictly a style enforces each editing convention.

    Attributes:
        jump_cut: Penalty for cutting between two shots too similar to read as a
            deliberate change. The classical requirement is a marked difference
            in size, angle or number of subjects.
        repeat_rig: Penalty for using the same rig twice running. Not a classical
            rule, but without it the scheduler discovers that one good camera
            position can cover a whole session, and the result reads as a fixed
            webcam rather than an edit.
        repeat_size: Penalty for holding the same shot size across a cut. Same
            argument, different axis: the jump-cut rule is satisfied by a change
            of *any* one dimension, so a scheduler alternating two rigs pays
            nothing for framing every shot identically. Measured on a real edit,
            that produced sixty consecutive full shots.
        establish: Penalty for a tight shot with no wider shot before it. An
            audience shown a detail before they know where they are spends the
            shot working out the geography instead of watching the car.
            Set to zero by styles that deliberately disorient.
        subject_switch: Penalty for changing which car the edit is about.
            Discourages ping-ponging between cars without forbidding it.
        rhythm: Weight on deviation from the target cutting rhythm.
        enforce_line: Whether crossing the line of action is forbidden outright.
    """

    jump_cut: float = 0.35
    repeat_rig: float = 0.15
    repeat_size: float = 0.12
    establish: float = 0.25
    subject_switch: float = 0.2
    rhythm: float = 0.3
    enforce_line: bool = True


def transition_cost(
    previous: ShotCandidate | None,
    current: ShotCandidate,
    rules: ContinuityRules,
    target_duration: float | None = None,
) -> float:
    """Compute what it costs to cut from one shot into another.

    Args:
        previous: The outgoing shot, or ``None`` at the start of the edit.
        current: The incoming shot.
        rules: How strictly this style enforces each convention.
        target_duration: The rhythm's target length for this position in the
            edit, or ``None`` to skip the rhythm term.

    Returns:
        The cost, or :data:`IMPOSSIBLE` when the transition is forbidden.
    """
    cost = 0.0

    if target_duration is not None and target_duration > 0.0:
        # Compared in the log domain because shot lengths are log-normally
        # distributed: being half a second short of a 1 s target is a large
        # error, and half a second short of an 8 s target is nothing.
        cost += rules.rhythm * abs(math.log(float(current.duration)) - math.log(target_duration))

    if previous is None:
        # An edit should open by establishing where it is.
        if current.size.is_tight:
            cost += rules.establish
        return cost

    same_subject = previous.shares_subject_with(current)

    if same_subject and previous.side.crosses_line_with(current.side):
        if rules.enforce_line:
            return IMPOSSIBLE
        cost += 1.0

    if same_subject and not _reads_as_a_change(previous, current):
        cost += rules.jump_cut

    if previous.rig is current.rig:
        cost += rules.repeat_rig

    if previous.size is current.size:
        cost += rules.repeat_size

    if current.size.is_tight and not previous.size.is_establishing:
        cost += rules.establish

    if not same_subject:
        cost += rules.subject_switch

    return cost


def _reads_as_a_change(previous: ShotCandidate, current: ShotCandidate) -> bool:
    """Test whether a cut is visibly deliberate rather than a glitch.

    Satisfied by a marked difference in any one of the three dimensions the
    classical rule names: apparent size, camera angle, or the number of subjects
    in frame.

    Args:
        previous: The outgoing shot.
        current: The incoming shot.

    Returns:
        ``True`` if the cut reads as an edit.
    """
    if previous.size.differs_markedly_from(current.size):
        return True
    if previous.side is not current.side:
        # Left to neutral, or neutral to right: a real change of angle without
        # crossing the line.
        return True
    if len(previous.subjects) != len(current.subjects):
        return True
    return previous.rig is not current.rig and _rig_angle_differs(previous, current)


def _rig_angle_differs(previous: ShotCandidate, current: ShotCandidate) -> bool:
    """Test whether two rigs frame the car from meaningfully different angles.

    A rough stand-in for the classical thirty-degree rule, which this project
    cannot apply literally: the real angular separation is only known once camera
    paths are generated, which happens after scheduling. Rig class is the best
    proxy available at selection time — a trackside camera and a bonnet mount are
    unarguably more than thirty degrees apart.

    Args:
        previous: The outgoing shot.
        current: The incoming shot.

    Returns:
        ``True`` if the two rigs see the car differently enough.
    """
    return (
        previous.rig.is_car_mounted != current.rig.is_car_mounted
        or previous.rig.is_fixed_position != current.rig.is_fixed_position
        or previous.rig.is_free_flying != current.rig.is_free_flying
    )


def establishing_deficit(shots: list[ShotCandidate]) -> int:
    """Count tight shots that arrive without an establishing shot before them.

    Exposed for tests and diagnostics: the scheduler prices this rule, but a
    property test wants to assert the finished edit actually respects it.

    Args:
        shots: The selected shots, in order.

    Returns:
        How many tight shots lack a wider predecessor on the same subject.
    """
    deficit = 0
    for index, shot in enumerate(shots):
        if not shot.size.is_tight:
            continue
        established = any(
            earlier.size >= ShotSize.FULL and earlier.shares_subject_with(shot)
            for earlier in shots[max(0, index - 3) : index]
        )
        if not established:
            deficit += 1
    return deficit
