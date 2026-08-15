"""Shot selection: a longest path through the editing graph.

**Why this is a dynamic program and not a ranked list.** Picking the highest
scoring moments greedily gives an edit that crosses the line of action, jump
cuts, and repeats one camera for a minute — because none of those faults are
visible in a per-moment score. They exist only between consecutive shots. Any
method that scores shots independently is structurally blind to them.

The formulation follows Galvane, Ronfard, Lino and Christie, *Continuity Editing
for 3D Animation* (AAAI-15), which casts editing as a minimum-cost path through
an editing graph under a semi-Markov assumption, with a cost decomposed into
shot-content error, continuity violations at each cut, and deviation from a
target cutting rhythm. That is exactly this problem, so it is implemented rather
than reinvented.

Concretely: candidates are nodes, an edge joins two candidates whose ends meet,
and the objective is

    maximise  sum of shot scores  -  sum of transition costs

Because every candidate begins and ends on a shared cut point, any path through
the graph tiles the horizon by construction — no gap repair is ever needed.

**One documented approximation.** Real film shot lengths are 1/f-correlated
rather than independent, so the rhythm target is a pink-noise sequence and the
k-th shot should be compared to the k-th target. But k is a property of the path,
not of a node, and carrying it in the DP state would multiply the state space by
the shot count. Instead each candidate is compared against the target at
``round(elapsed / median_shot_length)`` — its expected ordinal given where it
sits in the horizon. This preserves the pink-noise structure while keeping the
state space flat; it is an approximation of the exact formulation, and is called
one here rather than presented as equivalent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from assetto_ai_director.application.director.continuity import (
    IMPOSSIBLE,
    ContinuityRules,
    transition_cost,
)
from assetto_ai_director.application.director.rhythm import pink_noise_durations
from assetto_ai_director.application.ports.progress import NullProgress, Stage
from assetto_ai_director.domain.errors import NoCandidatesError, OperationCancelledError
from assetto_ai_director.domain.units import Seconds

if TYPE_CHECKING:
    from assetto_ai_director.application.director.candidates import CandidateGrid
    from assetto_ai_director.application.ports.cancellation import CancelToken
    from assetto_ai_director.application.ports.progress import ProgressReporter
    from assetto_ai_director.domain.shot import ShotCandidate
    from assetto_ai_director.domain.style import StyleProfile

__all__ = ["ScheduleResult", "schedule"]

_SNAP = 1e-6
"""Tolerance for treating two cut points as the same instant."""

_PROGRESS_BATCH = 512
"""Candidates processed between progress ticks.

This is the hot loop — hundreds of thousands of iterations on a full session —
and it is the one place where reporting can cost more than the thing it reports
on. Ticking per candidate would fire a cross-thread Qt signal per iteration and
drown the GUI's event loop, making the run measurably *slower* than reporting
nothing at all.

512 keeps the bar visibly moving on any session large enough for the wait to
matter, while reducing the reporting overhead to a rounding error.
"""


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    """The selected edit and what it cost.

    Attributes:
        shots: The chosen candidates, in order, tiling the horizon.
        total_score: Sum of the selected shots' scores.
        total_cost: Sum of the transition costs paid.
        targets: The rhythm targets the schedule was measured against.
        notes: Anything the user should know about how the edit was reached.
    """

    shots: tuple[ShotCandidate, ...]
    total_score: float
    total_cost: float
    targets: tuple[Seconds, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def objective(self) -> float:
        """The value the scheduler maximised.

        Returns:
            Score minus cost.
        """
        return self.total_score - self.total_cost

    @property
    def durations(self) -> tuple[float, ...]:
        """Every selected shot's duration, in order.

        Returns:
            Durations in seconds.
        """
        return tuple(float(shot.duration) for shot in self.shots)


def schedule(
    grid: CandidateGrid,
    style: StyleProfile,
    horizon: tuple[float, float],
    rules: ContinuityRules | None = None,
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> ScheduleResult:
    """Select the edit that maximises score minus continuity cost.

    Args:
        grid: The candidates to choose from.
        style: The style profile, supplying rhythm parameters and the seed.
        horizon: ``(start, end)`` of the span to cover, seconds.
        rules: How strictly to enforce the editing conventions. Defaults apply
            when omitted.
        progress: Optional reporter. Owns the whole :attr:`Stage.SCHEDULE`
            phase, ticking in batches of :data:`_PROGRESS_BATCH` — see the
            constant for why not per candidate.
        cancel: Optional cancellation token, polled on the same batch boundary.

    Returns:
        The selected shots and the objective they achieved.

    Raises:
        NoCandidatesError: If no chain of candidates can tile the horizon.
        OperationCancelledError: If cancellation is requested part-way through.
    """
    rules = rules or ContinuityRules()
    start, end = horizon
    if not grid.candidates:
        raise NoCandidatesError(style.name, horizon)

    median = max(float(style.rhythm.median_shot_length.value), 0.1)
    expected_shots = max(int((end - start) / median) + 2, 1)
    targets = pink_noise_durations(
        expected_shots,
        Seconds(float(style.rhythm.target_mean)),
        float(style.rhythm.spread.value),
        seed=style.seed,
        minimum=Seconds(float(style.rhythm.min_shot_length.value)),
        maximum=Seconds(float(style.rhythm.max_shot_length.value)),
    )

    by_end = _index_by_end(grid)
    best = [-math.inf] * grid.count
    parent = [-1] * grid.count

    # The null object rather than a None check, exactly as NullProgress's own
    # docstring intends: this is the hot loop, and a guard inside it would be
    # both a branch per iteration and a branch the linter counts against a
    # function whose real complexity is the dynamic program.
    reporter: ProgressReporter = progress if progress is not None else NullProgress()
    reporter.start(str(Stage.SCHEDULE), grid.count)

    order = sorted(range(grid.count), key=lambda i: float(grid.candidates[i].start))
    for step, index in enumerate(order):
        # Both the tick and the cancellation poll ride the same boundary. At 512
        # candidates a batch this is a handful of milliseconds of latency between
        # pressing Cancel and stopping, which is imperceptible; checking every
        # iteration would be measurable in a loop this hot.
        if step and step % _PROGRESS_BATCH == 0:
            reporter.advance(_PROGRESS_BATCH)
            if cancel is not None and cancel.cancelled():
                raise OperationCancelledError(str(Stage.SCHEDULE))
        candidate = grid.candidates[index]
        target = _target_for(candidate, start, median, targets)

        if abs(float(candidate.start) - start) <= _SNAP:
            opening = transition_cost(None, candidate, rules, target)
            if opening < IMPOSSIBLE:
                best[index] = float(candidate.score) - opening
            continue

        for previous_index in by_end.get(round(float(candidate.start), 3), ()):
            if best[previous_index] == -math.inf:
                continue
            cost = transition_cost(grid.candidates[previous_index], candidate, rules, target)
            if cost >= IMPOSSIBLE:
                continue
            value = best[previous_index] + float(candidate.score) - cost
            if value > best[index]:
                best[index] = value
                parent[index] = previous_index

    reporter.finish()

    finishers = [
        index
        for index, candidate in enumerate(grid.candidates)
        if abs(float(candidate.end) - end) <= _SNAP and best[index] > -math.inf
    ]
    if not finishers:
        raise NoCandidatesError(style.name, horizon)

    last = max(finishers, key=lambda index: best[index])
    chain: list[int] = []
    cursor = last
    while cursor >= 0:
        chain.append(cursor)
        cursor = parent[cursor]
    chain.reverse()

    shots = tuple(grid.candidates[index] for index in chain)
    total_score = sum(float(shot.score) for shot in shots)
    return ScheduleResult(
        shots=shots,
        total_score=total_score,
        total_cost=total_score - best[last],
        targets=targets,
    )


def _index_by_end(grid: CandidateGrid) -> dict[float, tuple[int, ...]]:
    """Group candidate indices by the instant they end.

    Args:
        grid: The candidate grid.

    Returns:
        Candidate indices keyed by rounded end time, so a shot's possible
        predecessors are a lookup rather than a scan.
    """
    index: dict[float, list[int]] = {}
    for position, candidate in enumerate(grid.candidates):
        index.setdefault(round(float(candidate.end), 3), []).append(position)
    return {key: tuple(value) for key, value in index.items()}


def _target_for(
    candidate: ShotCandidate, start: float, median: float, targets: tuple[Seconds, ...]
) -> float | None:
    """Find the rhythm target a candidate should be measured against.

    Args:
        candidate: The candidate.
        start: Horizon start, seconds.
        median: The style's median shot length, seconds.
        targets: The sampled pink-noise target sequence.

    Returns:
        The target duration, or ``None`` when there is no sequence to compare to.
    """
    if not targets:
        return None
    ordinal = round((float(candidate.start) - start) / median)
    return float(targets[min(max(ordinal, 0), len(targets) - 1)])
