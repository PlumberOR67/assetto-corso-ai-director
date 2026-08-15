"""Generating the shots the director may choose between.

A candidate is a *possibility*: this rig, this size, this side of the line, over
this span, worth roughly this much. The director generates many and keeps few.

**Candidates are built on a shared grid of cut points, not freely.** The finished
timeline must tile its horizon — consecutive shots meet exactly, with no gaps —
and the cheapest way to guarantee that is to let every candidate begin and end on
one of a fixed set of instants. Then any chain of candidates whose ends meet is a
valid tiling by construction, and the scheduler never has to repair one.

The cut points are the union of two things: every moment an event begins, peaks
or ends, so the director *can* cut on the action; and a regular grid, so it has
somewhere to cut during the quiet stretches that any real session is mostly made
of.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from assetto_ai_director.application.ports.progress import NullProgress, Stage
from assetto_ai_director.domain.camera import RigKind
from assetto_ai_director.domain.errors import OperationCancelledError
from assetto_ai_director.domain.events import EventKind
from assetto_ai_director.domain.shot import LineSide, ShotCandidate, ShotSize
from assetto_ai_director.domain.units import Score, Seconds

if TYPE_CHECKING:
    from collections.abc import Sequence

    from assetto_ai_director.application.ports.cancellation import CancelToken
    from assetto_ai_director.application.ports.progress import ProgressReporter
    from assetto_ai_director.domain.events import Event
    from assetto_ai_director.domain.scoring import ScoreVector, ScoreWeights
    from assetto_ai_director.domain.style import StyleProfile

__all__ = ["CandidateGrid", "build_candidates", "cut_points"]

GRID_SECONDS = 0.5
"""Spacing of the background cut grid.

Fine enough that a shot can end close to where the director wants, coarse enough
that the lattice stays small: halving it quadruples the scheduler's work, because
the search is over pairs of cut points.
"""

MAX_CONFIGS_PER_SPAN = 10
"""How many framings to keep for each (start, end) pair.

The full cross product of rigs, sizes and sides is a few hundred per span and
almost all of it is dominated. Keeping the best handful bounds the lattice
without changing the answer in practice.
"""

_ESTABLISH_SIZES = (ShotSize.LONG, ShotSize.FULL)
_TIGHT_SIZES = (ShotSize.CLOSE_UP, ShotSize.MEDIUM)


@dataclass(frozen=True, slots=True)
class CandidateGrid:
    """The candidates available to the scheduler, indexed by where they start.

    Attributes:
        candidates: Every candidate, ordered by start then end.
        cuts: The cut points candidates are aligned to.
        by_start: Index of the first candidate at each cut point, so the
            scheduler can find a span's successors without scanning.
    """

    candidates: tuple[ShotCandidate, ...]
    cuts: tuple[float, ...]
    by_start: dict[float, tuple[int, ...]]

    @property
    def count(self) -> int:
        """How many candidates were generated.

        Returns:
            The candidate count.
        """
        return len(self.candidates)


def cut_points(
    events: Sequence[Event], start: float, end: float, grid: float = GRID_SECONDS
) -> tuple[float, ...]:
    """Build the set of instants a shot may begin or end on.

    Args:
        events: Detected events, whose boundaries become cut points.
        start: Horizon start, seconds.
        end: Horizon end, seconds.
        grid: Background grid spacing, seconds.

    Returns:
        Sorted, deduplicated cut points spanning the horizon inclusive.
    """
    points = set(np.arange(start, end, grid).tolist())
    points.add(start)
    points.add(end)
    for event in events:
        for moment in (float(event.start), float(event.focus), float(event.end)):
            if start <= moment <= end:
                # Snap to a stable precision so event boundaries that land a
                # microsecond apart do not each spawn their own cut point.
                points.add(round(moment, 3))
    return tuple(sorted(points))


def build_candidates(
    events: Sequence[Event],
    scores: dict[int, list[ScoreVector]],
    time: np.ndarray,
    style: StyleProfile,
    horizon: tuple[float, float],
    *,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> CandidateGrid:
    """Generate every shot the director may choose from.

    Args:
        events: Detected events.
        scores: Per-frame score vectors per car id.
        time: Time per frame, seconds.
        style: The style profile, whose rig permissions and weights apply.
        horizon: ``(start, end)`` of the span to cover, seconds.
        progress: Optional reporter. Owns the whole :attr:`Stage.CANDIDATES`
            phase, ticking once per cut point — the outer loop, whose length is
            known before the work starts.
        cancel: Optional cancellation token, polled once per cut point.

    Returns:
        The candidate grid.

    Raises:
        OperationCancelledError: If cancellation is requested part-way through.
    """
    start, end = horizon
    cuts = cut_points(events, start, end)
    minimum = float(style.rhythm.min_shot_length.value)
    maximum = float(style.rhythm.max_shot_length.value)

    subjects_by_time = _subject_timeline(events, scores)
    candidates: list[ShotCandidate] = []

    reporter: ProgressReporter = progress if progress is not None else NullProgress()
    reporter.start(str(Stage.CANDIDATES), len(cuts))

    for i, span_start in enumerate(cuts):
        if cancel is not None and cancel.cancelled():
            raise OperationCancelledError(str(Stage.CANDIDATES))
        reporter.advance()
        for span_end in cuts[i + 1 :]:
            duration = span_end - span_start
            if duration < minimum:
                continue
            if duration > maximum:
                break
            candidates.extend(
                _configs_for_span(
                    span_start,
                    span_end,
                    events=events,
                    scores=scores,
                    time=time,
                    style=style,
                    subjects=subjects_by_time,
                )
            )

    candidates.sort(key=lambda c: (c.start, c.end))
    by_start: dict[float, list[int]] = {}
    for index, candidate in enumerate(candidates):
        by_start.setdefault(round(float(candidate.start), 3), []).append(index)

    reporter.finish()

    return CandidateGrid(
        candidates=tuple(candidates),
        cuts=cuts,
        by_start={key: tuple(value) for key, value in by_start.items()},
    )


def _subject_timeline(
    events: Sequence[Event], scores: dict[int, list[ScoreVector]]
) -> tuple[int, ...]:
    """Decide which cars are worth filming at all.

    Args:
        events: Detected events.
        scores: Per-frame scores per car.

    Returns:
        Car ids, most eventful first.
    """
    tally: dict[int, int] = dict.fromkeys(scores, 0)
    for event in events:
        for car in event.subjects:
            tally[car] = tally.get(car, 0) + 1
    return tuple(sorted(tally, key=lambda car: -tally[car]))


def _configs_for_span(
    span_start: float,
    span_end: float,
    *,
    events: Sequence[Event],
    scores: dict[int, list[ScoreVector]],
    time: np.ndarray,
    style: StyleProfile,
    subjects: tuple[int, ...],
) -> list[ShotCandidate]:
    """Build the framings worth considering for one span.

    Args:
        span_start: Span start, seconds.
        span_end: Span end, seconds.
        events: Detected events.
        scores: Per-frame scores per car.
        time: Time per frame.
        style: The style profile.
        subjects: Candidate subject cars, most eventful first.

    Returns:
        Up to :data:`MAX_CONFIGS_PER_SPAN` candidates, best first.
    """
    if not subjects:
        return []

    anchor = _dominant_event(events, span_start, span_end)
    primary = anchor.primary_subject if anchor is not None else subjects[0]
    if primary not in scores:
        primary = subjects[0]

    base = _span_score(scores[primary], time, span_start, span_end, style.weights)
    pairing = _pair_subject(anchor, primary)

    built: list[ShotCandidate] = []
    for rig in _rigs_for(style, anchor):
        for size in _sizes_for(rig, anchor):
            for side in _sides_for(rig):
                shot_subjects = (primary, pairing) if pairing is not None else (primary,)
                if len(shot_subjects) > 1 and not rig.can_frame_two_subjects:
                    shot_subjects = (primary,)
                built.append(
                    ShotCandidate(
                        start=Seconds(span_start),
                        end=Seconds(span_end),
                        subjects=shot_subjects,
                        rig=rig,
                        size=size,
                        side=side,
                        score=Score(min(1.0, base * _affinity(rig, size, anchor))),
                        anchor=anchor,
                    )
                )

    return _prune(built)


def _prune(built: list[ShotCandidate]) -> list[ShotCandidate]:
    """Keep the best candidates without collapsing the pool to one framing.

    A plain top-N by score keeps N variants of whatever framing scores highest,
    because score barely varies with rig or side over the same span. The
    scheduler is then offered a pool with no diversity in it and — obeying its
    own jump-cut rule as best it can — alternates two rigs on one side of the
    road for the entire session. The result is mechanically correct and looks
    like a fixed webcam.

    So the prune keeps the best of each ``(rig, side)`` group first, and only
    then fills any remaining budget by score. Continuity rules can only produce
    variety if variety is on the menu.

    Args:
        built: Every framing generated for a span.

    Returns:
        At most :data:`MAX_CONFIGS_PER_SPAN` candidates, best first.
    """
    built.sort(key=lambda c: -c.score)
    kept: list[ShotCandidate] = []
    seen: set[tuple[RigKind, LineSide, ShotSize]] = set()
    for candidate in built:
        key = (candidate.rig, candidate.side, candidate.size)
        if key not in seen:
            seen.add(key)
            kept.append(candidate)
        if len(kept) >= MAX_CONFIGS_PER_SPAN:
            return kept
    for candidate in built:
        if candidate not in kept:
            kept.append(candidate)
        if len(kept) >= MAX_CONFIGS_PER_SPAN:
            break
    return kept


def _dominant_event(events: Sequence[Event], span_start: float, span_end: float) -> Event | None:
    """Find the most significant event overlapping a span.

    Args:
        events: Detected events.
        span_start: Span start, seconds.
        span_end: Span end, seconds.

    Returns:
        The event whose focus falls inside the span with the highest confidence,
        or ``None`` when the span is quiet.
    """
    best: Event | None = None
    best_rank = -1.0
    for event in events:
        if not (span_start <= float(event.focus) <= span_end):
            continue
        rank = event.confidence * (2.0 if event.kind.is_dramatic else 1.0)
        if rank > best_rank:
            best, best_rank = event, rank
    return best


def _pair_subject(anchor: Event | None, primary: int) -> int | None:
    """Find a second car worth keeping in frame.

    Args:
        anchor: The span's dominant event, if any.
        primary: The main subject.

    Returns:
        A second car id, or ``None`` for a single-subject shot.
    """
    if anchor is None or not anchor.kind.is_multi_car:
        return None
    others = [car for car in anchor.subjects if car != primary]
    return others[0] if others else None


def _span_score(
    vectors: list[ScoreVector],
    time: np.ndarray,
    span_start: float,
    span_end: float,
    weights: ScoreWeights,
) -> float:
    """Compute what a span is worth under a style's weighting.

    Uses the span's peak rather than its mean. A shot is chosen for its best
    moment — the instant the drift reaches full angle — and averaging would let a
    long dull span outrank a short brilliant one simply by having fewer bad
    frames to dilute.

    Args:
        vectors: Per-frame scores for the subject car.
        time: Time per frame.
        span_start: Span start, seconds.
        span_end: Span end, seconds.
        weights: The style's weights.

    Returns:
        The span's score in ``[0, 1]``.
    """
    if not vectors or time.shape[0] == 0:
        return 0.0
    first = int(np.searchsorted(time, span_start))
    last = int(np.searchsorted(time, span_end))
    first = max(0, min(first, len(vectors) - 1))
    last = max(first + 1, min(last, len(vectors)))
    return max(float(weights.combine(vector)) for vector in vectors[first:last])


def _rigs_for(style: StyleProfile, anchor: Event | None) -> tuple[RigKind, ...]:
    """Choose which rigs to consider for a span.

    Args:
        style: The style profile, which may forbid rigs outright.
        anchor: The span's dominant event.

    Returns:
        Permitted rigs, weighted toward what suits the event.
    """
    preferred: tuple[RigKind, ...]
    if anchor is None:
        preferred = (RigKind.ROADSIDE, RigKind.FOLLOW, RigKind.DRONE, RigKind.HELICOPTER)
    elif anchor.kind.is_multi_car:
        preferred = (RigKind.ROADSIDE, RigKind.DRONE, RigKind.FOLLOW, RigKind.HELICOPTER)
    elif anchor.kind in (EventKind.DRIFT_START, EventKind.DRIFT_TRANSITION, EventKind.SPIN):
        # A slide has to be readable, and it reads from outside the car where the
        # angle between where it points and where it goes is visible.
        preferred = (RigKind.ROADSIDE, RigKind.STATIC, RigKind.FOLLOW, RigKind.WHEEL)
    else:
        preferred = (RigKind.ROADSIDE, RigKind.FOLLOW, RigKind.BONNET, RigKind.STATIC)
    return tuple(rig for rig in preferred if style.camera_mix.permits(rig))


def _sizes_for(rig: RigKind, anchor: Event | None) -> tuple[ShotSize, ...]:
    """Choose which shot sizes to consider.

    Args:
        rig: The rig in question.
        anchor: The span's dominant event.

    Returns:
        Plausible sizes for this rig and event.
    """
    if rig.is_car_mounted:
        return _TIGHT_SIZES
    if anchor is not None and anchor.kind.is_multi_car:
        return _ESTABLISH_SIZES
    return (ShotSize.FULL, ShotSize.MEDIUM, ShotSize.LONG)


def _sides_for(rig: RigKind) -> tuple[LineSide, ...]:
    """Choose which sides of the line of action to consider.

    Args:
        rig: The rig in question.

    Returns:
        The sides this rig can occupy. Car-mounted rigs look along the direction
        of travel, so they are neutral by construction and cannot cross the line.
    """
    if rig.is_car_mounted:
        return (LineSide.NEUTRAL,)
    return (LineSide.LEFT, LineSide.RIGHT)


def _affinity(rig: RigKind, size: ShotSize, anchor: Event | None) -> float:
    """Score how well a framing suits the moment it covers.

    Args:
        rig: The rig.
        size: The shot size.
        anchor: The span's dominant event.

    Returns:
        A multiplier in ``(0, 1]`` applied to the span's score.
    """
    if anchor is None:
        # Quiet spans still need covering, but should not outrank real events.
        return 0.55 if size in _ESTABLISH_SIZES else 0.4

    bonus = 1.0
    if anchor.kind.is_multi_car and rig.can_frame_two_subjects:
        bonus *= 1.0
    elif anchor.kind.is_multi_car:
        bonus *= 0.5
    if anchor.kind in (EventKind.DRIFT_START, EventKind.DRIFT_TRANSITION) and rig.is_fixed_position:
        # A locked-off frame is what makes a slide legible.
        bonus *= 1.0
    elif anchor.kind in (EventKind.DRIFT_START, EventKind.DRIFT_TRANSITION):
        bonus *= 0.8
    return bonus * anchor.confidence
