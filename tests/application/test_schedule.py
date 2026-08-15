"""Tests for candidate generation and the DP scheduler.

These assert **invariants of the finished edit**, not particular shots. Which
moment the director favours is a matter of taste encoded in weights and will
change; that the edit tiles its horizon, never crosses the line of action and
never jump-cuts are guarantees, and if any of them breaks the output stops being
an edit at all.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from assetto_ai_director.application.director.candidates import build_candidates, cut_points
from assetto_ai_director.application.director.continuity import (
    IMPOSSIBLE,
    ContinuityRules,
    establishing_deficit,
    transition_cost,
)
from assetto_ai_director.application.director.rhythm import spectral_slope
from assetto_ai_director.application.director.schedule import schedule
from assetto_ai_director.application.director.styles import BUILTIN_STYLES
from assetto_ai_director.domain.camera import RigKind
from assetto_ai_director.domain.errors import NoCandidatesError
from assetto_ai_director.domain.events import Event, EventKind
from assetto_ai_director.domain.scoring import ScoreVector
from assetto_ai_director.domain.shot import LineSide, ShotCandidate, ShotSize
from assetto_ai_director.domain.units import Score, Seconds

RATE = 33.3


def _candidate(
    start: float,
    end: float,
    *,
    rig: RigKind = RigKind.ROADSIDE,
    size: ShotSize = ShotSize.FULL,
    side: LineSide = LineSide.LEFT,
    subjects: tuple[int, ...] = (0,),
    score: float = 0.5,
) -> ShotCandidate:
    return ShotCandidate(
        start=Seconds(start),
        end=Seconds(end),
        subjects=subjects,
        rig=rig,
        size=size,
        side=side,
        score=Score(score),
    )


class TestContinuityCosts:
    def test_crossing_the_line_is_forbidden(self) -> None:
        # The one rule priced at infinity: a car that reverses screen direction
        # reads as a different car, and no score is worth that.
        a = _candidate(0.0, 2.0, side=LineSide.LEFT)
        b = _candidate(2.0, 4.0, side=LineSide.RIGHT)
        assert transition_cost(a, b, ContinuityRules()) == IMPOSSIBLE

    def test_neutral_shot_is_a_legal_bridge(self) -> None:
        a = _candidate(0.0, 2.0, side=LineSide.LEFT)
        b = _candidate(2.0, 4.0, rig=RigKind.BONNET, side=LineSide.NEUTRAL)
        assert transition_cost(a, b, ContinuityRules()) < IMPOSSIBLE

    def test_crossing_is_allowed_when_the_style_permits(self) -> None:
        a = _candidate(0.0, 2.0, side=LineSide.LEFT)
        b = _candidate(2.0, 4.0, side=LineSide.RIGHT)
        rules = ContinuityRules(enforce_line=False)
        assert transition_cost(a, b, rules) < IMPOSSIBLE

    def test_different_cars_may_be_framed_from_opposite_sides(self) -> None:
        # Continuity applies between shots of the same subject. Cutting to a
        # different car is a change of subject, not a line crossing.
        a = _candidate(0.0, 2.0, side=LineSide.LEFT, subjects=(0,))
        b = _candidate(2.0, 4.0, side=LineSide.RIGHT, subjects=(1,))
        assert transition_cost(a, b, ContinuityRules()) < IMPOSSIBLE

    def test_identical_framing_is_penalised(self) -> None:
        a = _candidate(0.0, 2.0)
        identical = _candidate(2.0, 4.0)
        distinct = _candidate(2.0, 4.0, rig=RigKind.DRONE, size=ShotSize.CLOSE_UP)
        rules = ContinuityRules()
        assert transition_cost(a, identical, rules) > transition_cost(a, distinct, rules)

    def test_opening_on_a_tight_shot_is_penalised(self) -> None:
        rules = ContinuityRules()
        tight = _candidate(0.0, 2.0, size=ShotSize.CLOSE_UP)
        wide = _candidate(0.0, 2.0, size=ShotSize.LONG)
        assert transition_cost(None, tight, rules) > transition_cost(None, wide, rules)

    def test_rhythm_deviation_costs_more_further_from_target(self) -> None:
        rules = ContinuityRules()
        on_target = _candidate(2.0, 4.0)
        way_off = _candidate(2.0, 10.0)
        assert transition_cost(None, way_off, rules, target_duration=2.0) > transition_cost(
            None, on_target, rules, target_duration=2.0
        )


class TestCutPoints:
    def test_spans_the_horizon(self) -> None:
        points = cut_points([], 0.0, 10.0)
        assert points[0] == 0.0
        assert points[-1] == 10.0

    def test_event_boundaries_become_cut_points(self) -> None:
        # The director must be able to cut *on* the action, not merely near it.
        event = Event(EventKind.APEX, Seconds(3.21), Seconds(4.0), (0,), peak=Seconds(3.6))
        points = cut_points([event], 0.0, 10.0)
        assert 3.21 in points
        assert 3.6 in points

    def test_points_are_sorted_and_unique(self) -> None:
        event = Event(EventKind.APEX, Seconds(2.0), Seconds(4.0), (0,))
        points = cut_points([event], 0.0, 10.0)
        assert list(points) == sorted(set(points))


def _session(
    duration: float = 30.0, events: list[Event] | None = None
) -> tuple[list[Event], dict[int, list[ScoreVector]], np.ndarray]:
    count = int(duration * RATE)
    time = np.arange(count) / RATE
    events = events if events is not None else _default_events()
    scores = {
        0: [
            ScoreVector(
                drift=Score(0.8 if 300 < i < 500 else 0.05),
                speed=Score(0.5),
                risk=Score(0.2),
                flow=Score(0.6),
                cinematic=Score(0.4),
            )
            for i in range(count)
        ]
    }
    return events, scores, time


def _default_events() -> list[Event]:
    return [
        Event(
            EventKind.DRIFT_START,
            Seconds(9.0),
            Seconds(15.0),
            (0,),
            peak=Seconds(12.0),
            confidence=0.9,
        ),
        Event(EventKind.APEX, Seconds(20.0), Seconds(23.0), (0,), peak=Seconds(21.5)),
    ]


class TestScheduling:
    def test_edit_tiles_the_horizon_without_gaps(self) -> None:
        # The defining property. A gap would mean a moment with no camera
        # assigned, which an edit cannot contain.
        events, scores, time = _session()
        style = BUILTIN_STYLES["initial_d"]
        grid = build_candidates(events, scores, time, style, (0.0, 30.0))
        result = schedule(grid, style, (0.0, 30.0))

        assert float(result.shots[0].start) == pytest.approx(0.0)
        assert float(result.shots[-1].end) == pytest.approx(30.0)
        for earlier, later in pairwise(result.shots):
            assert float(later.start) == pytest.approx(float(earlier.end), abs=1e-6)

    def test_no_selected_cut_crosses_the_line(self) -> None:
        events, scores, time = _session()
        style = BUILTIN_STYLES["initial_d"]
        grid = build_candidates(events, scores, time, style, (0.0, 30.0))
        result = schedule(grid, style, (0.0, 30.0))
        for earlier, later in pairwise(result.shots):
            if earlier.shares_subject_with(later):
                assert not earlier.side.crosses_line_with(later.side)

    def test_shot_lengths_respect_the_style_bounds(self) -> None:
        events, scores, time = _session()
        style = BUILTIN_STYLES["initial_d"]
        grid = build_candidates(events, scores, time, style, (0.0, 30.0))
        result = schedule(grid, style, (0.0, 30.0))
        floor = float(style.rhythm.min_shot_length.value)
        ceiling = float(style.rhythm.max_shot_length.value)
        for shot in result.shots:
            assert floor - 1e-6 <= float(shot.duration) <= ceiling + 1e-6

    def test_tight_shots_are_established_first(self) -> None:
        events, scores, time = _session()
        style = BUILTIN_STYLES["initial_d"]
        grid = build_candidates(events, scores, time, style, (0.0, 30.0))
        result = schedule(grid, style, (0.0, 30.0))
        assert establishing_deficit(list(result.shots)) == 0

    def test_the_edit_is_reproducible(self) -> None:
        # Two runs on the same session with the same seed must agree, or an edit
        # cannot be reviewed, shared or regression-tested.
        events, scores, time = _session()
        style = BUILTIN_STYLES["initial_d"]
        grid = build_candidates(events, scores, time, style, (0.0, 30.0))
        first = schedule(grid, style, (0.0, 30.0))
        second = schedule(grid, style, (0.0, 30.0))
        assert first.durations == second.durations

    def test_the_dp_beats_a_greedy_longest_shot_baseline(self) -> None:
        # If a trivial baseline matched it, the dynamic program would not be
        # earning its complexity.
        events, scores, time = _session()
        style = BUILTIN_STYLES["initial_d"]
        grid = build_candidates(events, scores, time, style, (0.0, 30.0))
        result = schedule(grid, style, (0.0, 30.0))

        rules = ContinuityRules()
        greedy: list[ShotCandidate] = []
        cursor = 0.0
        while cursor < 30.0 - 1e-6:
            options = [
                c
                for c in grid.candidates
                if abs(float(c.start) - cursor) < 1e-6 and float(c.end) <= 30.0 + 1e-6
            ]
            if not options:
                break
            pick = max(options, key=lambda c: float(c.score))
            greedy.append(pick)
            cursor = float(pick.end)

        if greedy and abs(cursor - 30.0) < 1e-6:
            objective = 0.0
            previous: ShotCandidate | None = None
            for shot in greedy:
                cost = transition_cost(previous, shot, rules)
                objective += float(shot.score) - (0.0 if cost == IMPOSSIBLE else cost)
                previous = shot
            assert result.objective >= objective

    def test_rhythm_is_not_white_noise(self) -> None:
        # The whole point of the pink-noise target: an edit whose shot lengths
        # are independent has the right average and feels mechanically wrong.
        events, scores, time = _session(duration=180.0)
        style = BUILTIN_STYLES["initial_d"]
        grid = build_candidates(events, scores, time, style, (0.0, 180.0))
        result = schedule(grid, style, (0.0, 180.0))
        if len(result.shots) >= 32:
            slope = spectral_slope(result.durations)
            assert slope is not None
            assert slope < 0.0

    def test_empty_candidate_grid_is_refused(self) -> None:
        events, scores, time = _session()
        style = BUILTIN_STYLES["initial_d"]
        grid = build_candidates(events, scores, time, style, (0.0, 0.0))
        with pytest.raises(NoCandidatesError):
            schedule(grid, style, (0.0, 0.0))


class TestStyleDifferences:
    def test_styles_produce_different_edits(self) -> None:
        # If every style yielded the same cut, styles would be decoration.
        events, scores, time = _session(duration=60.0)
        edits = {}
        for name in ("initial_d", "fast_furious", "best_motoring"):
            style = BUILTIN_STYLES[name]
            grid = build_candidates(events, scores, time, style, (0.0, 60.0))
            edits[name] = schedule(grid, style, (0.0, 60.0)).durations
        assert len({tuple(v) for v in edits.values()}) > 1

    def test_a_faster_style_cuts_more(self) -> None:
        events, scores, time = _session(duration=60.0)
        fast = BUILTIN_STYLES["fast_furious"]
        slow = BUILTIN_STYLES["best_motoring"]
        fast_grid = build_candidates(events, scores, time, fast, (0.0, 60.0))
        slow_grid = build_candidates(events, scores, time, slow, (0.0, 60.0))
        fast_shots = len(schedule(fast_grid, fast, (0.0, 60.0)).shots)
        slow_shots = len(schedule(slow_grid, slow, (0.0, 60.0)).shots)
        assert fast_shots > slow_shots

    def test_initial_d_forbids_drones(self) -> None:
        events, scores, time = _session()
        style = BUILTIN_STYLES["initial_d"]
        grid = build_candidates(events, scores, time, style, (0.0, 30.0))
        result = schedule(grid, style, (0.0, 30.0))
        assert all(shot.rig is not RigKind.DRONE for shot in result.shots)
