"""Tests for the event detectors.

Two layers. The synthetic tests build signals whose answer is known by
construction — a car placed on an exact circle has exactly known curvature — so
they check correctness rather than plausibility. The fixture tests then confirm
the detectors produce sane counts on a real touge run, which is the only way to
catch a detector that is mathematically right and practically useless.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from assetto_ai_director.application.events.corner import (
    CORNER_ENTRY_CURVATURE,
    detect_corners,
)
from assetto_ai_director.application.events.detect import CarSignals, detect_all
from assetto_ai_director.application.events.drift import (
    DRIFT_ENTRY_DEGREES,
    SPIN_DEGREES,
    detect_drifts,
    detect_transitions,
)
from assetto_ai_director.application.events.overtake import detect_overtakes
from assetto_ai_director.application.features.kinematics import (
    arc_length,
    curvature_by_arc_length,
    yaw_rate,
)
from assetto_ai_director.domain.events import EventKind

RATE = 33.3


def _straight(seconds: float, speed: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Build a car travelling straight along +z."""
    count = int(seconds * RATE)
    z = np.arange(count) * speed / RATE
    position = np.stack([np.zeros(count), np.zeros(count), z], axis=1)
    heading = np.zeros(count)
    return position, heading


def _circle(seconds: float, radius: float, speed: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Build a car on an exact circle, whose curvature is 1/radius by construction."""
    count = int(seconds * RATE)
    omega = speed / radius
    angle = np.arange(count) * omega / RATE
    position = np.stack([radius * np.cos(angle), np.zeros(count), radius * np.sin(angle)], axis=1)
    # Tangent to the circle, in the atan2(x, z) convention.
    heading = np.arctan2(-np.sin(angle), np.cos(angle))
    return position, heading


class TestKinematics:
    def test_circle_curvature_matches_its_radius(self) -> None:
        # The decisive correctness check: a 50 m circle must read 0.02 1/m.
        position, _ = _circle(20.0, radius=50.0)
        curvature = curvature_by_arc_length(position)
        settled = np.abs(curvature[20:-20])
        assert float(np.median(settled)) == pytest.approx(1.0 / 50.0, rel=0.1)

    def test_straight_has_no_curvature(self) -> None:
        position, _ = _straight(10.0)
        assert float(np.abs(curvature_by_arc_length(position)).max()) < 1e-6

    @pytest.mark.parametrize("speed_kmh", [40.0, 90.0, 119.0, 121.0, 140.0, 200.0])
    def test_curvature_is_speed_independent(self, speed_kmh: float) -> None:
        """The property the docstring claims, now actually tested.

        The original implementation snapped the arc-length window to whole
        samples, so above 120 km/h at 33 Hz the window collapsed and curvature
        was silently forced to zero. That failure was invisible: a straight and a
        fast corner both reported nothing.
        """
        position, _ = _circle(20.0, radius=50.0, speed=speed_kmh / 3.6)
        settled = np.abs(curvature_by_arc_length(position)[20:-20])
        assert float(np.median(settled)) == pytest.approx(1.0 / 50.0, rel=0.15)

    def test_arc_length_matches_distance_travelled(self) -> None:
        position, _ = _straight(10.0, speed=30.0)
        assert float(arc_length(position)[-1]) == pytest.approx(30.0 * 10.0, rel=0.02)

    def test_yaw_rate_of_a_circle_is_constant(self) -> None:
        _, heading = _circle(20.0, radius=50.0, speed=30.0)
        rate = yaw_rate(heading, RATE)
        expected = 30.0 / 50.0  # v / r
        assert float(np.median(np.abs(rate[20:-20]))) == pytest.approx(expected, rel=0.1)

    def test_yaw_rate_survives_the_branch_cut(self) -> None:
        # Heading wraps several times a lap. Differentiating without unwrapping
        # produces a huge spike that reads as a violent flick.
        _, heading = _circle(40.0, radius=30.0, speed=30.0)
        rate = yaw_rate(heading, RATE)
        assert float(np.abs(rate).max()) < 5.0


class TestDriftDetection:
    def _sideslip(self, count: int, spans: list[tuple[int, int, float]]) -> np.ndarray:
        signal = np.zeros(count)
        for start, end, degrees in spans:
            signal[start:end] = math.radians(degrees)
        return signal

    def test_sustained_slide_is_detected(self) -> None:
        count = int(10 * RATE)
        sideslip = self._sideslip(count, [(100, 200, 25.0)])
        speed = np.full(count, 30.0)
        time = np.arange(count) / RATE
        events = detect_drifts(0, sideslip, speed, time, RATE)
        assert len(events) == 1
        assert events[0].kind is EventKind.DRIFT_START

    def test_brief_twitch_is_rejected(self) -> None:
        count = int(10 * RATE)
        sideslip = self._sideslip(count, [(100, 103, 25.0)])
        speed = np.full(count, 30.0)
        time = np.arange(count) / RATE
        assert detect_drifts(0, sideslip, speed, time, RATE) == []

    def test_slow_rotation_is_not_a_drift(self) -> None:
        # A car rotating in a gravel trap at walking pace is not a drift.
        count = int(10 * RATE)
        sideslip = self._sideslip(count, [(100, 250, 40.0)])
        speed = np.full(count, 2.0)
        time = np.arange(count) / RATE
        assert detect_drifts(0, sideslip, speed, time, RATE) == []

    def test_spin_is_classified_separately(self) -> None:
        count = int(10 * RATE)
        sideslip = self._sideslip(count, [(100, 250, SPIN_DEGREES + 20.0)])
        speed = np.full(count, 30.0)
        time = np.arange(count) / RATE
        events = detect_drifts(0, sideslip, speed, time, RATE)
        assert [event.kind for event in events] == [EventKind.SPIN]

    def test_hysteresis_keeps_one_slide_whole(self) -> None:
        # A slide that dips just below the entry threshold mid-way must stay one
        # event, not fragment into two. Without hysteresis it fragments.
        count = int(10 * RATE)
        sideslip = np.zeros(count)
        sideslip[100:200] = math.radians(DRIFT_ENTRY_DEGREES + 5.0)
        sideslip[150:160] = math.radians(DRIFT_ENTRY_DEGREES - 2.0)
        speed = np.full(count, 30.0)
        time = np.arange(count) / RATE
        assert len(detect_drifts(0, sideslip, speed, time, RATE)) == 1

    def test_peak_is_marked_where_the_angle_peaks(self) -> None:
        count = int(10 * RATE)
        sideslip = np.zeros(count)
        sideslip[100:200] = math.radians(20.0)
        sideslip[180] = math.radians(45.0)
        speed = np.full(count, 30.0)
        time = np.arange(count) / RATE
        event = detect_drifts(0, sideslip, speed, time, RATE)[0]
        assert float(event.focus) == pytest.approx(180 / RATE, abs=0.1)


class TestTransitionDetection:
    def test_straight_line_yields_no_transitions(self) -> None:
        position, heading = _straight(10.0)
        count = position.shape[0]
        events = detect_transitions(
            0,
            np.zeros(count),
            heading,
            np.full(count, 30.0),
            time=np.arange(count) / RATE,
            sample_rate=RATE,
        )
        assert events == []

    def test_mild_slip_does_not_count_as_a_transition(self) -> None:
        # Below the drift entry threshold this is a steering correction, not a
        # linked drift, however cleanly the yaw rate reverses.
        count = int(8 * RATE)
        heading = np.concatenate(
            [np.linspace(0.0, 1.0, count // 2), np.linspace(1.0, 0.0, count - count // 2)]
        )
        sideslip = np.full(count, math.radians(DRIFT_ENTRY_DEGREES - 3.0))
        events = detect_transitions(
            0,
            sideslip,
            heading,
            np.full(count, 30.0),
            time=np.arange(count) / RATE,
            sample_rate=RATE,
        )
        assert events == []

    def test_spin_is_not_reported_as_a_transition(self) -> None:
        # A spin reverses yaw rate too, but it is the loss of control, not the
        # signature move.
        count = int(8 * RATE)
        heading = np.concatenate(
            [np.linspace(0.0, 2.0, count // 2), np.linspace(2.0, 0.0, count - count // 2)]
        )
        sideslip = np.full(count, math.radians(SPIN_DEGREES + 30.0))
        events = detect_transitions(
            0,
            sideslip,
            heading,
            np.full(count, 30.0),
            time=np.arange(count) / RATE,
            sample_rate=RATE,
        )
        assert events == []


class TestCornerDetection:
    def test_circle_is_one_corner_with_the_right_radius(self) -> None:
        position, heading = _circle(12.0, radius=40.0)
        count = position.shape[0]
        events = detect_corners(
            0,
            position,
            heading,
            np.full(count, 30.0),
            time=np.arange(count) / RATE,
            sample_rate=RATE,
        )
        apexes = [event for event in events if event.kind is EventKind.APEX]
        assert len(apexes) == 1
        assert apexes[0].detail["min_radius_m"] == pytest.approx(40.0, rel=0.15)

    def test_straight_has_no_corners(self) -> None:
        position, heading = _straight(15.0)
        count = position.shape[0]
        events = detect_corners(
            0,
            position,
            heading,
            np.full(count, 30.0),
            time=np.arange(count) / RATE,
            sample_rate=RATE,
        )
        assert events == []

    def test_gentle_bend_below_threshold_is_not_a_corner(self) -> None:
        # A 200 m radius is a straight with a kink, not a corner.
        position, heading = _circle(12.0, radius=1.0 / (CORNER_ENTRY_CURVATURE * 0.25))
        count = position.shape[0]
        events = detect_corners(
            0,
            position,
            heading,
            np.full(count, 30.0),
            time=np.arange(count) / RATE,
            sample_rate=RATE,
        )
        assert events == []

    def test_entry_apex_exit_are_ordered(self) -> None:
        position, heading = _circle(12.0, radius=40.0)
        count = position.shape[0]
        events = detect_corners(
            0,
            position,
            heading,
            np.full(count, 30.0),
            time=np.arange(count) / RATE,
            sample_rate=RATE,
        )
        by_kind = {event.kind: event for event in events}
        assert by_kind[EventKind.CORNER_ENTRY].focus <= by_kind[EventKind.APEX].focus
        assert by_kind[EventKind.APEX].focus <= by_kind[EventKind.CORNER_EXIT].focus


class TestOvertakeDetection:
    def test_pass_is_detected_with_the_right_direction(self) -> None:
        count = int(10 * RATE)
        time = np.arange(count) / RATE
        # Car 0 holds station; car 1 starts behind and ends ahead.
        z0 = np.arange(count) * 30.0 / RATE
        z1 = z0 - 10.0 + np.arange(count) * 2.0 / RATE
        first = np.stack([np.zeros(count), np.zeros(count), z0], axis=1)
        second = np.stack([np.full(count, 3.0), np.zeros(count), z1], axis=1)
        heading = np.zeros(count)
        events = detect_overtakes(
            {0: first, 1: second},
            {0: heading, 1: heading},
            {0: np.full(count, 30.0), 1: np.full(count, 32.0)},
            time,
            RATE,
        )
        assert len(events) == 1
        assert events[0].subjects == (1, 0)

    def test_distant_cars_do_not_produce_an_event(self) -> None:
        count = int(10 * RATE)
        time = np.arange(count) / RATE
        z0 = np.arange(count) * 30.0 / RATE
        z1 = z0 - 10.0 + np.arange(count) * 2.0 / RATE
        first = np.stack([np.zeros(count), np.zeros(count), z0], axis=1)
        # Same along-track crossing, but 200 m to the side: not an interaction.
        second = np.stack([np.full(count, 200.0), np.zeros(count), z1], axis=1)
        heading = np.zeros(count)
        events = detect_overtakes(
            {0: first, 1: second},
            {0: heading, 1: heading},
            {0: np.full(count, 30.0), 1: np.full(count, 32.0)},
            time,
            RATE,
        )
        assert events == []


class TestDetectAll:
    def test_parked_cars_are_skipped_and_reported(self) -> None:
        count = int(10 * RATE)
        time = np.arange(count) / RATE
        moving_pos, moving_head = _circle(10.0, radius=40.0)
        parked = np.zeros((count, 3))
        signals = {
            0: CarSignals(0, moving_pos, moving_head, np.full(count, 30.0), np.zeros(count)),
            1: CarSignals(1, parked, np.zeros(count), np.zeros(count), np.zeros(count)),
        }
        result = detect_all(signals, time, RATE)
        assert any("never moved" in note for note in result.notes)
        assert all(event.involves(0) for event in result.events)

    def test_undetectable_channels_are_always_disclosed(self) -> None:
        count = int(5 * RATE)
        position, heading = _straight(5.0)
        signals = {0: CarSignals(0, position, heading, np.full(count, 30.0), np.zeros(count))}
        result = detect_all(signals, np.arange(count) / RATE, RATE)
        assert any("Not detected" in note for note in result.notes)


REPLAY_DIR = Path(__file__).resolve().parents[2] / "replays"
FIXTURES = sorted(REPLAY_DIR.glob("*.acreplay")) if REPLAY_DIR.is_dir() else []


@pytest.mark.fixtures
@pytest.mark.skipif(not FIXTURES, reason="no .acreplay fixtures in replays/")
class TestOnRealReplay:
    def test_touge_run_produces_sane_event_counts(self) -> None:
        from assetto_ai_director.infrastructure.acreplay.frames import (  # noqa: PLC0415
            read_car_frames,
        )
        from assetto_ai_director.infrastructure.acreplay.structure import (  # noqa: PLC0415
            CarBlock,
            read_structure,
        )

        structure = read_structure(FIXTURES[0])
        rate = float(structure.sample_rate)
        frames = min(int(180 * rate), structure.cars[0].frame_count)
        block = structure.cars[0]
        decoded = read_car_frames(
            FIXTURES[0],
            CarBlock(
                identity=block.identity,
                frame_count=frames,
                wing_count=block.wing_count,
                frames_offset=block.frames_offset,
                extra_count=block.extra_count,
            ),
            rate,
        )
        signals = {
            0: CarSignals(0, decoded.position, decoded.heading, decoded.speed, decoded.sideslip)
        }
        result = detect_all(signals, np.arange(frames) / rate, rate)
        counts = result.counts

        # Akina downhill is a sequence of hairpins: three minutes must contain
        # corners, and an AE86 driven down it must drift.
        assert counts.get(EventKind.APEX, 0) > 5
        assert counts.get(EventKind.DRIFT_START, 0) > 0
        # Every corner yields exactly one entry, one apex and one exit.
        assert counts.get(EventKind.CORNER_ENTRY) == counts.get(EventKind.APEX)
        assert counts.get(EventKind.CORNER_EXIT) == counts.get(EventKind.APEX)
